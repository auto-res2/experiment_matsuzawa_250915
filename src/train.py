"""train.py – model construction, training loop, (optional) hardware-energy phase
Strictly self-contained; obeys the constraints defined in the root README.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from ogb.nodeproppred import Evaluator
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from tqdm import tqdm

from .preprocess import load_ogbn_papers400m
from .models import CraftGNN
from .metrics import accuracy, logdet_jacobian, macro_f1, row_diff

# heavy / optional deps – imported lazily to keep the smoke-test light-weight
try:
    import optuna  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore
    from fvcore.nn import FlopCountAnalysis  # noqa: F401 – optional heavy dep
except ModuleNotFoundError:  # pragma: no cover (CI may skip optional deps)
    optuna = None  # type: ignore
    plt, sns = None, None  # type: ignore

# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_model(cfg: Dict[str, Any], in_dim: int, num_classes: int) -> torch.nn.Module:
    if cfg["model"]["arch"].lower() != "craft":
        raise ValueError("Only 'craft' architecture is supported in this reference implementation.")
    return CraftGNN(
        in_dim=in_dim,
        hid_dim=cfg["model"]["hid_dim"],
        num_classes=num_classes,
        num_layers=cfg["model"]["num_layers"],
        cheb_K=cfg["model"]["cheb_K"],
        pool_size=cfg["model"]["pool_size"],
        dp_sigma=cfg["model"]["dp_sigma"],
        curvature_lambda=cfg["model"]["curvature_lambda"],
        energy_lambda=cfg["model"]["energy_lambda"],
    )


# ---------------------------------------------------------------------------
# Trainer – single-GPU / FSDP full-batch implementation
# ---------------------------------------------------------------------------
class Trainer:
    """Full-batch trainer for OGBN-papers400M (or a 50 k-node subset in smoke-test)."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------- data ----------------------------------------------------
        g, splits = load_ogbn_papers400m(
            cfg["dataset"]["root"], cfg["dataset"].get("use_subset", False)
        )
        self.g = g.to(self.device)
        self.splits = splits
        self.num_classes: int = int(self.g.ndata["label"].max() + 1)

        # ---------- model ---------------------------------------------------
        self.model = _build_model(
            cfg, in_dim=self.g.ndata["feat"].shape[1], num_classes=self.num_classes
        ).to(self.device)

        # optional FSDP if multi-GPU available
        if torch.cuda.device_count() > 1:
            self.model = FSDP(
                self.model,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                auto_wrap_policy=transformer_auto_wrap_policy,
                mixed_precision=True,
            )

        # ---------- optimisation -------------------------------------------
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["train"]["lr"],
            betas=(0.9, 0.95),
            weight_decay=5e-4,
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=cfg["train"]["epochs"]
        )
        self.evaluator = Evaluator("ogbn-papers400m")

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        out, aux = self.model(self.g, self.g.ndata["feat"])
        loss: torch.Tensor = F.cross_entropy(
            out[self.splits["train"]], self.g.ndata["label"][self.splits["train"]]
        )
        loss = loss + aux["curvature"] + aux["energy"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg["train"].get("clip_grad", 1.0)
        )
        self.opt.step()
        self.sched.step()
        return float(loss.item())

    @torch.no_grad()
    def _evaluate(self, split: str = "valid") -> Dict[str, float]:
        self.model.eval()
        logits, _ = self.model(self.g, self.g.ndata["feat"])
        idx = self.splits[split]
        y_true = self.g.ndata["label"][idx]
        y_pred = logits[idx]
        return {
            "acc": accuracy(y_pred, y_true),
            "f1": macro_f1(y_pred, y_true, self.num_classes),
            "rowdiff": row_diff(self.g, logits),
            "logdetJ": logdet_jacobian(
                lambda _g, _x: self.model(_g, _x)[0],
                self.g,
                self.g.ndata["feat"][:2048],
            ),
        }

    # ------------------------------------------------------------------
    def fit(self) -> tuple[List[Dict[str, float]], Dict[str, float]]:
        history: List[Dict[str, float]] = []
        for epoch in range(1, self.cfg["train"]["epochs"] + 1):
            loss_val = self._train_epoch(epoch)
            if epoch % self.cfg.get("eval_every", 1) == 0:
                metrics = self._evaluate("valid")
                print(
                    f"[Epoch {epoch:03d}] loss={loss_val:.5f}  acc={metrics['acc']:.4f}  "
                    f"f1={metrics['f1']:.4f}"
                )
                history.append({"epoch": epoch, **metrics})
        test_metrics = self._evaluate("test")
        return history, test_metrics


# ---------------------------------------------------------------------------
# (Optional) hyper-parameter optimisation via Optuna
# ---------------------------------------------------------------------------

def _run_hyperopt(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if optuna is None:
        raise RuntimeError("Optuna is not installed – cannot run hyper-parameter optimisation.")

    def _objective(trial: optuna.trial.Trial):  # type: ignore
        new_cfg = yaml.safe_load(yaml.dump(cfg))  # deep copy via YAML round-trip
        new_cfg["train"]["lr"] = trial.suggest_loguniform("lr", 1e-4, 5e-3)  # type: ignore
        new_cfg["model"]["dp_sigma"] = trial.suggest_float("sigma", 0.0, 1.0)
        new_cfg["model"]["curvature_lambda"] = trial.suggest_float("lam_c", 0.0, 1.0)
        new_cfg["model"]["energy_lambda"] = trial.suggest_float("lam_e", 0.0, 1.0)
        trainer = Trainer(new_cfg)
        _, val = trainer.fit()
        trial.report(val["f1"], step=0)
        return 1 - val["f1"]  # minimise error

    study = optuna.create_study(  # type: ignore
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        _objective,
        n_trials=cfg["hyperopt"]["trials"],
        n_jobs=cfg["hyperopt"].get("parallel", 1),
    )
    print("Best hyper-parameters:", study.best_params)

    # update cfg in-place – downstream runner will use tuned values
    cfg["train"]["lr"] = study.best_params["lr"]
    cfg["model"]["dp_sigma"] = study.best_params["sigma"]
    cfg["model"]["curvature_lambda"] = study.best_params["lam_c"]
    cfg["model"]["energy_lambda"] = study.best_params["lam_e"]
    return cfg


# ---------------------------------------------------------------------------
# Hardware-energy evaluation (heavy dependency path – executed only when asked)
# ---------------------------------------------------------------------------

def run_hardware_eval(cfg_path: str) -> None:  # noqa: C901 – large but self-contained
    """Compile for targets listed in the YAML and measure average power draw."""
    import subprocess
    import yaml as _yaml
    import numpy as _np
    from pathlib import Path as _Path

    print("[HARDWARE] starting hardware-energy evaluation …")
    cfg = _yaml.safe_load(open(cfg_path, "r"))
    if "hardware_eval" not in cfg:
        print("[HARDWARE] section missing – skipping")
        return

    # tiny example graph for shape tracing
    g, _ = load_ogbn_papers400m(cfg["dataset"]["root"], use_subset=True)
    model = _build_model(cfg, g.ndata["feat"].shape[1], int(g.ndata["label"].max() + 1))
    model.eval()
    example_in = g.ndata["feat"][:1024]

    # TVM imports are heavy – do them only here
    try:
        import tvm  # type: ignore
        from tvm import relay  # type: ignore
    except ModuleNotFoundError:
        raise RuntimeError("TVM is required for hardware evaluation but is not installed.")

    results: Dict[str, Any] = {}

    def _compile(target: str, cache_kb: int) -> _Path:
        mod, params = relay.frontend.from_pytorch(model, [("input", example_in.shape)])  # type: ignore
        with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
            lib = relay.build(mod, target=target, params=params)  # type: ignore
        out_dir = _Path("artifacts") / f"{target}_{cache_kb}KB"
        out_dir.mkdir(parents=True, exist_ok=True)
        lib_path = out_dir / "model.so"
        lib.export_library(lib_path)
        return lib_path

    def _measure_power(cmd: List[str], log_file: _Path) -> Dict[str, float]:
        with open(log_file, "w") as f:
            smi = subprocess.Popen(  # noqa: S603,S607 – trusted binary
                [
                    "nvidia-smi",
                    "--loop-ms=100",
                    "--format=csv,noheader,nounits",
                    "--query-gpu=power.draw",
                ],
                stdout=f,
            )
            t0 = time.time()
            subprocess.run(cmd, check=True)
            t1 = time.time()
            smi.terminate()
        data = _np.loadtxt(log_file)
        mean_power = float(data.mean())
        return {"mean_power_W": mean_power, "energy_J": mean_power * (t1 - t0)}

    for tgt, settings in cfg["hardware_eval"].items():
        for kb in settings["cache_kb"]:
            so_path = _compile(tgt, kb)
            power_log = _Path("artifacts") / f"{tgt}_{kb}KB" / "power.txt"
            res = _measure_power(
                ["python", "-c", "import ctypes,sys;ctypes.CDLL(sys.argv[1])", str(so_path)],
                power_log,
            )
            results[f"{tgt}_{kb}KB"] = res

    out_path = Path(cfg.get("save_dir", ".research/iteration6/results")) / "hardware_energy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)
    print("[HARDWARE] results saved to", out_path)
    print(json.dumps(results, indent=2))


# ---------------------------------------------------------------------------
# Public API – called from src.main
# ---------------------------------------------------------------------------

def run(cfg_file: str) -> None:
    cfg = yaml.safe_load(open(cfg_file, "r"))

    # run Optuna if requested
    if cfg.get("hyperopt") and optuna is not None:
        cfg = _run_hyperopt(cfg)

    base_out = Path(cfg.get("save_dir", ".research/iteration6"))
    base_out.mkdir(parents=True, exist_ok=True)

    for seed in cfg["seeds"]:
        _set_seed(seed)
        trainer = Trainer(cfg)
        history, test_metrics = trainer.fit()

        # ------------------ saving ----------------------------------------
        run_dir = base_out / cfg["run_name"] / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        out_json = run_dir / "metrics.json"
        with open(out_json, "w") as f:
            json.dump({"history": history, "test": test_metrics}, f, indent=2)

        # print full JSON for verification (mandatory for CI)
        print("===== JSON RESULT (seed", seed, ") =====")
        print(json.dumps({"history": history, "test": test_metrics}, indent=2))

        # ------------------ quick plots -----------------------------------
        if plt is not None and sns is not None:
            images_dir = Path(".research/iteration6/images")
            images_dir.mkdir(parents=True, exist_ok=True)
            epochs = [h["epoch"] for h in history]
            accs = [h["acc"] for h in history]
            f1s = [h["f1"] for h in history]
            sns.set()
            plt.figure()
            plt.plot(epochs, accs, label="Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(images_dir / f"accuracy_seed{seed}.pdf")
            plt.close()

            plt.figure()
            plt.plot(epochs, f1s, label="Macro-F1")
            plt.legend()
            plt.tight_layout()
            plt.savefig(images_dir / f"macroF1_seed{seed}.pdf")
            plt.close()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
