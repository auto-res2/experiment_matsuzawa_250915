import argparse
import yaml
from pathlib import Path
import traceback

from . import preprocess as prep
from . import train as trn
from . import evaluate as evl

CFG_DIR = Path(__file__).parent.parent / "config"


def _load_cfg(which: str):
    cfg_path = CFG_DIR / which
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    cfg["_name"] = which.stem  # attach for book-keeping
    return cfg


def _run(cfg):
    # 1) Verify dataset availability
    prep.prepare_datasets(cfg)

    # 2) Model loading / (TinyFormer) training
    sd_pipe, aux_models = trn.load_models(cfg)
    trn.train_tinyformer(cfg)

    # 3) Minimal evaluation – generate a preview grid for smoke-test.
    if cfg.get("task", "diffusion") == "diffusion":
        prompts = cfg["prompts"]
        evl.evaluate_diffusion(sd_pipe, prompts, cfg)
    else:
        raise NotImplementedError("Only diffusion preview is wired up at the moment.")


# -----------------------------------------------------------------------------
# CLI ENTRY POINT
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="PHOENIX-RELAX experiment runner")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true", help="run quick validation")
    g.add_argument("--full-experiment", action="store_true", help="run full experiment")
    args = ap.parse_args()

    try:
        if args.smoke_test:
            cfg = _load_cfg(Path("smoke_test.yaml"))
            _run(cfg)
        elif args.full_experiment:
            cfg = _load_cfg(Path("full_experiment.yaml"))
            _run(cfg)
    except Exception as e:  # noqa: BLE001
        # Print full stack – easier debugging on cluster
        traceback.print_exc()
        raise SystemExit(1) from e


if __name__ == "__main__":  # pragma: no cover
    main()
