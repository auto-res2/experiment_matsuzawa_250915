import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import torch

# NOTE: heavy libraries (diffusers/transformers) are imported lazily only when
#       required for the *full* experiment.  This keeps the smoke-test fast and
#       memory-friendly on CPU-only CI runners.

HF_TOKEN = os.getenv("HF_TOKEN", None)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _assert_path(path: Path, resource_name: str) -> None:
    """Ensures that a required file/folder exists.
    Raises FileNotFoundError when the path is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required {resource_name} not found at {path}. "
            "Per NO-FALLBACK policy the run is aborted."
        )


def _select_device(cfg: Dict[str, Any]) -> torch.device:
    requested = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise EnvironmentError("CUDA requested but not available on this system.")
    return torch.device(requested)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def _dummy_sd_pipeline():  # noqa: D401 – minimal stub for smoke-test
    """Returns a light-weight stub mimicking Diffusers pipeline."""
    from types import SimpleNamespace
    from PIL import Image
    import numpy as np

    class _Pipe:  # pylint: disable=too-few-public-methods
        def __call__(self, prompts):  # noqa: D401 – mimic diffusers API
            images = []
            for _ in prompts:
                arr = (np.random.rand(64, 64, 3) * 255).astype("uint8")
                images.append(Image.fromarray(arr))
            return SimpleNamespace(images=images)

    return _Pipe()


def load_models(cfg: Dict[str, Any]) -> Tuple[object, Dict[str, Any]]:
    """Download/instantiate HuggingFace models specified in the config.

    For the *smoke-test* we instantiate *stub* objects only so that CI runs fast
    and offline.  The full heavyweight models are created for the
    "full_experiment" configuration.
    """
    if cfg.get("_name") == "smoke_test":
        # ------------------------------------------------------------------ stubs
        sd_pipe = _dummy_sd_pipeline()
        auxiliary: Dict[str, Any] = {}
        return sd_pipe, auxiliary

    # ------------------------------------------------------------------ full run
    from diffusers import StableDiffusionPipeline  # heavy import – avoid in CI
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _select_device(cfg)

    # ---- Stable-Diffusion ----------------------------------------------------
    sd_id = "sd-legacy/stable-diffusion-v1-5"
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        sd_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        use_auth_token=HF_TOKEN,
    ).to(device)
    if device.type == "cuda":
        # xFormers only available on CUDA
        sd_pipe.enable_xformers_memory_efficient_attention()

    # ---- LLM-Chat -----------------------------------------------------------
    llama_id = "NousResearch/Llama-2-7b-chat-hf"
    llama_tok = AutoTokenizer.from_pretrained(llama_id, use_auth_token=HF_TOKEN)
    llama_model = AutoModelForCausalLM.from_pretrained(
        llama_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto",
        use_auth_token=HF_TOKEN,
    )

    # ---- Whisper ASR --------------------------------------------------------
    whisper_id = "openai/whisper-tiny"
    whisper_tok = AutoTokenizer.from_pretrained(whisper_id)
    whisper_model = AutoModelForCausalLM.from_pretrained(
        whisper_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto",
    )

    auxiliary = {
        "llm_tokenizer": llama_tok,
        "llm_model": llama_model,
        "whisper_tokenizer": whisper_tok,
        "whisper_model": whisper_model,
    }

    return sd_pipe, auxiliary


def train_tinyformer(cfg: Dict[str, Any]) -> Path:  # noqa: D401
    """Fine-tunes / trains the TinyFormer used for 1-second thermal forecast.

    For the smoke-test we train a *tiny* linear model on randomly generated
    tensors so that the step finishes within a few hundred milliseconds.
    """
    from torch.utils.data import DataLoader, TensorDataset  # local import
    import numpy as np
    import torch.nn as nn
    import torch.optim as optim

    ckpt_dir = Path(cfg["artifacts"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "tinyformer_s.pt"

    if ckpt_path.exists():
        return ckpt_path  # Already trained – skip.

    if cfg.get("_name") == "smoke_test":
        # ------------------------------------------------------------- tiny stub
        x = np.random.rand(256, 8).astype("float32")
        y = np.random.rand(256, 1).astype("float32")
    else:
        dataset_dir = Path(cfg["datasets"]["gplay_thermal"])
        _assert_path(dataset_dir, "GPLAY-Thermal dataset")
        x = np.load(dataset_dir / "train_inputs.npy")
        y = np.load(dataset_dir / "train_targets.npy")

    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=cfg["hyperparams"]["batch_size"], shuffle=True)

    model = nn.Sequential(nn.Linear(x.shape[-1], 32), nn.ReLU(), nn.Linear(32, 1))
    device = _select_device(cfg)
    model.to(device)

    loss_fn = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=cfg["hyperparams"]["lr"])

    for _ in range(cfg["hyperparams"]["epochs"]):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred.squeeze(), yb.squeeze())
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def dump_result(payload: Dict[str, Any], cfg_name: str) -> None:
    """Saves *payload* under .research/iteration3/ and prints to stdout."""
    out_dir = Path(".research/iteration3")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_file = out_dir / f"result_{cfg_name}_{ts}.json"
    with out_file.open("w") as f:
        json.dump(payload, f, indent=2)
    # Also print to STDOUT for quick validation
    print(json.dumps(payload, indent=2))
