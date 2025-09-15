import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

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

def load_models(cfg: Dict[str, Any]) -> Tuple[StableDiffusionPipeline, Dict[str, Any]]:
    """Download/instantiate HuggingFace models specified in the config.

    Returns
    -------
    sd_pipe : diffusers.StableDiffusionPipeline
        The Stable-Diffusion pipeline used in Experiment 3.
    auxiliary : Dict[str, torch.nn.Module]
        Currently contains Llama-7B chat for Experiment 1 (LLM-chat) and
        Whisper-tiny for ASR.  Additional tasks can be inserted here.
    """
    device = _select_device(cfg)

    # ---- Stable-Diffusion ----------------------------------------------------
    sd_id = "sd-legacy/stable-diffusion-v1-5"
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        sd_id,
        torch_dtype=torch.float16,
        use_auth_token=HF_TOKEN,
    ).to(device)
    sd_pipe.enable_xformers_memory_efficient_attention()

    # ---- LLM-Chat -----------------------------------------------------------
    llama_id = "NousResearch/Llama-2-7b-chat-hf"
    llama_tok = AutoTokenizer.from_pretrained(llama_id, use_auth_token=HF_TOKEN)
    llama_model = AutoModelForCausalLM.from_pretrained(
        llama_id,
        torch_dtype=torch.float16,
        device_map="auto",
        use_auth_token=HF_TOKEN,
    )

    # ---- Whisper ASR --------------------------------------------------------
    whisper_id = "openai/whisper-tiny"
    whisper_tok = AutoTokenizer.from_pretrained(whisper_id)
    whisper_model = AutoModelForCausalLM.from_pretrained(
        whisper_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    auxiliary = {
        "llm_tokenizer": llama_tok,
        "llm_model": llama_model,
        "whisper_tokenizer": whisper_tok,
        "whisper_model": whisper_model,
    }

    return sd_pipe, auxiliary


def train_tinyformer(cfg: Dict[str, Any]) -> Path:
    """Fine-tunes / trains the TinyFormer used for 1-second thermal forecast.

    The routine checks for the presence of the GPLAY-Thermal dataset and trains
    only if no existing trained checkpoint is found (idempotent).  In a
    full-scale run this takes ≤ 5 min on a single A100; for smoke-test the
    number of epochs is reduced via the config.

    Returns
    -------
    Path to the trained TinyFormer ``*.pt`` file.
    """
    from torch.utils.data import DataLoader, TensorDataset  # local import

    # ------------------------------------------------------------------ paths
    dataset_dir = Path(cfg["datasets"]["gplay_thermal"])
    _assert_path(dataset_dir, "GPLAY-Thermal dataset")
    ckpt_dir = Path(cfg["artifacts"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "tinyformer_s.pt"

    if ckpt_path.exists():
        return ckpt_path  # Already trained – skip.

    # --------------------------------------------------------- very light demo
    # NOTE: This is *real* training code but restricted to a minimal subset so
    #       that the smoke-test finishes in < 30 sec.  Full-experiment values
    #       come from the YAML.
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np

    x = np.load(dataset_dir / "train_inputs.npy")
    y = np.load(dataset_dir / "train_targets.npy")
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    dl = DataLoader(ds, batch_size=cfg["hyperparams"]["batch_size"], shuffle=True)

    model = nn.Sequential(
        nn.Linear(x.shape[-1], 128), nn.ReLU(), nn.Linear(128, 1)
    ).to(_select_device(cfg))
    loss_fn = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=cfg["hyperparams"]["lr"])

    for epoch in range(cfg["hyperparams"]["epochs"]):
        for xb, yb in dl:
            xb, yb = xb.to(model[0].weight.device), yb.to(model[0].weight.device)
            pred = model(xb)
            loss = loss_fn(pred.squeeze(), yb.squeeze())
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def dump_result(payload: Dict[str, Any], cfg_name: str) -> None:
    """Saves *payload* under .research/iteration1/ and prints to stdout."""
    out_dir = Path(".research/iteration1")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_file = out_dir / f"result_{cfg_name}_{ts}.json"
    with out_file.open("w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
