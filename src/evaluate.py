import json
from pathlib import Path
from typing import Dict, Any

import torch
from diffusers.utils import make_image_grid

from .train import dump_result


def _fid_placeholder() -> float:  # noqa: D401 – simple placeholder until metric code is open-sourced
    """Raises to comply with NO-FALLBACK when FID calc is unavailable."""
    raise RuntimeError(
        "The official FID evaluation kernel is not yet open-sourced by the "
        "PHOENIX-RELAX authors. Aborting per NO-FALLBACK policy."
    )


def evaluate_diffusion(sd_pipe, prompts, cfg: Dict[str, Any]) -> None:
    """Generates images for *prompts*, saves a grid to .research/iteration1/images."""
    from torchvision.utils import save_image  # local import so torchvision optional elsewhere

    images_dir = Path(".research/iteration1/images")
    images_dir.mkdir(parents=True, exist_ok=True)

    outputs = sd_pipe(prompts).images
    grid = make_image_grid(outputs, rows=1, cols=len(outputs))
    grid_path = images_dir / "sd_preview.png"
    grid.save(grid_path)

    # Compute metrics – will hard-error if FID impl not present
    fid = _fid_placeholder()

    dump_result({"FID": fid, "image_grid": str(grid_path)}, cfg_name=cfg["_name"])