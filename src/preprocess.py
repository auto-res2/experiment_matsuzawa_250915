"""Pre-processing stub.
For this toy example there is *no real* data to preprocess, but the function is
kept so that the full {preprocess → train → evaluate} workflow exists.  We
simply validate the configuration and create the required directories so that
later stages do not fail when trying to write artefacts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

__all__ = ["preprocess"]


def preprocess(cfg: Dict) -> Dict:
    # Create the mandatory output directories ahead of time.
    Path(".research/iteration2").mkdir(parents=True, exist_ok=True)
    Path(".research/iteration2/images").mkdir(parents=True, exist_ok=True)
    return {"status": "noop", "detail": "No preprocessing required for synthetic demo."}
