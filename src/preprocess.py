"""Pre-processing stub.
Generates a **tiny synthetic** dataset *only* when the configuration
explicitly requests it via ``use_synthetic_data: true``.  Otherwise the
function raises ``FileNotFoundError`` to respect the NO-FALLBACK rule.
"""
from __future__ import annotations

from typing import Dict, Any
import os

import numpy as np


def preprocess(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Return a toy dataset or fail fast if real data is required."""

    # ------------------------------------------------------------------
    # Synthetic data path (allowed only when explicitly requested)
    # ------------------------------------------------------------------
    if config.get("use_synthetic_data", False):
        seed = int(config.get("random_seed", 0))
        rng = np.random.default_rng(seed)
        X = rng.standard_normal(size=(100, 10), dtype=np.float32)
        y = rng.integers(0, 2, size=100, dtype=np.int32)
        return {"X": X, "y": y}

    # ------------------------------------------------------------------
    # Real-data path – must exist; otherwise abort (NO-FALLBACK)
    # ------------------------------------------------------------------
    data_path = config.get("dataset_path", "")
    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(
            "Dataset not found — experiment terminated as per NO-FALLBACK constraint"
        )

    # NOTE: Real dataset loading is not implemented in this stub.  We
    #       raise so that future developers are forced to implement it
    #       instead of silently passing.
    raise FileNotFoundError("Real-dataset processing not implemented in this stub.")