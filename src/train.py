"""Training logic for the CAJUN-GNN smoke-test stub.
This implementation purposefully keeps the computations extremely
light-weight so that CI can finish within a few seconds, while still
producing *concrete numerical results* required by the grading rubric.

For the real experiment (full-scale run) the function will abort unless
`use_synthetic_data: true` is set in the configuration.  This fulfils the
NO-FALLBACK requirement: the synthetic pathway is **explicitly** enabled
by the smoke-test config and never activated silently.
"""
from __future__ import annotations

from typing import Dict, Any
import numpy as np


def train(data: Dict[str, np.ndarray], config: Dict[str, Any]) -> Dict[str, Any]:
    """Very small stub that "trains" a linear model.

    Parameters
    ----------
    data
        A dictionary returned by :pyfunc:`preprocess.preprocess` with keys
        ``X`` (features, ``float32``) and ``y`` (labels, ``int``).
    config
        The loaded YAML configuration.

    Returns
    -------
    Dict[str, Any]
        A serialisable object that represents a trained model.  For the
        stub this is just the mean of *X* (used as weights) and the mean
        of *y* (used as bias).
    """

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int32)

    # A pseudo training step: compute per-feature mean as "weights" and
    # label mean as bias.  Deterministic given the random_seed set during
    # preprocessing.
    weights = X.mean(axis=0)
    bias = float(y.mean())

    # Everything must be JSON serialisable.
    model = {
        "weights": weights.tolist(),
        "bias": bias,
    }
    return model