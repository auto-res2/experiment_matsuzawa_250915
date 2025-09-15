import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

# -----------------------------------------------------------------------------
# Helper – create minimal dummy data for the smoke-test so that the pipeline
# passes the dataset availability checks without relying on external files.
# -----------------------------------------------------------------------------

def _create_dummy_datasets(cfg: Dict[str, Any]) -> None:  # noqa: D401
    base = Path("./data")
    for key, path in cfg["datasets"].items():
        p = Path(path)
        if p.exists():
            continue  # already there
        if key == "long_bench_trace":
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w") as f:
                json.dump([{"t": 0, "event": "dummy"}], f)
        elif key == "gplay_thermal":
            (p).mkdir(parents=True, exist_ok=True)
            # minimal npy tensors
            x = np.random.rand(8, 8).astype("float32")
            y = np.random.rand(8, 1).astype("float32")
            np.save(p / "train_inputs.npy", x)
            np.save(p / "train_targets.npy", y)
        else:
            p.mkdir(parents=True, exist_ok=True)


class TracePlayer:
    """Light-weight JSON trace reader for LONG-BENCH-v4."""

    def __init__(self, trace_path: Path):
        if not trace_path.exists():
            raise FileNotFoundError(
                f"Trace file {trace_path} missing – cannot continue by design."
            )
        with trace_path.open() as f:
            self._events = json.load(f)
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self._events):
            raise StopIteration
        evt = self._events[self._idx]
        self._idx += 1
        return evt


def prepare_datasets(cfg: Dict[str, Any]) -> None:
    """Sanity-checks that all datasets declared in *cfg* are present on disk.

    For the *smoke-test* we auto-generate tiny dummy datasets to keep the run
    self-contained.
    """
    if cfg.get("_name") == "smoke_test":
        _create_dummy_datasets(cfg)

    for key, path in cfg["datasets"].items():
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Dataset for '{key}' expected at {p} but was not found."
            )
