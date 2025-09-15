import json
from pathlib import Path
from typing import Dict, Any


class TracePlayer:
    """Light-weight JSON trace reader for LONG-BENCH-v4."""

    def __init__(self, trace_path: Path):
        if not trace_path.exists():
            raise FileNotFoundError(
                f"Trace file {trace_path} missing – cannot continue by design.""
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
    """Sanity-checks that all datasets declared in *cfg* are present on disk."""
    for key, path in cfg["datasets"].items():
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Dataset for '{key}' expected at {p} but was not found."
            )
