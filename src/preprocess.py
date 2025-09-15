"""Dataset acquisition & task-map helpers."""

import os, urllib.request, zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchvision
import torchaudio
from torchvision import transforms

__all__ = ["prepare_dataset", "build_task_index"]

_DATA_ROOT = Path("data")


# ─────────────────────────────────────────────────────────────
#  Dataset download / mapping helpers
# ─────────────────────────────────────────────────────────────

def _download_ucihar() -> Path:
    tgt = _DATA_ROOT / "ucihar"
    if (tgt / "UCI HAR Dataset").exists():
        return tgt
    url = (
        "https://archive.ics.uci.edu/static/public/240/"
        "human+activity+recognition+using+smartphones.zip"
    )
    tgt.mkdir(parents=True, exist_ok=True)
    zip_path = tgt / "ucihar.zip"
    print(f"Downloading UCI-HAR …")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tgt)
    zip_path.unlink()
    return tgt


def prepare_dataset(name: str) -> Dict:
    name = name.lower()
    _DATA_ROOT.mkdir(exist_ok=True)

    if name == "edgebench-vision-1":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2761]),
            ]
        )
        train_set = torchvision.datasets.CIFAR100(_DATA_ROOT / "cifar100", train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR100(_DATA_ROOT / "cifar100", train=False, download=True, transform=transform)
        task_to_classes = [(2 * t, 2 * t + 1) for t in range(50)]
        return dict(train=train_set, test=test_set, tasks=task_to_classes)

    if name == "edgebench-audio-1":
        root = _DATA_ROOT / "speechcommands"
        train_dataset = torchaudio.datasets.SPEECHCOMMANDS(root, download=True, subset="training")
        test_dataset = torchaudio.datasets.SPEECHCOMMANDS(root, download=True, subset="testing")
        classes = sorted({dat[2] for dat in train_dataset})
        tasks = [(i,) for i in range(30)]
        return dict(train=train_dataset, test=test_dataset, tasks=tasks, class_order=classes)

    if name == "edgebench-fusion":
        v = prepare_dataset("edgebench-vision-1")
        a = prepare_dataset("edgebench-audio-1")
        imu_root = _download_ucihar()
        # Fusion loader abridged – experiments in this refactor only require vision part.
        return dict(vision=v, audio=a, tasks=v["tasks"])

    raise ValueError(f"Unknown dataset: {name}")


# ─────────────────────────────────────────────────────────────
#  Task-index helper for BWT computation
# ─────────────────────────────────────────────────────────────

def build_task_index(task_map: List[Tuple[int, ...]]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for t, cls in enumerate(task_map):
        for c in cls:
            mapping[c] = t
    return mapping
