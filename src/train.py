"""
Full training / rehearsal pipeline for Experiment 1 (Variable-Rate Replay vs. Baselines).

The original monolithic script has been refactored and lightly debugged:
• fixed a double-wrapped DataLoader that caused a TypeError
• added explicit handling for feature-based (EPQ) vs. image-based (JPEG) replay buffers
• corrected JPEG buffer utilities (missing BytesIO import, de-normalisation before encoding)
• minor style / typing clean-ups – **no research logic changed**
"""

from __future__ import annotations

import json, math, os, random, time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Union, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from tqdm import tqdm

from .preprocess import prepare_dataset, build_task_index

# ─────────────────────────────────────────────────────────────
#  General utilities
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────────────────────
#  Backbone & heads
# ─────────────────────────────────────────────────────────────

def build_backbone(device: torch.device) -> Tuple[nn.Module, int]:
    """MobileNet-V2-0.35 backbone (ImageNet-pre-trained). Returns net & feature dim."""
    mnet = models.mobilenet_v2(width_mult=0.35, weights="IMAGENET1K_V1")
    backbone = nn.Sequential(*list(mnet.features), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    backbone.eval().to(device)

    # Measure FLOPs once for the controller (not used in Exp-1 runner but kept for completeness)
    dummy = torch.randn(1, 3, 224, 224, device=device)
    _ = profile(backbone, inputs=(dummy,), verbose=False)
    return backbone, 1280  # MobileNet-V2 feature size


class CosineClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.kaiming_normal_(self.W)

    def forward(self, x: torch.Tensor):
        x = F.normalize(x, dim=-1)
        W = F.normalize(self.W, dim=-1)
        return 30.0 * (x @ W.t())


# ─────────────────────────────────────────────────────────────
#  EPQ latent compressor & buffer
# ─────────────────────────────────────────────────────────────

class EPQ(nn.Module):
    """Elastic Product Quantiser (variable or fixed depth)."""

    def __init__(self, code_dim: int = 512, K: int = 256, L: int = 4, threshold: float = 0.005):
        super().__init__()
        self.code_dim, self.K, self.L = code_dim, K, L
        self.codebooks = nn.Parameter(torch.randn(L, K, code_dim))
        self.threshold = threshold
        nn.init.normal_(self.codebooks, std=0.02)

    def forward(self, z: torch.Tensor, *, force_depth: int | None = None) -> Tuple[torch.Tensor, float]:
        residual = z
        indices: List[torch.Tensor] = []
        bits = 0.0
        for l in range(self.L):
            dist = (residual.unsqueeze(1) - self.codebooks[l]).pow(2).sum(-1)  # (B,K)
            idx = dist.argmin(-1)
            quant = self.codebooks[l][idx]
            gain = (residual.pow(2).mean() - (residual - quant).pow(2).mean()).item()
            if force_depth is None and gain < self.threshold:
                break
            residual = residual - quant
            indices.append(idx)
            bits += math.log2(self.K)
            if force_depth is not None and len(indices) >= force_depth:
                break
        codes = (
            torch.stack(indices, dim=1)
            if indices
            else torch.empty(z.size(0), 0, dtype=torch.long, device=z.device)
        )
        return codes, bits


class ReplayBufferBase:
    """Abstract buffer interface."""

    def push(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, batch_size: int):
        raise NotImplementedError


class EPQBuffer(ReplayBufferBase):
    """Variable-rate (ours) or fixed-rate (R3_NoVR) depending on *force_depth*."""

    def __init__(self, epq: EPQ, force_depth: int | None, max_bits: int):
        self.epq, self.force_depth, self.max_bits = epq, force_depth, max_bits
        self.codes: List[Tuple[torch.Tensor, int]] = []
        self.cur_bits = 0

    # ----------------------------------------
    def push(self, z: torch.Tensor):
        codes, bits_f = self.epq(z, force_depth=self.force_depth)
        bits = int(bits_f)
        if self.cur_bits + bits > self.max_bits:
            # FIFO eviction
            while self.codes and self.cur_bits + bits > self.max_bits:
                _, old_bits = self.codes.pop(0)
                self.cur_bits -= old_bits
        self.codes.append((codes.cpu(), bits))
        self.cur_bits += bits

    # ----------------------------------------
    def sample(self, batch_size: int) -> torch.Tensor:
        idx = np.random.choice(len(self.codes), batch_size)
        batch_codes = torch.cat([self.codes[i][0] for i in idx], dim=0).to(self.epq.codebooks.device)

        # Decode
        z_rec = torch.zeros(batch_codes.size(0), self.epq.code_dim, device=batch_codes.device)
        for l in range(batch_codes.size(1)):
            z_rec += self.epq.codebooks[l][batch_codes[:, l]]
        return z_rec


class JPEGBuffer(ReplayBufferBase):
    """Stores images compressed as JPEG byte blobs under a memory budget."""

    def __init__(self, quality: int, max_bytes: int, device: torch.device):
        from PIL import Image  # type: ignore

        self.quality, self.max_bytes, self.device = quality, max_bytes, device
        self.buffer: List[bytes] = []
        self.cur_bytes = 0
        self._bio = BytesIO()

    # ----------------------------------------
    def push(self, img: torch.Tensor):
        """img: (C,H,W) float in [0,1] on *any* device."""
        from torchvision.transforms.functional import to_pil_image  # type: ignore

        pil = to_pil_image(img.cpu().clamp(0, 1))
        self._bio.seek(0)
        pil.save(self._bio, format="jpeg", quality=self.quality)
        arr = self._bio.getvalue()

        if self.cur_bytes + len(arr) > self.max_bytes:
            while self.buffer and self.cur_bytes + len(arr) > self.max_bytes:
                self.cur_bytes -= len(self.buffer.pop(0))
        self.buffer.append(arr)
        self.cur_bytes += len(arr)

    # ----------------------------------------
    def sample(self, batch_size: int) -> torch.Tensor:
        from PIL import Image  # type: ignore
        from torchvision.transforms import ToTensor  # type: ignore

        idx = np.random.choice(len(self.buffer), batch_size)
        imgs = [ToTensor()(Image.open(BytesIO(self.buffer[i]))) for i in idx]
        return torch.stack(imgs, dim=0).to(self.device)


# ─────────────────────────────────────────────────────────────
#  Drift-aware translator (identity for JPEG replay)
# ─────────────────────────────────────────────────────────────

class Translator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())

    def forward(self, z_old: torch.Tensor):
        return self.fc(z_old)


# ─────────────────────────────────────────────────────────────
#  Energy & privacy helpers (GPU power – smoke-level only)
# ─────────────────────────────────────────────────────────────

class EnergyMeter:
    def __init__(self):
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.handle = None
        self.hist: List[float] = []

    def tick(self):
        if self.handle is None:
            return
        import pynvml  # type: ignore

        power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1e3  # W
        self.hist.append(power)

    def joule_per_step(self, steps: int):
        if not self.hist:
            return None
        return (sum(self.hist) / len(self.hist)) * steps / 1000  # J / step


def estimate_dp_epsilon(bits: int):
    """Very coarse privacy ε upper bound used in the paper."""
    return 2.2 * bits / (32 * 1024)


# ─────────────────────────────────────────────────────────────
#  Experiment-1 runner (Variable-Rate vs. Baselines)
# ─────────────────────────────────────────────────────────────

def _denormalise(img: torch.Tensor) -> torch.Tensor:
    """Undo CIFAR-100 normalisation so that JPEG encoding receives [0,1] floats."""
    mean = torch.tensor([0.5071, 0.4866, 0.4409], device=img.device).view(3, 1, 1)
    std = torch.tensor([0.2673, 0.2564, 0.2761], device=img.device).view(3, 1, 1)
    return img * std + mean


def _classes_upto(task_map: Sequence[Tuple[int, ...]], t_inclusive: int) -> List[int]:
    """Flatten classes up to and including task t."""
    classes: List[int] = []
    for t in range(t_inclusive + 1):
        classes.extend(task_map[t])
    return classes


def run_experiment_1(cfg: Dict, device: torch.device, results_dir: Path):
    """Main training loop for Experiment 1 (smoke & full)."""

    print("\n[Exp-1] Preparing dataset …")
    data = prepare_dataset(cfg["dataset"])
    task_map = data["tasks"][: cfg["tasks"]]

    backbone, dim = build_backbone(device)
    epq = EPQ(code_dim=dim, threshold=0.005).to(device)
    translator = Translator(dim).to(device)

    criterion = nn.CrossEntropyLoss()

    # Helper --------------------------------------------------
    def build_buffer(name: str, memory_kb: int) -> ReplayBufferBase:
        budget_bits = memory_kb * 8 * 1024
        if name == "R3":
            return EPQBuffer(epq, force_depth=None, max_bits=budget_bits)
        if name == "R3_NoVR":
            return EPQBuffer(epq, force_depth=4, max_bits=budget_bits)
        if name == "ER_JPEG":
            return JPEGBuffer(quality=25, max_bytes=memory_kb * 1024, device=device)
        raise ValueError(f"Unknown variant {name}")

    # ---------------------------------------------------------
    for mem_kb in cfg["budgets"]["memory_kb"]:
        for variant in cfg["variants"]:
            print(f"\n[Exp-1] Variant={variant}  Memory={mem_kb} kB")
            buffer = build_buffer(variant, mem_kb)
            clf = CosineClassifier(dim, len(task_map) * 2).to(device)
            opt = torch.optim.AdamW(
                list(backbone.parameters()) + list(clf.parameters()), lr=1e-3, weight_decay=1e-2
            )

            acc_history: List[float] = []
            energy = EnergyMeter()
            global_step = 0

            for t_idx, classes in enumerate(task_map):
                # Dataloader for current task
                idx = [i for i, (_, y) in enumerate(data["train"]) if y in classes]
                loader = DataLoader(
                    Subset(data["train"], idx),
                    batch_size=64,
                    shuffle=True,
                    num_workers=cfg["num_workers"],
                )

                backbone.train()
                for _ in range(cfg["hyper"]["epochs_per_pass"]):
                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        energy.tick()
                        feat = backbone(x)
                        logits = clf(feat)
                        loss = criterion(logits, y)

                        # ---------------- Replay ----------------
                        if (isinstance(buffer, EPQBuffer) and len(buffer.codes) >= 32) or (
                            isinstance(buffer, JPEGBuffer) and len(buffer.buffer) >= 32
                        ):
                            rep_data = buffer.sample(32)
                            if isinstance(buffer, EPQBuffer):
                                rep_feat = translator(rep_data)
                            else:  # JPEG → images
                                rep_feat = backbone(rep_data)
                            logits_rep = clf(rep_feat)
                            loss += criterion(
                                logits_rep,
                                torch.randint(0, clf.W.size(0), (32,), device=device),
                            )

                        # update
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                        # ---------------- Store in buffer ----------------
                        with torch.no_grad():
                            if isinstance(buffer, EPQBuffer):
                                buffer.push(feat.detach())
                            else:  # JPEG → store original images (denormalised)
                                imgs_dn = _denormalise(x.detach())
                                for img_single in imgs_dn:
                                    buffer.push(img_single.cpu())
                        global_step += 1

                # ---------------- Validation on seen classes ----------------
                backbone.eval()
                val_idx = [i for i, (_, y) in enumerate(data["test"]) if y in _classes_upto(task_map, t_idx)]
                val_loader = DataLoader(
                    Subset(data["test"], val_idx),
                    batch_size=128,
                    shuffle=False,
                    num_workers=cfg["num_workers"],
                )
                correct = total = 0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        logits = clf(backbone(x))
                        pred = logits.argmax(-1)
                        correct += (pred == y).sum().item()
                        total += y.size(0)
                aa = 100 * correct / total
                acc_history.append(aa)
                print(
                    f"  Task {t_idx:02d}  AA={aa:5.2f}%  BufferBits={getattr(buffer,'cur_bits',0):>7}  BufferBytes={getattr(buffer,'cur_bytes',0):>7}"
                )

            # ---------------- Persist results ----------------
            result = {
                "variant": variant,
                "memory_kb": mem_kb,
                "AA": float(np.mean(acc_history)),
                "BWT": float(np.mean(acc_history) - acc_history[-1]),
                "energy_mJ_step": energy.joule_per_step(global_step) or -1,
                "privacy_epsilon": estimate_dp_epsilon(getattr(buffer, "cur_bits", 8 * getattr(buffer, "cur_bytes", 0))),
            }
            out_file = results_dir / f"exp1_{variant}_{mem_kb}kB.json"
            out_file.write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))


__all__ = [
    "run_experiment_1",
]
