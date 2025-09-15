import os
import random
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# timm is far more light-weight than the full torchvision classification zoo and already
# included in the external resources list.
import timm

from .preprocess import build_datasets


class _RandomFallbackDataset(Dataset):
    """A tiny in-RAM dataset that is only used during the smoke-test phase when the
    real datasets are intentionally skipped for CI speed. DO NOT use this in the
    full experiment – the `main.py` driver will load the real data paths.
    """

    def __init__(self, length: int = 32, num_classes: int = 10):
        self.length = length
        self.x = torch.randn(length, 3, 224, 224)
        self.y = torch.randint(0, num_classes, (length,))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Trainer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = cfg["model"].get("name", "timm/resnet18.a1_in1k")
        num_classes = cfg["model"].get("num_classes", 1000)
        try:
            # `timm.create_model` automatically downloads the weights if not cached.
            self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        except Exception as exc:
            raise RuntimeError(f"Unable to load model '{model_name}': {exc}") from exc

        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg["training"].get("lr", 3e-4),
            weight_decay=cfg["training"].get("weight_decay", 1e-2),
        )
        self.epochs = int(cfg["training"].get("epochs", 1))
        self.batch_size = int(cfg["training"].get("batch_size", 32))
        self.num_workers = int(cfg["training"].get("num_workers", 4))

        self.train_loader, self.val_loader = self._build_loaders()

    # ---------------------------------------------------------------------
    # public api
    # ---------------------------------------------------------------------
    def fit(self) -> Tuple[float, float]:
        best_val_acc = 0.0
        for epoch in range(1, self.epochs + 1):
            self._train_one_epoch(epoch)
            val_loss, val_acc = self._evaluate(self.val_loader)
            best_val_acc = max(best_val_acc, val_acc)
            print(f"Epoch {epoch:02d}/{self.epochs} – val_loss: {val_loss:.4f} – val_acc: {val_acc:.3%}")
        return best_val_acc, val_loss

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _build_loaders(self):
        smoke = bool(self.cfg.get("smoke_test", False))
        if smoke:
            train_ds = _RandomFallbackDataset()
            val_ds = _RandomFallbackDataset()
        else:
            train_ds, val_ds = build_datasets(self.cfg)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader

    def _train_one_epoch(self, epoch: int):
        self.model.train()
        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        total, correct, running_loss = 0, 0, 0.0
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            outputs = self.model(images)
            running_loss += self.criterion(outputs, labels).item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        avg_loss = running_loss / max(total, 1)
        acc = correct / max(total, 1)
        return avg_loss, acc
