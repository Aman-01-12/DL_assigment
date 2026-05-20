# File: src/utils.py
# Purpose: Reproducibility helpers, checkpoint IO, and CSV logging.
# Author: Aman Verma (23bcs012), Ankit Kashyap (23bcs017), Abhishek Chauhan (23bcs003)
# Affiliation: School of Computer science and Engineering, Shri Mata Vaishno Devi University
# Contact: info@smvdu.ac.in

from __future__ import annotations

import csv
from pathlib import Path
import random
from typing import Dict, Iterable, Optional

import numpy as np
import torch


def set_seed(seed: int):
    """Set random seeds and deterministic flags for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    val_loss: float,
    class_names: Iterable[str],
    hyperparams: Dict,
):
    """Save a checkpoint with the required metadata."""
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "class_names": list(class_names),
        "hyperparams": dict(hyperparams),
    }
    torch.save(payload, str(path))


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str | torch.device = "cpu",
):
    """Load checkpoint contents into model (and optimizer when provided)."""
    checkpoint = torch.load(str(path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


class CSVLogger:
    """Lightweight CSV logger for epoch metrics."""

    def __init__(self, log_path: str | Path, fieldnames: Iterable[str]):
        self.log_path = Path(log_path)
        ensure_dir(self.log_path.parent)
        self.fieldnames = list(fieldnames)
        self._file = self.log_path.open("a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        if self.log_path.stat().st_size == 0:
            self._writer.writeheader()

    def log(self, row: Dict):
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
