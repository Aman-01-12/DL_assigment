# File: src/train.py
# Purpose: Train VGG-19 on BCCD with early stopping and checkpointing.
# Author: Aman Verma (23bcs012), Ankit Kashyap (23bcs017), Abhishek Chauhan (23bcs003)
# Affiliation: School of Computer science and Engineering, Shri Mata Vaishno Devi University
# Contact: info@smvdu.ac.in

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import create_train_val_loaders
from src.model import build_vgg19
from src.utils import CSVLogger, ensure_dir, save_checkpoint, set_seed


def load_config(config_path: str | Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


class EarlyStopping:
    """Stop training when validation loss has not improved."""

    def __init__(self, patience: int = 5, delta: float = 0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            return
        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def plot_curves(history: Dict[str, List[float]], figures_dir: Path):
    ensure_dir(figures_dir)

    plt.figure(figsize=(8, 6))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(figures_dir / "loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig(figures_dir / "accuracy_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Accuracy")
    axes[1].plot(history["val_acc"], label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.suptitle("Training Curves")
    fig.tight_layout()
    fig.savefig(figures_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def train_model(config: Dict):
    set_seed(int(config["project"]["seed"]))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = create_train_val_loaders(config)

    model = build_vgg19(
        num_classes=int(config["data"]["num_classes"]),
        pretrained=bool(config["model"]["pretrained"]),
        dropout=float(config["model"]["dropout"]),
        freeze_features=bool(config["model"]["freeze_features"]),
    )
    model = model.to(device)

    optimizer = Adam(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scheduler = StepLR(
        optimizer,
        step_size=int(config["scheduler"]["step_size"]),
        gamma=float(config["scheduler"]["gamma"]),
    )

    criterion = nn.CrossEntropyLoss()

    logs_dir = Path(config["paths"]["logs_dir"])
    checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    ensure_dir(logs_dir)
    ensure_dir(checkpoints_dir)
    ensure_dir(figures_dir)

    log_path = logs_dir / "training_log.csv"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    early_stopping = EarlyStopping(
        patience=int(config["train"]["early_stopping_patience"]),
        delta=float(config["train"]["early_stopping_delta"]),
    )

    best_val_acc = -1.0
    best_val_loss = float("inf")

    hyperparams = {
        "lr": float(config["optimizer"]["lr"]),
        "batch_size": int(config["data"]["batch_size"]),
        "epochs": int(config["train"]["epochs"]),
        "optimizer": str(config["optimizer"]["name"]),
        "scheduler": str(config["scheduler"]["name"]),
    }

    with CSVLogger(log_path, ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]) as logger:
        for epoch in range(1, int(config["train"]["epochs"]) + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            current_lr = optimizer.param_groups[0]["lr"]
            logger.log(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "train_acc": f"{train_acc:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_acc": f"{val_acc:.6f}",
                    "lr": f"{current_lr:.8f}",
                }
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                save_checkpoint(
                    checkpoints_dir / "best_model.pth",
                    model,
                    optimizer,
                    epoch,
                    val_acc,
                    val_loss,
                    class_names,
                    hyperparams,
                )

            early_stopping(val_loss)
            if early_stopping.early_stop:
                break

            scheduler.step()

    plot_curves(history, figures_dir)

    return {
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "epochs_ran": len(history["train_loss"]),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train VGG-19 on BCCD")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train_model(config)


if __name__ == "__main__":
    main()
