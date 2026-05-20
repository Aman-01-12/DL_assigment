# File: src/evaluate.py
# Purpose: Evaluate the best checkpoint on the test set with full metrics and plots.
# Author: Aman Verma (23bcs012), Ankit Kashyap (23bcs017), Abhishek Chauhan (23bcs003)
# Affiliation: School of Computer science and Engineering, Shri Mata Vaishno Devi University
# Contact: info@smvdu.ac.in

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import create_test_loader
from src.model import build_vgg19
from src.utils import ensure_dir, load_checkpoint, set_seed


def load_config(config_path: str | Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def evaluate_model(config: Dict, checkpoint_path: str | Path):
    set_seed(int(config["project"]["seed"]))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    test_loader, class_names = create_test_loader(config)

    model = build_vgg19(
        num_classes=int(config["data"]["num_classes"]),
        pretrained=False,
        dropout=float(config["model"]["dropout"]),
        freeze_features=False,
    )
    model = model.to(device)

    checkpoint = load_checkpoint(checkpoint_path, model, optimizer=None, device=device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    all_probs_np = np.array(all_probs)

    accuracy = accuracy_score(all_labels_np, all_preds_np)
    f1_macro = f1_score(all_labels_np, all_preds_np, average="macro")
    f1_per_class = f1_score(all_labels_np, all_preds_np, average=None)

    report = classification_report(
        all_labels_np,
        all_preds_np,
        target_names=class_names,
        digits=4,
    )

    y_bin = label_binarize(all_labels_np, classes=list(range(len(class_names))))
    auc_macro = roc_auc_score(y_bin, all_probs_np, average="macro", multi_class="ovr")

    figures_dir = Path(config["paths"]["figures_dir"])
    ensure_dir(figures_dir)

    cm = confusion_matrix(all_labels_np, all_preds_np)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - VGGNet-19 on BCCD")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    fpr = {}
    tpr = {}
    auc_scores = {}
    for i, name in enumerate(class_names):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_probs_np[:, i])
        auc_scores[i] = roc_auc_score(y_bin[:, i], all_probs_np[:, i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_names))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(class_names)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(class_names)
    macro_auc_curve = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(10, 7))
    for i, name in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], label=f"{name} (AUC={auc_scores[i]:.3f})")
    plt.plot(
        all_fpr,
        mean_tpr,
        linestyle="--",
        label=f"Macro (AUC={macro_auc_curve:.3f})",
    )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - VGGNet-19 on BCCD")
    plt.legend()
    plt.savefig(figures_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_per_class": dict(zip(class_names, f1_per_class.tolist())),
        "auc_macro": auc_macro,
        "checkpoint_epoch": checkpoint.get("epoch"),
    }

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Macro AUC-ROC (OvR): {auc_macro:.4f}")
    print("\nClassification Report:\n")
    print(report)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate VGG-19 on BCCD test set")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate_model(config, args.checkpoint)


if __name__ == "__main__":
    main()
