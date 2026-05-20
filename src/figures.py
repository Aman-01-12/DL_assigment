# File: src/figures.py
# Purpose: Generate report-ready figures for the BCCD VGG-19 project.
# Author: Aman Verma (23bcs012), Ankit Kashyap (23bcs017), Abhishek Chauhan (23bcs003)
# Affiliation: School of Computer science and Engineering, Shri Mata Vaishno Devi University
# Contact: info@smvdu.ac.in

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import patches
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    count_by_class,
    limit_samples_per_class,
    load_samples,
    stratified_split_indices,
)
from src.utils import ensure_dir


def load_config(config_path: str | Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_splits(config: Dict):
    class_names = list(config["data"]["class_names"])
    train_dir = Path(config["paths"]["train_dir"])
    test_dir = Path(config["paths"]["test_dir"])
    val_split = float(config["data"]["val_split"])
    seed = int(config["project"]["seed"])

    train_samples, train_labels = load_samples(train_dir, class_names)
    max_per_class = config["data"].get("max_samples_per_class")
    if max_per_class is not None:
        max_per_class = int(max_per_class)
        train_samples, train_labels = limit_samples_per_class(
            train_samples,
            train_labels,
            max_per_class,
            seed,
        )

    train_indices, val_indices = stratified_split_indices(train_labels, val_split, seed)
    train_subset = [train_samples[i] for i in train_indices]
    val_subset = [train_samples[i] for i in val_indices]
    test_samples, _ = load_samples(test_dir, class_names)

    return train_subset, val_subset, test_samples, class_names


def save_class_distribution(
    train_subset: List[Tuple[Path, int]],
    val_subset: List[Tuple[Path, int]],
    test_samples: List[Tuple[Path, int]],
    class_names: List[str],
    figures_dir: Path,
    logs_dir: Path,
):
    train_counts = count_by_class(train_subset, class_names)
    val_counts = count_by_class(val_subset, class_names)
    test_counts = count_by_class(test_samples, class_names)

    stats_df = pd.DataFrame({
        "Train": train_counts,
        "Val": val_counts,
        "Test": test_counts,
    })
    stats_df["Total"] = stats_df.sum(axis=1)
    stats_df.loc["Total"] = stats_df.sum(axis=0)

    ensure_dir(logs_dir)
    stats_df.to_csv(logs_dir / "dataset_stats.csv")

    labels = class_names
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(9, 5))
    plt.bar(x - width, [train_counts[label] for label in labels], width, label="Train")
    plt.bar(x, [val_counts[label] for label in labels], width, label="Val")
    plt.bar(x + width, [test_counts[label] for label in labels], width, label="Test")
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Count")
    plt.title("Class Distribution by Split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_sample_grid(
    train_subset: List[Tuple[Path, int]],
    class_names: List[str],
    figures_dir: Path,
):
    samples_by_class = {name: [] for name in class_names}
    for path, label in train_subset:
        class_name = class_names[label]
        if len(samples_by_class[class_name]) < 2:
            samples_by_class[class_name].append(path)
        if all(len(paths) == 2 for paths in samples_by_class.values()):
            break

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    index = 0
    for class_name in class_names:
        for img_path in samples_by_class[class_name]:
            image = Image.open(img_path).convert("RGB")
            axes[index].imshow(image)
            axes[index].set_title(class_name)
            axes[index].axis("off")
            index += 1

    plt.suptitle("Sample Images (2 per class)")
    plt.tight_layout()
    plt.savefig(figures_dir / "sample_images.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_architecture_diagram(figures_dir: Path):
    blocks = [
        "Input 224x224x3",
        "Conv3-64 x2",
        "MaxPool",
        "Conv3-128 x2",
        "MaxPool",
        "Conv3-256 x4",
        "MaxPool",
        "Conv3-512 x4",
        "MaxPool",
        "Conv3-512 x4",
        "MaxPool",
        "FC 4096",
        "FC 4096",
        "FC 4 (Softmax)",
    ]

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.95
    height = 0.045
    gap = 0.012

    for block in blocks:
        rect = patches.FancyBboxPatch(
            (0.1, y - height),
            0.8,
            height,
            boxstyle="round,pad=0.01",
            linewidth=1,
            edgecolor="#333333",
            facecolor="#d9ead3",
        )
        ax.add_patch(rect)
        ax.text(0.5, y - height / 2, block, ha="center", va="center", fontsize=9)
        y -= height + gap

    ax.set_title("VGG-19 Architecture", fontsize=12, pad=10)
    plt.savefig(figures_dir / "architecture_vgg19.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(logs_dir: Path, figures_dir: Path):
    log_path = logs_dir / "training_log.csv"
    if not log_path.exists():
        return

    df = pd.read_csv(log_path)
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(df["train_loss"], label="Train Loss")
    axes[0].plot(df["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(df["train_acc"], label="Train Accuracy")
    axes[1].plot(df["val_acc"], label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.suptitle("Training Curves")
    fig.tight_layout()
    fig.savefig(figures_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate report figures")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    figures_dir = Path(config["paths"]["figures_dir"])
    logs_dir = Path(config["paths"]["logs_dir"])
    ensure_dir(figures_dir)

    train_subset, val_subset, test_samples, class_names = build_splits(config)
    save_class_distribution(train_subset, val_subset, test_samples, class_names, figures_dir, logs_dir)
    save_sample_grid(train_subset, class_names, figures_dir)
    save_architecture_diagram(figures_dir)
    save_training_curves(logs_dir, figures_dir)


if __name__ == "__main__":
    main()
