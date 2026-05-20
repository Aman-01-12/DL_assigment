# File: src/dataset.py
# Purpose: Dataset utilities, transforms, and stratified DataLoader factory.
# Author: Aman Verma (23bcs012), Ankit Kashyap (23bcs017), Abhishek Chauhan (23bcs003)
# Affiliation: School of Computer science and Engineering, Shri Mata Vaishno Devi University
# Contact: info@smvdu.ac.in

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class BCCDImageDataset(Dataset):
    """Torch Dataset for BCCD images using an explicit sample list."""

    def __init__(self, samples: Sequence[Tuple[Path, int]], class_names: Sequence[str], transform=None):
        if not samples:
            raise ValueError("Samples list is empty.")
        self.samples = list(samples)
        self.class_names = list(class_names)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_transforms(input_size: int):
    """Create train and eval transforms using ImageNet normalization."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    return train_transform, eval_transform


def _list_image_files(class_dir: Path) -> List[Path]:
    files = []
    for ext in VALID_IMAGE_EXTENSIONS:
        files.extend(class_dir.glob(f"*{ext}"))
        files.extend(class_dir.glob(f"*{ext.upper()}"))
    return sorted({p.resolve() for p in files})


def load_samples(root_dir: str | Path, class_names: Sequence[str]):
    """Load image file paths and labels from a class-based directory tree."""
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {root_path}")

    samples: List[Tuple[Path, int]] = []
    labels: List[int] = []

    for label_index, class_name in enumerate(class_names):
        class_dir = root_path / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        class_files = _list_image_files(class_dir)
        for file_path in class_files:
            samples.append((file_path, label_index))
            labels.append(label_index)

    if not samples:
        raise ValueError(f"No images found under {root_path}")

    return samples, labels


def stratified_split_indices(labels: Sequence[int], val_split: float, seed: int):
    """Return stratified train/val indices for a label sequence."""
    indices = np.arange(len(labels))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_split,
        random_state=seed,
        stratify=labels,
        shuffle=True,
    )
    return train_indices.tolist(), val_indices.tolist()


def limit_samples_per_class(
    samples: Sequence[Tuple[Path, int]],
    labels: Sequence[int],
    max_samples_per_class: int,
    seed: int,
):
    """Limit samples per class for quick experiments, keeping class balance."""
    if max_samples_per_class <= 0:
        return list(samples), list(labels)

    rng = np.random.default_rng(seed)
    label_to_indices = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    limited_samples = []
    limited_labels = []
    for label, indices in label_to_indices.items():
        if len(indices) <= max_samples_per_class:
            chosen = indices
        else:
            chosen = rng.choice(indices, size=max_samples_per_class, replace=False).tolist()
        for idx in chosen:
            limited_samples.append(samples[idx])
            limited_labels.append(labels[idx])

    return limited_samples, limited_labels


def count_by_class(samples: Sequence[Tuple[Path, int]], class_names: Sequence[str]):
    """Count samples per class index and map to class names."""
    counts = {name: 0 for name in class_names}
    for _, label in samples:
        counts[class_names[label]] += 1
    return counts


def create_datasets(config: dict):
    """Create train/val/test datasets based on the provided config."""
    train_dataset, val_dataset, class_names = create_train_val_datasets(config)
    test_samples, _ = load_samples(config["paths"]["test_dir"], class_names)
    _, eval_transform = get_transforms(int(config["data"]["input_size"]))
    test_dataset = BCCDImageDataset(test_samples, class_names, transform=eval_transform)

    return train_dataset, val_dataset, test_dataset, class_names


def create_train_val_datasets(config: dict):
    """Create train and validation datasets without touching the test set."""
    class_names = list(config["data"]["class_names"])
    input_size = int(config["data"]["input_size"])
    val_split = float(config["data"]["val_split"])
    seed = int(config["project"]["seed"])

    train_dir = config["paths"]["train_dir"]

    train_transform, eval_transform = get_transforms(input_size)

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

    train_dataset = BCCDImageDataset(train_subset, class_names, transform=train_transform)
    val_dataset = BCCDImageDataset(val_subset, class_names, transform=eval_transform)

    return train_dataset, val_dataset, class_names


def create_dataloaders(config: dict):
    """Factory that returns stratified train/val loaders and test loader."""
    train_dataset, val_dataset, test_dataset, class_names = create_datasets(config)

    batch_size = int(config["data"]["batch_size"])
    num_workers = int(config["data"]["num_workers"])
    pin_memory = bool(config["data"]["pin_memory"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, class_names


def create_train_val_loaders(config: dict):
    """Factory that returns train/val loaders without accessing the test set."""
    train_dataset, val_dataset, class_names = create_train_val_datasets(config)

    batch_size = int(config["data"]["batch_size"])
    num_workers = int(config["data"]["num_workers"])
    pin_memory = bool(config["data"]["pin_memory"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, class_names


def create_test_loader(config: dict):
    """Create a test-only DataLoader from the held-out test set."""
    class_names = list(config["data"]["class_names"])
    input_size = int(config["data"]["input_size"])
    batch_size = int(config["data"]["batch_size"])
    num_workers = int(config["data"]["num_workers"])
    pin_memory = bool(config["data"]["pin_memory"])

    _, eval_transform = get_transforms(input_size)
    test_samples, _ = load_samples(config["paths"]["test_dir"], class_names)
    test_dataset = BCCDImageDataset(test_samples, class_names, transform=eval_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return test_loader, class_names
