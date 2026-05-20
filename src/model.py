# File: src/model.py
# Purpose: Build a VGG-19 model with an ImageNet-pretrained backbone and 4-class head.
# Author: Aman Verma (23bcs012), Ankit Kashyap (23bcs017), Abhishek Chauhan (23bcs003)
# Affiliation: School of Computer science and Engineering, Shri Mata Vaishno Devi University
# Contact: info@smvdu.ac.in

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_vgg19(num_classes: int, pretrained: bool = True, dropout: float = 0.5, freeze_features: bool = False):
    """Build a VGG-19 model with a replaced classifier head."""
    weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg19(weights=weights)

    if dropout != 0.5:
        if isinstance(model.classifier[2], nn.Dropout):
            model.classifier[2].p = dropout
        if isinstance(model.classifier[5], nn.Dropout):
            model.classifier[5].p = dropout

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    return model
