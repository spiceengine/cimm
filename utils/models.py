from __future__ import annotations

import torch
import torchvision


def build_model(name: str, pretrained: bool) -> torch.nn.Module:
    if name == "resnet18":
        return torchvision.models.resnet18(weights="DEFAULT" if pretrained else None)
    if name == "resnet50":
        return torchvision.models.resnet50(weights="DEFAULT" if pretrained else None)
    raise ValueError(f"Unsupported model: {name}")
