from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .base import Compressor


def _registry(model: torch.nn.Module) -> list[Compressor]:
    registry = getattr(model, "_rasnatune_compressors", None)
    if registry is None:
        registry = []
        setattr(model, "_rasnatune_compressors", registry)
    return registry


@dataclass(frozen=True)
class Recipe:
    compressor: type[Compressor]
    targets: Callable[[torch.nn.Module], bool] | type[torch.nn.Module] | tuple[type[torch.nn.Module], ...]
    kwargs: dict = field(default_factory=dict)

    def apply(self, model: torch.nn.Module) -> list[Compressor]:
        if isinstance(self.targets, type):
            selected = lambda module: isinstance(module, self.targets)
        elif isinstance(self.targets, tuple) and all(isinstance(target, type) for target in self.targets):
            selected = lambda module: isinstance(module, self.targets)
        elif callable(self.targets):
            selected = self.targets
        else:
            raise TypeError("targets must be a module type, tuple of module types, or callable")

        registrations: list[Compressor] = []
        for module in model.modules():
            if not selected(module):
                continue
            compressor = self.compressor(**self.kwargs)
            compressor.attach(module)
            registrations.append(compressor)
        return registrations


def compress(model: torch.nn.Module, *recipes: Recipe) -> torch.nn.Module:
    registry = _registry(model)
    for recipe in recipes:
        registry.extend(recipe.apply(model))
    return model


def remove_compression(model: torch.nn.Module) -> torch.nn.Module:
    registry = _registry(model)
    for compressor in list(registry):
        compressor.detach(model)
    registry.clear()
    return model


def compression_count(model: torch.nn.Module) -> int:
    return len(_registry(model))
