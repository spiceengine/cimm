from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

LayerSelector = Callable[[torch.nn.Module], bool]

class CompressionBase:
    """Base interface for per-layer compression instances."""

    def attach(self, module: torch.nn.Module) -> list[torch.utils.hooks.RemovableHandle]:
        return []

    def detach(self, module: torch.nn.Module) -> None:
        return None


@dataclass(slots=True)
class _LayerRegistration:
    module: torch.nn.Module
    instance: CompressionBase
    handles: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)

    def clear(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.instance.detach(self.module)
        self.handles.clear()


class Compression(torch.nn.Module):
    """Wraps a model and manages hook-based compressions."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self._registrations: dict[type[CompressionBase], list[_LayerRegistration]] = {}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def attach(
        self,
        compressor: type[CompressionBase],
        apply_layers: LayerSelector | None = None,
        **compression_parameters,
    ) -> "Compression":
        self.detach(compressor)
        registrations: list[_LayerRegistration] = []

        layer_selector = apply_layers or (lambda _module: True)
        if not callable(layer_selector):
            raise TypeError("apply_layers must be callable[[torch.nn.Module], bool]")

        for name, module in self.model.named_modules():
            if not name:
                continue
            if not layer_selector(module):
                continue

            instance = compressor(**compression_parameters)
            handles = instance.attach(module)
            registrations.append(_LayerRegistration(module=module, instance=instance, handles=handles))

        self._registrations[compressor] = registrations
        return self

    def detach(self, compressor: type[CompressionBase]) -> "Compression":
        registrations = self._registrations.pop(compressor, [])
        for registration in registrations:
            registration.clear()
        return self

    def compression_clear(self) -> None:
        for compressor in list(self._registrations.keys()):
            self.detach(compressor)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

def compression(model: torch.nn.Module) -> Compression:
    return Compression(model)
