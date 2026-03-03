from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import Compressor

_SUPPORTED_SPARSIFICATION_MODULES = (torch.nn.Conv2d, torch.nn.Linear)


def sparsify(x: torch.Tensor, sparsity: float) -> torch.Tensor:
    threshold = x.detach().abs().flatten().quantile(sparsity)
    sparsity_mask = x.detach().abs() > threshold
    return x * sparsity_mask.to(x.dtype)


@dataclass
class SparseWeightUnstructured(Compressor):
    sparsity: float = 0.5

    def attach(self, module: torch.nn.Module) -> None:
        assert isinstance(module, _SUPPORTED_SPARSIFICATION_MODULES), f"{self.__class__.__name__} supports only Conv2d/Linear, got {module.__class__.__name__}"

        def pre_hook(mod: torch.nn.Module, _inputs) -> None:
            self._backup = mod.weight.data.detach().clone()
            mod.weight.data.copy_(
                sparsify(mod.weight.data, sparsity=self.sparsity)
            )

        def post_hook(mod: torch.nn.Module, inputs, output):
            mod.weight.data.copy_(self._backup)
            return output

        self._hooks = [
            module.register_forward_pre_hook(pre_hook),
            module.register_forward_hook(post_hook),
        ]


@dataclass
class SparseActivationUnstructured(Compressor):
    sparsity: float = 0.5

    def attach(self, module: torch.nn.Module) -> None:
        assert isinstance(module, _SUPPORTED_SPARSIFICATION_MODULES), f"{self.__class__.__name__} supports only Conv2d/Linear, got {module.__class__.__name__}"

        def pre_hook(_module: torch.nn.Module, inputs):
            return (sparsify(inputs[0], sparsity=self.sparsity), *inputs[1:])

        self._hooks = [
            module.register_forward_pre_hook(pre_hook),
        ]
