from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import Compressor


def sparsify(x: torch.Tensor, sparsity: float) -> torch.Tensor:
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be between 0 and 1, got {sparsity}")

    flat_abs = x.detach().abs().flatten()
    if flat_abs.numel() == 0 or sparsity == 0.0:
        return x
    if sparsity == 1.0:
        return torch.zeros_like(x)

    num_zero = int(flat_abs.numel() * sparsity)
    if num_zero == 0:
        return x

    zero_indices = flat_abs.topk(num_zero, largest=False).indices
    sparsity_mask = torch.ones_like(flat_abs, dtype=torch.bool)
    sparsity_mask[zero_indices] = False
    return x * sparsity_mask.view_as(x).to(x.dtype)


@dataclass
class SparseWeightUnstructured(Compressor):
    sparsity: float = 0.5

    def attach(self, module: torch.nn.Module) -> None:
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
        def pre_hook(_module: torch.nn.Module, inputs):
            return (sparsify(inputs[0], sparsity=self.sparsity), *inputs[1:])

        self._hooks = [
            module.register_forward_pre_hook(pre_hook),
        ]
