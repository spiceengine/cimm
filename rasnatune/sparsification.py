from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .base import Compressor
from .compression import Recipe


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
        self._mask = torch.ones_like(module.weight.data, dtype=torch.bool)

        def pre_hook(mod: torch.nn.Module, _inputs) -> None:
            self._backup = mod.weight.data.detach().clone()
            sparse_weight = sparsify(mod.weight.data, sparsity=self.sparsity)
            self._mask = sparse_weight.detach().ne(0)
            mod.weight.data.copy_(sparse_weight)

        def post_hook(mod: torch.nn.Module, _inputs, output):
            mod.weight.data.copy_(self._backup)
            return output

        def grad_hook(grad: torch.Tensor) -> torch.Tensor:
            return grad * self._mask.to(dtype=grad.dtype, device=grad.device)

        self._hooks = [
            module.register_forward_pre_hook(pre_hook),
            module.register_forward_hook(post_hook, always_call=True),
            module.weight.register_hook(grad_hook),
        ]


def sparse(
    sparsity: float = 0.5,
    *,
    targets: Callable[[torch.nn.Module], bool] | type[torch.nn.Module] | tuple[type[torch.nn.Module], ...],
) -> Recipe:
    return Recipe(SparseWeightUnstructured, targets=targets, kwargs={"sparsity": sparsity})
