from __future__ import annotations

from dataclasses import dataclass

import torch

from .compression import CompressionBase


@dataclass(slots=True)
class SparseUnstructuredWeights(CompressionBase):
    sparsity: float = 0.5
    min_elements: int = 64
    mask: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.sparsity <= 1.0:
            raise ValueError(f"sparsity must be in [0.0, 1.0], got {self.sparsity}")

    def attach(self, module: torch.nn.Module) -> list[torch.utils.hooks.RemovableHandle]:
        if not self.mask:
            return []

        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.nn.Parameter):
            return []
        if weight.numel() < self.min_elements:
            return []

        def pre_hook(mod: torch.nn.Module, _inputs) -> None:
            if not self.mask or self.sparsity <= 0.0:
                return

            w = getattr(mod, "weight", None)
            if w is None:
                return

            w_data = w.data
            n_total = w_data.numel()
            n_prune = int(n_total * self.sparsity)
            if n_prune <= 0:
                return
            if n_prune >= n_total:
                w_data.zero_()
                return

            flat_abs = w_data.abs().flatten()
            threshold = torch.kthvalue(flat_abs, n_prune).values
            sparsity_mask = w_data.abs() > threshold
            w_data.mul_(sparsity_mask.to(w_data.dtype))

        return [module.register_forward_pre_hook(pre_hook)]


@dataclass(slots=True)
class SparseUnstructuredActivations(CompressionBase):
    sparsity: float = 0.5
    min_elements: int = 64
    mask: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.sparsity <= 1.0:
            raise ValueError(f"sparsity must be in [0.0, 1.0], got {self.sparsity}")

    def attach(self, module: torch.nn.Module) -> list[torch.utils.hooks.RemovableHandle]:
        if not self.mask:
            return []

        def forward_hook(_module: torch.nn.Module, _inputs, output):
            def apply_one(x: torch.Tensor) -> torch.Tensor:
                if not self.mask or self.sparsity <= 0.0 or x.numel() < self.min_elements:
                    return x
                n_total = x.numel()
                n_prune = int(n_total * self.sparsity)
                if n_prune <= 0:
                    return x
                if n_prune >= n_total:
                    return torch.zeros_like(x)
                flat_abs = x.detach().abs().flatten()
                threshold = torch.kthvalue(flat_abs, n_prune).values
                sparsity_mask = x.detach().abs() > threshold
                return x * sparsity_mask.to(x.dtype)

            def apply_output(value):
                if isinstance(value, torch.Tensor):
                    return apply_one(value)
                if isinstance(value, tuple):
                    return tuple(apply_output(v) for v in value)
                if isinstance(value, list):
                    return [apply_output(v) for v in value]
                if isinstance(value, dict):
                    return {k: apply_output(v) for k, v in value.items()}
                return value

            return apply_output(output)

        return [module.register_forward_hook(forward_hook)]
