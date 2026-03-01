from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .compression import CompressionBase


def quantize_tensor(x: torch.Tensor, bits: int, eps: float = 1e-8) -> torch.Tensor:
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")

    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax - 1
    scale = x.detach().abs().max() / max(qmax, 1)
    scale = torch.clamp(scale, min=eps)
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    return q * scale


def fake_quantize_tensor(x: torch.Tensor, bits: int, eps: float = 1e-8) -> torch.Tensor:
    quantized = quantize_tensor(x, bits=bits, eps=eps)
    quantization_error = x.detach() - quantized
    return x - quantization_error


@dataclass(slots=True)
class QuantizeFakeWeights(CompressionBase):
    bits: int = 8
    eps: float = 1e-8
    _weight_backup: dict[int, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bits < 2:
            raise ValueError(f"bits must be >= 2, got {self.bits}")

    def attach(self, module: torch.nn.Module) -> list[torch.utils.hooks.RemovableHandle]:
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.nn.Parameter):
            return []

        def pre_hook(mod: torch.nn.Module, _inputs) -> None:
            w = getattr(mod, "weight", None)
            if not isinstance(w, torch.nn.Parameter):
                return
            module_id = id(mod)
            self._weight_backup[module_id] = w.data.detach().clone()
            w.data.copy_(fake_quantize_tensor(w.data, bits=self.bits, eps=self.eps))

        def post_hook(mod: torch.nn.Module, _inputs, output):
            w = getattr(mod, "weight", None)
            if not isinstance(w, torch.nn.Parameter):
                return output
            backup = self._weight_backup.pop(id(mod), None)
            if backup is not None:
                w.data.copy_(backup)
            return output

        return [
            module.register_forward_pre_hook(pre_hook),
            module.register_forward_hook(post_hook),
        ]

    def detach(self, module: torch.nn.Module) -> None:
        w = getattr(module, "weight", None)
        if not isinstance(w, torch.nn.Parameter):
            return
        backup = self._weight_backup.pop(id(module), None)
        if backup is not None:
            w.data.copy_(backup)


@dataclass(slots=True)
class QuantizeFakeActivations(CompressionBase):
    bits: int = 8
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.bits < 2:
            raise ValueError(f"bits must be >= 2, got {self.bits}")

    def attach(self, module: torch.nn.Module) -> list[torch.utils.hooks.RemovableHandle]:
        def forward_hook(_module: torch.nn.Module, _inputs, output):
            def apply_output(value):
                if isinstance(value, torch.Tensor):
                    return fake_quantize_tensor(value, bits=self.bits, eps=self.eps)
                if isinstance(value, tuple):
                    return tuple(apply_output(v) for v in value)
                if isinstance(value, list):
                    return [apply_output(v) for v in value]
                if isinstance(value, dict):
                    return {k: apply_output(v) for k, v in value.items()}
                return value

            return apply_output(output)

        return [module.register_forward_hook(forward_hook)]
