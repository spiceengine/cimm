from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .base import Compressor
from .compression import Recipe
from .sparsification import sparsify


def _signed_range(bits: int) -> tuple[int, int]:
    if bits <= 0:
        raise ValueError(f"bits must be greater than 0, got {bits}")
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


def _quantization_scale(x: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    max_abs = x.detach().abs().max()
    if max_abs.item() == 0:
        return torch.ones((), dtype=x.dtype, device=x.device)
    return max_abs / max(qmax, -qmin)


def _quantize_tensor(x: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    scale = _quantization_scale(x, qmin=qmin, qmax=qmax)
    if x.detach().abs().max().item() == 0:
        return x
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    quantized = q * scale
    error = x.detach() - quantized
    return x - error


def quantize_accumulator(
    x: torch.Tensor,
    bits: int,
    scale: torch.Tensor | float,
) -> torch.Tensor:
    qmin, _qmax = _signed_range(bits)
    scale = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    if scale.detach().abs().max().item() == 0:
        scale = torch.ones((), dtype=x.dtype, device=x.device)

    modulo = 2 ** bits
    q = torch.round(x / scale)
    wrapped = torch.remainder(q - qmin, modulo) + qmin
    quantized = wrapped * scale
    error = x.detach() - quantized
    return x - error


@dataclass
class QuantizeWeight(Compressor):
    min: int = -128
    max: int = 127

    def attach(self, module: torch.nn.Module) -> None:
        def pre_hook(mod: torch.nn.Module, inputs) -> None:
            self._backup = mod.weight.data.detach().clone()
            mod.weight.data.copy_(_quantize_tensor(mod.weight.data, qmin=self.min, qmax=self.max))

        def post_hook(mod: torch.nn.Module, inputs, output):
            mod.weight.data.copy_(self._backup)
            return output

        self._hooks = [
            module.register_forward_pre_hook(pre_hook),
            module.register_forward_hook(post_hook, always_call=True)
        ]


@dataclass
class QuantizeActivation(Compressor):
    min: int = -128
    max: int = 127

    def attach(self, module: torch.nn.Module) -> None:
        def pre_hook(_module: torch.nn.Module, inputs):
            return (_quantize_tensor(inputs[0], qmin=self.min, qmax=self.max), *inputs[1:])

        self._hooks = [
            module.register_forward_pre_hook(pre_hook)
        ]


@dataclass
class QuantizeMAC(Compressor):
    activation_bits: int = 8
    weight_bits: int = 8
    accumulator_bits: int = 17
    overflow: str = "wrap"
    weight_sparsity: float = 0.0

    def attach(self, module: torch.nn.Module) -> None:
        if self.overflow != "wrap":
            raise ValueError(f"unsupported overflow mode: {self.overflow}")

        def pre_hook(mod: torch.nn.Module, inputs):
            activation_qmin, activation_qmax = _signed_range(self.activation_bits)
            weight_qmin, weight_qmax = _signed_range(self.weight_bits)
            activation_scale = _quantization_scale(
                inputs[0],
                qmin=activation_qmin,
                qmax=activation_qmax,
            )
            weight_scale = _quantization_scale(
                mod.weight.data,
                qmin=weight_qmin,
                qmax=weight_qmax,
            )

            self._accumulator_scale = activation_scale * weight_scale
            self._backup = mod.weight.data.detach().clone()
            weight = mod.weight.data
            if self.weight_sparsity > 0.0:
                weight = sparsify(weight, sparsity=self.weight_sparsity)
            mod.weight.data.copy_(
                _quantize_tensor(weight, qmin=weight_qmin, qmax=weight_qmax)
            )
            return (
                _quantize_tensor(inputs[0], qmin=activation_qmin, qmax=activation_qmax),
                *inputs[1:],
            )

        def post_hook(mod: torch.nn.Module, _inputs, output):
            mod.weight.data.copy_(self._backup)
            if not isinstance(output, torch.Tensor):
                return output
            return quantize_accumulator(
                output,
                bits=self.accumulator_bits,
                scale=self._accumulator_scale,
            )

        self._hooks = [
            module.register_forward_pre_hook(pre_hook),
            module.register_forward_hook(post_hook, always_call=True),
        ]


def quantize(
    activation: int = 8,
    weight: int = 8,
    accumulator: int = 17,
    *,
    targets: Callable[[torch.nn.Module], bool] | type[torch.nn.Module] | tuple[type[torch.nn.Module], ...],
    overflow: str = "wrap",
    weight_sparsity: float = 0.0,
) -> Recipe:
    return Recipe(
        QuantizeMAC,
        targets=targets,
        kwargs={
            "activation_bits": activation,
            "weight_bits": weight,
            "accumulator_bits": accumulator,
            "overflow": overflow,
            "weight_sparsity": weight_sparsity,
        },
    )
