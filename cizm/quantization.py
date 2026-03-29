from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .base import Compressor


def quantize(x: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    max_abs = x.detach().abs().max()
    if max_abs.item() == 0:
        return x
    scale = max_abs / max(qmax, -qmin)
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    quantized = q * scale
    error = x.detach() - quantized
    return x - error


@dataclass
class QuantizeWeight(Compressor):
    min: int = -128
    max: int = 127

    def attach(self, module: torch.nn.Module) -> None:
        def pre_hook(mod: torch.nn.Module, inputs) -> None:
            self._backup = mod.weight.data.detach().clone()
            mod.weight.data.copy_(quantize(mod.weight.data, qmin=self.min, qmax=self.max))

        def post_hook(mod: torch.nn.Module, inputs, output):
            mod.weight.data.copy_(self._backup)
            return output

        self._hooks = [
            module.register_forward_pre_hook(pre_hook),
            module.register_forward_hook(post_hook)
        ]


@dataclass
class QuantizeActivation(Compressor):
    min: int = -128
    max: int = 127

    def attach(self, module: torch.nn.Module) -> None:
        def pre_hook(_module: torch.nn.Module, inputs):
            return (quantize(inputs[0], qmin=self.min, qmax=self.max), *inputs[1:])

        self._hooks = [
            module.register_forward_pre_hook(pre_hook)
        ]
