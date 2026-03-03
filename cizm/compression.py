from __future__ import annotations
from pygments.unistring import Co

from dataclasses import dataclass
from typing import Callable

import torch

from .base import Compressor


class Compression(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.registrations: list[Compressor] = []

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def attach(self, compressor: type[Compressor], filter: Callable, *args, **kwargs,):
        for name, module in self.model.named_modules():
            if not filter(module):
                continue
            instance = compressor(*args, **kwargs)
            instance.attach(module)
            self.registrations.append(instance)

    def clear(self):
        for compressor in self.registrations:
            self.detach(compressor)
        self.registrations.clear()
