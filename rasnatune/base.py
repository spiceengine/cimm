from __future__ import annotations

import torch


class Compressor:
    _backup: torch.Tensor
    _hooks: list[torch.utils.hooks.RemovableHandle]

    def attach(self, module: torch.nn.Module) -> None:
        raise NotImplementedError

    def detach(self, module: torch.nn.Module) -> None:
        for hook in self._hooks:
            hook.remove()
