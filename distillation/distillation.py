from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class DistillationLosses:
    total: torch.Tensor
    student: torch.Tensor
    teacher_l1: torch.Tensor


def _collect_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, tuple):
        tensors: list[torch.Tensor] = []
        for item in value:
            tensors.extend(_collect_tensors(item))
        return tensors
    if isinstance(value, list):
        tensors = []
        for item in value:
            tensors.extend(_collect_tensors(item))
        return tensors
    if isinstance(value, dict):
        tensors = []
        for item in value.values():
            tensors.extend(_collect_tensors(item))
        return tensors
    return []


class Distillation(torch.nn.Module):
    """Wraps student/teacher models and computes weighted distillation loss."""

    def __init__(
        self,
        student_model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        *,
        weight: float = 0.5,
        criterion: torch.nn.Module | None = None,
        freeze_teacher: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {weight}")

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.weight = float(weight)
        self.criterion = criterion
        self.freeze_teacher = freeze_teacher

        if self.freeze_teacher:
            self.teacher_model.eval()
            for parameter in self.teacher_model.parameters():
                parameter.requires_grad_(False)

    def forward(self, *args, **kwargs):
        return self.student_model(*args, **kwargs)

    def teacher_forward(self, *args, **kwargs):
        if self.freeze_teacher:
            with torch.no_grad():
                return self.teacher_model(*args, **kwargs)
        return self.teacher_model(*args, **kwargs)

    def set_weight(self, weight: float) -> "Distillation":
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {weight}")
        self.weight = float(weight)
        return self

    def _teacher_l1_loss(self, student_output: Any, teacher_output: Any) -> torch.Tensor:
        student_tensors = _collect_tensors(student_output)
        teacher_tensors = _collect_tensors(teacher_output)

        if not student_tensors:
            raise ValueError("student_output does not contain any torch.Tensor")
        if not teacher_tensors:
            raise ValueError("teacher_output does not contain any torch.Tensor")
        if len(student_tensors) != len(teacher_tensors):
            raise ValueError(
                f"student/teacher output tensor count mismatch: {len(student_tensors)} != {len(teacher_tensors)}"
            )

        l1_losses: list[torch.Tensor] = []
        for index, (student_tensor, teacher_tensor) in enumerate(zip(student_tensors, teacher_tensors)):
            if student_tensor.shape != teacher_tensor.shape:
                raise ValueError(
                    f"student/teacher output shape mismatch at tensor {index}: "
                    f"{tuple(student_tensor.shape)} != {tuple(teacher_tensor.shape)}"
                )
            l1_losses.append(F.l1_loss(student_tensor, teacher_tensor.detach(), reduction="mean"))

        return torch.stack(l1_losses).mean()

    def loss(
        self,
        student_output: Any,
        *,
        target: torch.Tensor | None = None,
        inputs: torch.Tensor | tuple[Any, ...] | None = None,
        student_loss: torch.Tensor | None = None,
        teacher_output: Any | None = None,
    ) -> DistillationLosses:
        if student_loss is None:
            if self.criterion is None:
                raise ValueError("criterion is not set. Provide student_loss or set criterion in Distillation.")
            if target is None:
                raise ValueError("target is required when student_loss is not provided.")
            student_loss = self.criterion(student_output, target)

        if teacher_output is None:
            if inputs is None:
                raise ValueError("inputs is required when teacher_output is not provided.")
            if isinstance(inputs, tuple):
                teacher_output = self.teacher_forward(*inputs)
            else:
                teacher_output = self.teacher_forward(inputs)

        teacher_l1 = self._teacher_l1_loss(student_output, teacher_output)
        total = student_loss * (1.0 - self.weight) + teacher_l1 * self.weight
        return DistillationLosses(total=total, student=student_loss, teacher_l1=teacher_l1)

    def forward_with_loss(
        self,
        inputs: torch.Tensor | tuple[Any, ...],
        target: torch.Tensor,
        *,
        student_loss: torch.Tensor | None = None,
    ) -> tuple[Any, DistillationLosses]:
        if isinstance(inputs, tuple):
            student_output = self.forward(*inputs)
            teacher_output = self.teacher_forward(*inputs)
        else:
            student_output = self.forward(inputs)
            teacher_output = self.teacher_forward(inputs)

        losses = self.loss(
            student_output,
            target=target,
            student_loss=student_loss,
            teacher_output=teacher_output,
        )
        return student_output, losses


def distillation(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    *,
    weight: float = 0.5,
    criterion: torch.nn.Module | None = None,
    freeze_teacher: bool = True,
) -> Distillation:
    return Distillation(
        student_model=student_model,
        teacher_model=teacher_model,
        weight=weight,
        criterion=criterion,
        freeze_teacher=freeze_teacher,
    )
