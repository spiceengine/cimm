import torch
import torch.nn.functional as F

try:
    _TORCHVISION_NMS_LIB = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_NMS_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except RuntimeError:
    _TORCHVISION_NMS_LIB = None

from torchvision.models import efficientnet_b0, resnet50

from rasnatune.compression import Compression
from rasnatune.quantization import QuantizeActivation, QuantizeWeight
from rasnatune.sparsification import SparseActivationUnstructured, SparseWeightUnstructured


def _supported_module_filter(module: torch.nn.Module) -> bool:
    return isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))


def _supported_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    return [module for module in model.modules() if _supported_module_filter(module)]


def _make_toy_model() -> torch.nn.Sequential:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 2, bias=False),
    )
    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor(
                [
                    [1.75, -0.25, 0.5, -1.1],
                    [0.2, -0.6, 1.4, -0.8],
                    [0.9, 0.3, -0.4, 1.1],
                    [-1.2, 0.7, 0.6, -0.5],
                ],
                dtype=torch.float32,
            )
        )
        model[2].weight.copy_(
            torch.tensor(
                [
                    [0.8, -1.4, 0.6, -0.2],
                    [-0.5, 0.9, -1.1, 1.3],
                ],
                dtype=torch.float32,
            )
        )
    return model


def _attach_quantization(wrapped: Compression) -> None:
    wrapped.attach(QuantizeWeight, filter=_supported_module_filter, min=-8, max=7)
    wrapped.attach(QuantizeActivation, filter=_supported_module_filter, min=-8, max=7)


def _attach_sparsification(wrapped: Compression) -> None:
    wrapped.attach(SparseWeightUnstructured, filter=_supported_module_filter, sparsity=0.25)
    wrapped.attach(SparseActivationUnstructured, filter=_supported_module_filter, sparsity=0.25)


def _run_compressed_training_step(model: torch.nn.Module, attach_fn) -> None:
    wrapped = Compression(model)
    modules = _supported_modules(wrapped.model)
    first_module = modules[0]
    original_weight = first_module.weight.detach().clone()

    attach_fn(wrapped)

    assert len(wrapped.registrations) == 2 * len(modules)

    optimizer = torch.optim.SGD(wrapped.parameters(), lr=1e-2)
    inputs = torch.randn(2, 3, 64, 64, dtype=torch.float32)
    targets = torch.tensor([1, 3], dtype=torch.long)

    optimizer.zero_grad(set_to_none=True)
    logits = wrapped(inputs)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    gradients = [parameter.grad for parameter in wrapped.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all().item() for gradient in gradients)
    torch.testing.assert_close(first_module.weight.detach(), original_weight)

    parameters_before_step = [parameter.detach().clone() for parameter in wrapped.parameters()]
    optimizer.step()

    assert any(
        not torch.equal(parameter.detach(), before_step)
        for parameter, before_step in zip(wrapped.parameters(), parameters_before_step)
    )

    wrapped.clear()
    assert wrapped.registrations == []

    cleared_logits = wrapped(inputs)
    assert cleared_logits.shape == logits.shape


def test_compression_attach_and_detach_manage_registrations() -> None:
    model = _make_toy_model()
    wrapped = Compression(model)
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float32)

    wrapped.attach(QuantizeWeight, filter=lambda module: isinstance(module, torch.nn.Linear), min=-1, max=1)

    assert len(wrapped.registrations) == 2

    compressed_output = wrapped(x)
    wrapped.detach(wrapped.registrations[0])

    assert len(wrapped.registrations) == 1

    wrapped.clear()
    assert wrapped.registrations == []

    restored_output = wrapped(x)
    reference_output = model(x)

    torch.testing.assert_close(restored_output, reference_output)
    assert not torch.equal(compressed_output, reference_output)


def test_compression_clear_removes_sparse_registrations() -> None:
    model = _make_toy_model()
    wrapped = Compression(model)

    _attach_sparsification(wrapped)

    assert len(wrapped.registrations) == 4

    wrapped.clear()

    assert wrapped.registrations == []
    output = wrapped(torch.randn(1, 4, dtype=torch.float32))
    assert output.shape == (1, 2)


def test_resnet50_quantized_training_step_runs() -> None:
    torch.manual_seed(0)
    model = resnet50(weights=None, num_classes=5)
    model.train()

    _run_compressed_training_step(model, _attach_quantization)


def test_efficientnet_b0_quantized_training_step_runs() -> None:
    torch.manual_seed(0)
    model = efficientnet_b0(weights=None, num_classes=5)
    model.train()

    _run_compressed_training_step(model, _attach_quantization)


def test_resnet50_sparse_training_step_runs() -> None:
    torch.manual_seed(0)
    model = resnet50(weights=None, num_classes=5)
    model.train()

    _run_compressed_training_step(model, _attach_sparsification)


def test_efficientnet_b0_sparse_training_step_runs() -> None:
    torch.manual_seed(0)
    model = efficientnet_b0(weights=None, num_classes=5)
    model.train()

    _run_compressed_training_step(model, _attach_sparsification)
