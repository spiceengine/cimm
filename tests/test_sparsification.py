import pytest
import torch
import torch.nn.functional as F

from rasnatune.sparsification import (
    SparseActivationUnstructured,
    SparseWeightUnstructured,
    sparsify,
)


def _make_linear() -> torch.nn.Linear:
    module = torch.nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float32))
    return module


def _make_conv2d() -> torch.nn.Conv2d:
    module = torch.nn.Conv2d(1, 1, kernel_size=2, bias=False)
    with torch.no_grad():
        module.weight.copy_(
            torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]], dtype=torch.float32)
        )
    return module


def test_sparsify_returns_input_for_zero_sparsity() -> None:
    x = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float32)

    actual = sparsify(x, sparsity=0.0)

    torch.testing.assert_close(actual, x)


def test_sparsify_zeroes_smallest_magnitudes() -> None:
    x = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float32)

    actual = sparsify(x, sparsity=0.5)
    expected = torch.tensor([0.0, 0.0, 3.0, -4.0], dtype=torch.float32)

    torch.testing.assert_close(actual, expected)


def test_sparsify_zeros_all_values_for_full_sparsity() -> None:
    x = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float32)

    actual = sparsify(x, sparsity=1.0)

    torch.testing.assert_close(actual, torch.zeros_like(x))


def test_sparsify_rejects_invalid_sparsity() -> None:
    with pytest.raises(ValueError):
        sparsify(torch.ones(2, dtype=torch.float32), sparsity=1.5)


def test_sparse_weight_applies_and_restores_on_linear() -> None:
    module = _make_linear()
    compressor = SparseWeightUnstructured(sparsity=0.5)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    original_weight = module.weight.detach().clone()

    compressor.attach(module)
    actual = module(x)

    expected_weight = sparsify(original_weight, sparsity=0.5)
    expected = F.linear(x, expected_weight)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(module.weight.detach(), original_weight)


def test_sparse_weight_applies_and_restores_on_conv2d() -> None:
    module = _make_conv2d()
    compressor = SparseWeightUnstructured(sparsity=0.5)
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
    original_weight = module.weight.detach().clone()

    compressor.attach(module)
    actual = module(x)

    expected_weight = sparsify(original_weight, sparsity=0.5)
    expected = F.conv2d(x, expected_weight)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(module.weight.detach(), original_weight)


def test_sparse_activation_applies_to_linear_inputs() -> None:
    module = _make_linear()
    compressor = SparseActivationUnstructured(sparsity=0.5)
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float32)

    compressor.attach(module)
    actual = module(x)

    expected = F.linear(sparsify(x, sparsity=0.5), module.weight.detach())

    torch.testing.assert_close(actual, expected)

    compressor.detach(module)
    restored = module(x)
    torch.testing.assert_close(restored, F.linear(x, module.weight.detach()))
    assert not torch.equal(actual, restored)


def test_sparse_activation_applies_to_conv2d_inputs() -> None:
    module = _make_conv2d()
    compressor = SparseActivationUnstructured(sparsity=0.5)
    x = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]], dtype=torch.float32)

    compressor.attach(module)
    actual = module(x)

    expected = F.conv2d(sparsify(x, sparsity=0.5), module.weight.detach())

    torch.testing.assert_close(actual, expected)

    compressor.detach(module)
    restored = module(x)
    torch.testing.assert_close(restored, F.conv2d(x, module.weight.detach()))
    assert not torch.equal(actual, restored)
