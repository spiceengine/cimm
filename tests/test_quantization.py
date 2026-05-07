import torch
import torch.nn.functional as F

from rasnatune.quantization import QuantizeActivation, QuantizeWeight, quantize


def _make_linear() -> torch.nn.Linear:
    module = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[1.75, -0.25]], dtype=torch.float32))
    return module


def _make_conv2d() -> torch.nn.Conv2d:
    module = torch.nn.Conv2d(1, 1, kernel_size=2, bias=False)
    with torch.no_grad():
        module.weight.copy_(
            torch.tensor([[[[1.75, -0.25], [0.5, -1.1]]]], dtype=torch.float32)
        )
    return module


def test_quantize_maps_values_to_expected_levels() -> None:
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32)

    actual = quantize(x, qmin=-2, qmax=2)
    expected = torch.tensor([-3.0, -1.5, 0.0, 1.5, 3.0], dtype=torch.float32)

    torch.testing.assert_close(actual, expected)


def test_quantize_clamps_to_asymmetric_range() -> None:
    x = torch.tensor([-2.0, -0.6, 0.2, 1.7], dtype=torch.float32)

    actual = quantize(x, qmin=-1, qmax=2)
    expected = torch.tensor([-1.0, -1.0, 0.0, 2.0], dtype=torch.float32)

    torch.testing.assert_close(actual, expected)


def test_quantize_preserves_straight_through_gradients() -> None:
    x = torch.tensor([0.3, -0.6], dtype=torch.float32, requires_grad=True)
    upstream = torch.tensor([2.0, -3.0], dtype=torch.float32)

    loss = (quantize(x, qmin=-2, qmax=1) * upstream).sum()
    loss.backward()

    torch.testing.assert_close(x.grad, upstream)


def test_quantize_zero_tensor_returns_zero_without_nan() -> None:
    x = torch.zeros(4, dtype=torch.float32, requires_grad=True)

    actual = quantize(x, qmin=-128, qmax=127)
    actual.sum().backward()

    torch.testing.assert_close(actual, torch.zeros_like(x))
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    assert torch.isfinite(actual).all().item()


def test_quantize_weight_applies_and_restores_on_linear() -> None:
    module = _make_linear()
    compressor = QuantizeWeight(min=-1, max=1)
    x = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
    original_weight = module.weight.detach().clone()

    compressor.attach(module)
    actual = module(x)

    expected_weight = quantize(original_weight, qmin=-1, qmax=1)
    expected = F.linear(x, expected_weight)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(module.weight.detach(), original_weight)


def test_quantize_weight_applies_and_restores_on_conv2d() -> None:
    module = _make_conv2d()
    compressor = QuantizeWeight(min=-1, max=1)
    x = torch.tensor([[[[1.0, 0.5], [-0.25, 0.75]]]], dtype=torch.float32)
    original_weight = module.weight.detach().clone()

    compressor.attach(module)
    actual = module(x)

    expected_weight = quantize(original_weight, qmin=-1, qmax=1)
    expected = F.conv2d(x, expected_weight)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(module.weight.detach(), original_weight)


def test_quantize_activation_applies_to_linear_inputs() -> None:
    module = _make_linear()
    compressor = QuantizeActivation(min=-1, max=1)
    x = torch.tensor([[1.75, -0.25]], dtype=torch.float32)

    compressor.attach(module)
    actual = module(x)

    expected = F.linear(quantize(x, qmin=-1, qmax=1), module.weight.detach())

    torch.testing.assert_close(actual, expected)

    compressor.detach(module)
    restored = module(x)
    torch.testing.assert_close(restored, F.linear(x, module.weight.detach()))
    assert not torch.equal(actual, restored)


def test_quantize_activation_applies_to_conv2d_inputs() -> None:
    module = _make_conv2d()
    compressor = QuantizeActivation(min=-1, max=1)
    x = torch.tensor([[[[1.75, -0.25], [0.5, -1.1]]]], dtype=torch.float32)

    compressor.attach(module)
    actual = module(x)

    expected = F.conv2d(quantize(x, qmin=-1, qmax=1), module.weight.detach())

    torch.testing.assert_close(actual, expected)

    compressor.detach(module)
    restored = module(x)
    torch.testing.assert_close(restored, F.conv2d(x, module.weight.detach()))
    assert not torch.equal(actual, restored)
