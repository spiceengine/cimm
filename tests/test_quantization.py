import unittest

import torch

from cizm.quantization import QuantizeActivation, QuantizeWeight, quantize


class WeightedDot(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.75, -0.25], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.dot(x, self.weight)


class AddAndScale(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        offset: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return (x + offset) * scale


class QuantizeFunctionTests(unittest.TestCase):
    def test_quantize_maps_values_to_expected_levels(self) -> None:
        x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32)

        actual = quantize(x, qmin=-2, qmax=2)
        expected = torch.tensor([-3.0, -1.5, 0.0, 1.5, 3.0], dtype=torch.float32)

        torch.testing.assert_close(actual, expected)

    def test_quantize_clamps_to_asymmetric_range(self) -> None:
        x = torch.tensor([-2.0, -0.6, 0.2, 1.7], dtype=torch.float32)

        actual = quantize(x, qmin=-1, qmax=2)
        expected = torch.tensor([-1.0, -1.0, 0.0, 2.0], dtype=torch.float32)

        torch.testing.assert_close(actual, expected)

    def test_quantize_preserves_straight_through_gradients(self) -> None:
        x = torch.tensor([0.3, -0.6], dtype=torch.float32, requires_grad=True)
        upstream = torch.tensor([2.0, -3.0], dtype=torch.float32)

        loss = (quantize(x, qmin=-2, qmax=1) * upstream).sum()
        loss.backward()

        torch.testing.assert_close(x.grad, upstream)

    def test_quantize_zero_tensor_returns_zero_without_nan(self) -> None:
        x = torch.zeros(4, dtype=torch.float32, requires_grad=True)

        actual = quantize(x, qmin=-128, qmax=127)
        actual.sum().backward()

        torch.testing.assert_close(actual, torch.zeros_like(x))
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        self.assertTrue(torch.isfinite(actual).all().item())


class QuantizeWeightTests(unittest.TestCase):
    def test_attach_uses_quantized_weight_for_forward_and_restores_weight(self) -> None:
        module = WeightedDot()
        compressor = QuantizeWeight(min=-1, max=1)
        original_weight = module.weight.detach().clone()
        x = torch.tensor([2.0, 3.0], dtype=torch.float32)

        compressor.attach(module)
        actual = module(x)

        expected_quantized_weight = torch.tensor([1.75, 0.0], dtype=torch.float32)
        expected_output = torch.dot(x, expected_quantized_weight)

        torch.testing.assert_close(actual, expected_output)
        torch.testing.assert_close(module.weight.detach(), original_weight)

    def test_detach_disables_weight_quantization(self) -> None:
        module = WeightedDot()
        compressor = QuantizeWeight(min=-1, max=1)
        x = torch.tensor([2.0, 3.0], dtype=torch.float32)

        compressor.attach(module)
        quantized_output = module(x)
        compressor.detach(module)
        restored_output = module(x)

        self.assertNotEqual(quantized_output.item(), restored_output.item())
        self.assertAlmostEqual(restored_output.item(), 2.75)


class QuantizeActivationTests(unittest.TestCase):
    def test_attach_quantizes_only_first_positional_input(self) -> None:
        module = AddAndScale()
        compressor = QuantizeActivation(min=-1, max=2)
        x = torch.tensor([-2.2, -0.1, 0.6], dtype=torch.float32)
        offset = torch.tensor([0.4, -0.3, 0.1], dtype=torch.float32)
        scale = torch.tensor(1.5, dtype=torch.float32)

        compressor.attach(module)
        actual = module(x, offset, scale)

        expected_quantized_x = torch.tensor([-1.1, 0.0, 1.1], dtype=torch.float32)
        expected = (expected_quantized_x + offset) * scale

        torch.testing.assert_close(actual, expected)

    def test_detach_disables_activation_quantization(self) -> None:
        module = AddAndScale()
        compressor = QuantizeActivation(min=-1, max=2)
        x = torch.tensor([-2.2, -0.1, 0.6], dtype=torch.float32)
        offset = torch.tensor([0.4, -0.3, 0.1], dtype=torch.float32)
        scale = torch.tensor(1.5, dtype=torch.float32)

        compressor.attach(module)
        quantized_output = module(x, offset, scale)
        compressor.detach(module)
        restored_output = module(x, offset, scale)

        self.assertFalse(torch.equal(quantized_output, restored_output))
        torch.testing.assert_close(restored_output, (x + offset) * scale)


if __name__ == "__main__":
    unittest.main()
