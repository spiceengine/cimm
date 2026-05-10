import torch
import torch.nn.functional as F

from torchvision.models import efficientnet_b0, resnet50

from rasnatune import (
    Recipe,
    QuantizeActivation,
    QuantizeWeight,
    compress,
    compression_count,
    quantize,
    remove_compression,
    sparse,
)


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


def _run_compressed_training_step(
    model: torch.nn.Module,
    *recipes,
) -> None:
    modules = _supported_modules(model)
    first_module = modules[0]
    original_weight = first_module.weight.detach().clone()
    original_state_keys = list(model.state_dict().keys())

    returned = compress(model, *recipes)

    assert returned is model
    assert compression_count(model) == len(recipes) * len(modules)
    assert list(model.state_dict().keys()) == original_state_keys

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    inputs = torch.randn(2, 3, 64, 64, dtype=torch.float32)
    targets = torch.tensor([1, 3], dtype=torch.long)

    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all().item() for gradient in gradients)
    torch.testing.assert_close(first_module.weight.detach(), original_weight)

    parameters_before_step = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.step()

    assert any(
        not torch.equal(parameter.detach(), before_step)
        for parameter, before_step in zip(model.parameters(), parameters_before_step)
    )

    remove_compression(model)
    assert compression_count(model) == 0

    cleared_logits = model(inputs)
    assert cleared_logits.shape == logits.shape


def test_compress_applies_recipes_in_place_and_keeps_state_dict_keys() -> None:
    model = _make_toy_model()
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float32)
    state_keys = list(model.state_dict().keys())

    returned = compress(
        model,
        Recipe(QuantizeWeight, targets=torch.nn.Linear, kwargs={"min": -2, "max": 1}),
        Recipe(QuantizeActivation, targets=torch.nn.Linear, kwargs={"min": -2, "max": 1}),
    )

    assert returned is model
    assert compression_count(model) == 4
    assert list(model.state_dict().keys()) == state_keys

    compressed_output = model(x)
    remove_compression(model)

    assert compression_count(model) == 0
    restored_output = model(x)
    reference_output = _make_toy_model()(x)

    torch.testing.assert_close(restored_output, reference_output)
    assert not torch.equal(compressed_output, reference_output)


def test_compress_accepts_sparse_recipe_group() -> None:
    model = _make_toy_model()

    compress(model, sparse(0.25, targets=torch.nn.Linear))

    assert compression_count(model) == 2

    remove_compression(model)

    assert compression_count(model) == 0
    output = model(torch.randn(1, 4, dtype=torch.float32))
    assert output.shape == (1, 2)


def test_compress_without_recipes_leaves_model_unregistered() -> None:
    model = _make_toy_model()

    compress(model)

    assert compression_count(model) == 0
    output = model(torch.randn(1, 4, dtype=torch.float32))
    assert output.shape == (1, 2)

    remove_compression(model)


def test_resnet50_quantize_training_step_runs() -> None:
    torch.manual_seed(0)
    model = resnet50(weights=None, num_classes=5)
    model.train()

    _run_compressed_training_step(
        model,
        quantize(activation=8, weight=8, accumulator=17, targets=_supported_module_filter),
    )


def test_efficientnet_b0_quantize_training_step_runs() -> None:
    torch.manual_seed(0)
    model = efficientnet_b0(weights=None, num_classes=5)
    model.train()

    _run_compressed_training_step(
        model,
        quantize(activation=8, weight=8, accumulator=17, targets=_supported_module_filter),
    )


def test_resnet50_sparse_training_step_runs() -> None:
    torch.manual_seed(0)
    model = resnet50(weights=None, num_classes=5)
    model.train()

    _run_compressed_training_step(
        model,
        sparse(0.25, targets=_supported_module_filter),
    )


def test_efficientnet_b0_sparse_training_step_runs() -> None:
    torch.manual_seed(0)
    model = efficientnet_b0(weights=None, num_classes=5)
    model.train()

    _run_compressed_training_step(
        model,
        sparse(0.25, targets=_supported_module_filter),
    )
