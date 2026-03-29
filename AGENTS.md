# AGENTS.md

## Project Snapshot

- `cizm` is a lightweight PyTorch compression helper built around forward hooks.
- The package exposes temporary runtime transformations rather than permanently rewriting model weights.
- Current packaged version is `0.0.1` in [`pyproject.toml`](pyproject.toml).
- The codebase is intentionally small: one wrapper module, one abstract base, and two built-in compression families.

## Repository Layout

- [`cizm/__init__.py`](cizm/__init__.py): top-level public exports.
- [`cizm/base.py`](cizm/base.py): `Compressor` base class and hook removal helper.
- [`cizm/compression.py`](cizm/compression.py): `Compression` wrapper that walks a model, attaches compressors, and forwards calls to the wrapped model.
- [`cizm/quantization.py`](cizm/quantization.py): quantization utility plus weight/activation quantization hooks.
- [`cizm/sparsification.py`](cizm/sparsification.py): sparsification utility plus weight/activation sparsity hooks.
- [`tests/test_quantization.py`](tests/test_quantization.py): current automated test coverage.
- [`README.md`](README.md): installation and API overview.
- [`Makefile`](Makefile): local environment bootstrap and test shortcut.
- [`.github/workflows/pypi-publish.yml`](.github/workflows/pypi-publish.yml): builds and publishes distributions on `v*` tags.

## Packaging And Runtime Dependencies

- Package metadata lives in [`pyproject.toml`](pyproject.toml).
- The only declared runtime dependency is `torch>=2.0`.
- There is no declared test dependency group yet.
- Prebuilt artifacts currently exist under [`dist/`](dist).

## Core Architecture

### `Compressor`

- `Compressor` in [`cizm/base.py`](cizm/base.py) is the minimal abstraction.
- Subclasses are expected to implement `attach(module)`.
- Hook handles are stored in `self._hooks`.
- `detach()` removes those hook handles, but does not clear metadata or restore any state on its own.

### `Compression`

- `Compression` in [`cizm/compression.py`](cizm/compression.py) wraps an arbitrary `torch.nn.Module`.
- `forward()` simply delegates to the wrapped model.
- `attach(compressor_cls, filter, *args, **kwargs)` walks `self.model.named_modules()`, instantiates one compressor per matching module, and stores the instances in `self.registrations`.
- The design assumes callers choose target modules with `filter`.

### Quantization Path

- `quantize(x, qmin, qmax)` computes a scale from the maximum absolute value, rounds into the requested integer range, and returns a straight-through-estimator style tensor.
- `QuantizeWeight` temporarily rewrites `module.weight.data` in a forward pre-hook, then restores the original weight in a forward hook.
- `QuantizeActivation` rewrites only the first positional input in a forward pre-hook.

### Sparsification Path

- `sparsify(x, sparsity)` computes an absolute-value threshold via `quantile` and zeroes entries below or equal to that threshold.
- `SparseWeightUnstructured` follows the same backup/mutate/restore hook pattern as `QuantizeWeight`.
- `SparseActivationUnstructured` rewrites only the first positional input in a forward pre-hook.

## Makefile And Local Workflow

The current [`Makefile`](Makefile) defines two targets:

- `make activate`
- Creates `./.venv` with `python3 -m venv`.
- Installs `torch` and `torchvision`.

- `make test`
- Runs `./.venv/bin/pytest tests/`.

Important caveat:

- `make activate` does not install `pytest`, so `make test` currently fails unless `pytest` is installed manually into `./.venv`.

## Current Test Coverage

Automated coverage is currently limited to quantization:

- `quantize()` numeric behavior for symmetric and asymmetric ranges.
- `quantize()` straight-through gradient behavior.
- `quantize()` zero-input stability.
- `QuantizeWeight` successful forward behavior and weight restoration after normal execution.
- `QuantizeWeight.detach()` behavior.
- `QuantizeActivation` first-positional-input rewriting and detach behavior.

What is not covered yet:

- `Compression.attach()` and `Compression.clear()`.
- Any sparsification behavior.
- Exception safety for weight compressors.
- README examples as executable integration tests.

## Known Issues And Current Caveats

These are worth knowing before making changes.

- `Compression.clear()` is broken in the current implementation.
- In `cizm/compression.py:28`, it calls `self.detach(compressor)`, but `Compression` does not define `detach()`.
- In practice, calling `clear()` after registering compressors raises `AttributeError`.

- `sparsify(..., sparsity=0.0)` is not a no-op.
- In `cizm/sparsification.py:11` and `cizm/sparsification.py:12`, the threshold is the minimum absolute value and the mask uses `> threshold`.
- Example observed locally: `tensor([1.0, 2.0])` becomes `[0.0, 2.0]` at `sparsity=0.0`.
- Tied magnitudes can also cause achieved sparsity to overshoot the requested value.

- Weight restoration is not exception-safe.
- `cizm/quantization.py:31` and `cizm/sparsification.py:27` restore weights in regular forward hooks.
- If the wrapped module raises during `forward()`, those hooks do not restore the original weight under the current registration style.
- This leaves quantized or sparsified weights resident on the module after a failed forward.

- README and code are slightly out of sync.
- `README.md:64` says `filter` can be omitted, but `cizm/compression.py:20` requires it.
- `README.md:65` and `README.md:66` describe assertion-based validation for unsupported modules, but the built-in compressors currently do not perform those assertions.

- Activation compressors only rewrite the first positional input.
- This is intentional in the current implementation, but easy to miss if a module takes multiple tensor arguments or keyword-only tensors.

## Practical Guidance For Future Changes

- If you modify hook behavior, test normal forward, detach/cleanup, and exception during forward.
- If you touch `Compression`, add direct tests for registration count, module selection via `filter`, and cleanup behavior.
- If you change compressor applicability, update both [`README.md`](README.md) and the relevant tests under [`tests/`](tests).
- If you keep `make test`, also make sure the environment setup installs the tools it expects.
- Before refactoring sparsification, lock in intended semantics for `sparsity=0.0`, `sparsity=1.0`, and duplicate magnitudes around the quantile threshold.

## Release Flow

- Packaging uses `setuptools.build_meta`.
- GitHub Actions publishes to PyPI when a tag matching `v*` is pushed.
- The workflow builds distributions with `python -m build` and publishes via `pypa/gh-action-pypi-publish`.

## Recommended Next Steps

- Fix `Compression.clear()` and add direct tests for the wrapper class.
- Add a full test module for sparsification.
- Make weight restoration exception-safe.
- Align README claims with the current implementation.
- Decide whether `Makefile` should install test dependencies or whether testing should move to a documented optional dependency group.
