# cizm

`cizm` is a lightweight PyTorch compression helper based on forward hooks.

## Installation

```bash
pip install cizm
```

## What It Does

- Applies compression at runtime using hooks (no permanent weight rewrite).
- Supports quantization and unstructured sparsification.
- Targets `torch.nn.Conv2d` and `torch.nn.Linear`.

## Quick Start

```python
import torch
from cizm import Compression, SparseWeightUnstructured

model = torch.nn.Sequential(
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 10),
)

wrapped = Compression(model)
wrapped.attach(
    SparseWeightUnstructured,
    filter=lambda m: isinstance(m, torch.nn.Linear),
    sparsity=0.5,
)

x = torch.randn(128)
y = wrapped(x)
```

## Public API

Top-level exports:

- `cizm.Compression`
- `cizm.Compressor`
- `cizm.QuantizeWeight`
- `cizm.QuantizeActivation`
- `cizm.SparseWeightUnstructured`
- `cizm.SparseActivationUnstructured`

## Compression Classes

- `QuantizeWeight(min=-128, max=127)`:
  Quantizes layer weights only during forward, then restores original weights.
- `QuantizeActivation(min=-128, max=127)`:
  Quantizes the first input activation of the layer before forward.
- `SparseWeightUnstructured(sparsity=0.5)`:
  Applies unstructured sparsity to layer weights only during forward, then restores.
- `SparseActivationUnstructured(sparsity=0.5)`:
  Applies unstructured sparsity to the first input activation before forward.

## Notes

- `Compression.attach` uses `filter=` to choose target modules.
- If `filter` is omitted, it will try all submodules, and unsupported modules will raise an assertion.
- Supported module types for built-in compressors are `torch.nn.Conv2d` and `torch.nn.Linear`.
