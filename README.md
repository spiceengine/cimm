# cimm

`cimm` は `Compression Image Models` の略です。
PyTorch向けの軽量な圧縮・蒸留ユーティリティです。

## Install

```bash
pip install cimm
```

## Quick Start

```python
import torch
from cimm import Compression, SparseUnstructuredWeights

model = torch.nn.Linear(128, 10)
wrapped = Compression(model)
wrapped.attach(SparseUnstructuredWeights, sparsity=0.5)
```

## Main APIs

- `cimm.Compression`
- `cimm.QuantizeFakeWeights`
- `cimm.QuantizeFakeActivations`
- `cimm.SparseUnstructuredWeights`
- `cimm.SparseUnstructuredActivations`
- `cimm.Distillation`

## Build and Upload

```bash
python -m pip install -U build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

本番公開前に TestPyPI での確認を推奨します。
