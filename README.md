# cizm - Compression Image Models

`cizm` は PyTorch 向けの軽量な圧縮ユーティリティです。

## Install

```bash
pip install cizm
```

## Overview

- Hook ベースでモデルに圧縮を適用
- 対応圧縮: 量子化 / 非構造スパース化
- 対象レイヤ: `torch.nn.Conv2d`, `torch.nn.Linear`

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
    apply_layers=lambda m: isinstance(m, torch.nn.Linear),
    sparsity=0.5,
)

x = torch.randn(4, 128)
y = wrapped(x)
```

## APIs

- `cizm.Compression`
- `cizm.Compressor`
- `cizm.QuantizeWeight`
- `cizm.QuantizeActivation`
- `cizm.SparseWeightUnstructured`
- `cizm.SparseActivationUnstructured`
- `cizm.quantize`
- `cizm.sparsify`

## Compression Classes

- `QuantizeWeight(min=-128, max=127)`
: レイヤ重みを forward 中のみ量子化します。

- `QuantizeActivation(min=-128, max=127)`
: レイヤ入力（activation）を量子化します。

- `SparseWeightUnstructured(sparsity=0.5)`
: レイヤ重みを forward 中のみ非構造スパース化します。

- `SparseActivationUnstructured(sparsity=0.5)`
: レイヤ入力（activation）を非構造スパース化します。

## Notes

- `apply_layers` を指定しない場合、`Compression.attach` は全サブモジュールに適用を試みます。
- 圧縮クラス側は `Conv2d/Linear` のみ受け付けるため、通常は `apply_layers` の指定を推奨します。

## Build

```bash
python -m pip install -U build twine
python -m build
python -m twine check dist/*
```
