# rasnatune

`rasnatune` は、ハードウェア寄りの量子化やスパース化を PyTorch モデルで学習・検証するための軽量ツールキットです。
既存モデルに一時的な forward hook を直接取り付けるため、モデルオブジェクトそのものや `state_dict()` のキーは変わりません。

## インストール

```bash
pip install rasnatune
```

## クイックスタート

```python
import torch
from rasnatune import compress, quantize, remove_compression

model = torch.nn.Sequential(
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 10),
)

compress(
    model,
    quantize(
        activation=8,
        weight=8,
        accumulator=17,
        weight_sparsity=0.9,
        targets=torch.nn.Linear,
    ),
)

x = torch.randn(4, 128)
y = model(x)

torch.save(model.state_dict(), "model.pt")
remove_compression(model)
```

## Recipe

Recipe は「モデルにどんな圧縮挙動を足すか」を表す設定です。
`compress()` は recipe をモデルへ適用し、同じモデルインスタンスを返します。

```python
from rasnatune import compress, quantize, sparse

compress(
    model,
    quantize(
        activation=8,
        weight=8,
        accumulator=17,
        weight_sparsity=0.9,
        targets=(torch.nn.Conv2d, torch.nn.Linear),
    ),
    sparse(0.5, targets=torch.nn.Linear),
)
```

対象 module は recipe ごとに明示します。
`targets=` には module 型、module 型の tuple、または predicate を渡せます。

```python
compress(
    model,
    quantize(activation=8, weight=8, accumulator=17, targets=torch.nn.Linear),
)
```

## 公開 API

トップレベルの recipe API:

- `rasnatune.compress(model, *recipes)`
- `rasnatune.remove_compression(model)`
- `rasnatune.compression_count(model)`
- `rasnatune.quantize(activation=8, weight=8, accumulator=17, targets=..., overflow="wrap", weight_sparsity=0.0)`
- `rasnatune.sparse(sparsity=0.5, targets=...)`

低レベル compressor クラスも、独自ワークフロー向けに引き続き利用できます。

- `rasnatune.Compressor`
- `rasnatune.QuantizeWeight`
- `rasnatune.QuantizeActivation`
- `rasnatune.QuantizeMAC`
- `rasnatune.SparseWeightUnstructured`

## 量子化 MAC シミュレーション

`quantize()` は、実際の PyTorch 実行は FP32 のまま保ちつつ、signed integer MAC ハードウェアに近い挙動をシミュレートします。

- activation と weight は straight-through gradient 付きで量子化されます
- `weight_sparsity` を指定すると、weight をスパース化してから量子化します
- `Linear` / `Conv2d` の演算自体は FP32 で実行されます
- layer の最終出力に signed accumulator 幅での wrap-around を適用します
- 同じ layer 内では出力 activation の再量子化は行いません

例えば `quantize(activation=8, weight=8, accumulator=17, weight_sparsity=0.9, targets=torch.nn.Linear)` は、weight を 90% スパース化し、signed 8-bit の入力・重みを使い、最終的な MAC 出力に signed 17-bit の wrap-around を適用します。
