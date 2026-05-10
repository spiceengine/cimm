# AGENTS.md

## プロジェクト概要

- `rasnatune` は forward hook を使った軽量な PyTorch 圧縮ヘルパーです。
- モデルの重みを恒久的に書き換えるのではなく、実行時だけ一時的な変換を差し込みます。
- 現在のパッケージ版は [`pyproject.toml`](pyproject.toml) の `0.0.2` です。
- コードベースは小さく保たれています。recipe API、抽象基底、量子化、スパース化が中心です。

## リポジトリ構成

- [`rasnatune/__init__.py`](rasnatune/__init__.py): トップレベルの公開 API。
- [`rasnatune/base.py`](rasnatune/base.py): `Compressor` 基底クラスと hook 解除ヘルパー。
- [`rasnatune/compression.py`](rasnatune/compression.py): in-place な `compress()` API、recipe ヘルパー、モデル単位の登録管理、cleanup。
- [`rasnatune/quantization.py`](rasnatune/quantization.py): 量子化ユーティリティ、weight / activation / MAC 量子化 hook。
- [`rasnatune/sparsification.py`](rasnatune/sparsification.py): スパース化ユーティリティ、weight / activation スパース化 hook。
- [`tests/test_quantization.py`](tests/test_quantization.py): 量子化まわりのテスト。
- [`tests/test_sparsification.py`](tests/test_sparsification.py): スパース化まわりのテスト。
- [`tests/test_compression.py`](tests/test_compression.py): recipe API と統合挙動のテスト。
- [`README.md`](README.md): インストール方法と API 概要。
- [`Makefile`](Makefile): ローカル環境作成とテストのショートカット。
- [`.github/workflows/rasnatune-pypi-publish.yml`](.github/workflows/rasnatune-pypi-publish.yml): `v*` タグで distribution をビルドし PyPI へ publish します。

## パッケージングと依存関係

- パッケージメタデータは [`pyproject.toml`](pyproject.toml) にあります。
- runtime dependency は `torch>=2.0` のみです。
- test dependency group はまだありません。
- ビルド済み artifact はリポジトリに保持しない方針です。publish workflow が `dist/` を作り直します。

## コア設計

### `Compressor`

- [`rasnatune/base.py`](rasnatune/base.py) の `Compressor` が最小の抽象です。
- サブクラスは `attach(module)` を実装します。
- hook handle は `self._hooks` に保存します。
- `detach()` は hook handle を remove し、`self._hooks` を空にします。

### In-Place Compression API

- [`rasnatune/compression.py`](rasnatune/compression.py) の `compress(model, *recipes)` は、ユーザーのモデルへ直接 hook を取り付け、同じ model インスタンスを返します。
- `remove_compression(model)` は、そのモデルに登録された compressor を解除します。
- `compression_count(model)` は active な compressor 登録数を返します。
- recipe API には `quantize()`, `sparse()` があります。
- `quantize(weight_sparsity=...)` は weight をスパース化してから量子化するため、weight を書き換える recipe を別々に重ねずにスパース化 + MAC 量子化を表現できます。
- recipe の `targets=` は必須です。
- `targets=` には module 型、module 型の tuple、または predicate を渡せます。
- `compress()` は wrapper module を作らないため、`state_dict()` のキーは元モデルのまま維持されます。

### 量子化パス

- 内部の tensor 量子化処理は最大絶対値から scale を計算し、指定整数範囲へ丸め、straight-through estimator 形式の tensor を返します。
- `quantize_accumulator(x, bits, scale)` は signed accumulator 幅で wrap-around し、straight-through estimator 形式で返します。
- `QuantizeWeight` は forward pre-hook で `module.weight.data` を一時的に量子化し、forward hook で元の重みへ戻します。
- `QuantizeActivation` は forward pre-hook で最初の positional input だけを書き換えます。
- `QuantizeMAC` は最初の positional input と weight を量子化し、module 演算は FP32 で実行し、最終出力へ signed accumulator wrap-around を適用します。`weight_sparsity > 0` の場合は、weight をスパース化してから量子化します。

### スパース化パス

- `sparsify(x, sparsity)` は絶対値の小さい要素を指定割合だけ 0 にします。
- `SparseWeightUnstructured` は `QuantizeWeight` と同じ backup / mutate / restore hook パターンです。
- activation スパース化 API は削除済みです。

## Makefile とローカルワークフロー

現在の [`Makefile`](Makefile) には 2 つの target があります。

- `make activate`
  - `python3 -m venv` で `./.venv` を作ります。
  - `torch` と `torchvision` をインストールします。

- `make test`
  - `./.venv/bin/pytest tests/` を実行します。

注意点:

- `make activate` は `pytest` をインストールしません。そのため、`pytest` を手動で `./.venv` に入れていない場合、`make test` は失敗します。

## 現在のテスト範囲

自動テストは現在以下をカバーしています。

- 内部 tensor 量子化処理の symmetric / asymmetric range に対する数値挙動。
- 内部 tensor 量子化処理の straight-through gradient。
- 内部 tensor 量子化処理の zero tensor 安定性。
- `quantize_accumulator()` の wrap-around と straight-through gradient。
- `QuantizeWeight` の forward 挙動と通常実行後の weight 復元。
- `QuantizeWeight.detach()` の挙動。
- `QuantizeActivation` が最初の positional input を書き換えること、および detach 挙動。
- `QuantizeMAC` の `Linear` / `Conv2d` に対する挙動。
- `compress()` の in-place 挙動、recipe targeting、cleanup、`state_dict()` key 維持。
- スパース化ユーティリティと hook 挙動。

## 既知の注意点

- activation 系 compressor は最初の positional input だけを書き換えます。
- これは現在の意図した実装ですが、複数 tensor 引数や keyword-only tensor を受け取る module では見落としやすいです。

- 圧縮は runtime hook によって適用されます。
- `model.state_dict()` を保存する場合、元モデルの key は維持されます。
- 圧縮が active な model object 全体を pickle すると、runtime hook も object の一部になります。基本的には `state_dict()` 保存を使うか、full-object serialization の前に `remove_compression(model)` を呼ぶ方針を推奨します。

## 今後の変更時の実務メモ

- hook 挙動を変更する場合は、通常 forward、detach / cleanup、forward 中の例外をテストしてください。
- `compress()` や recipe を触る場合は、登録数、`targets=` による module 選択、`state_dict()` key 維持、cleanup の直接テストを追加してください。
- compressor の適用対象や意味を変える場合は、[`README.md`](README.md) と関連テストを同時に更新してください。
- `make test` を維持するなら、環境作成が必要なテストツールをインストールするか、optional dependency group へ移すかを検討してください。
- スパース化を再設計する場合は、`sparsity=0.0`, `sparsity=1.0`, quantile 付近で値が重複する場合の意味を先に固定してください。

## リリースフロー

- パッケージングは `setuptools.build_meta` を使います。
- `v*` に一致するタグが push されると GitHub Actions が PyPI へ publish します。
- workflow は `python -m build` で distribution を作り、`pypa/gh-action-pypi-publish` で publish します。

## 推奨される次の作業

- `Makefile` が test dependency を入れるべきか、あるいは documented optional dependency group に移すべきかを決める。
- README のサンプルを実行可能なテストとして追加する。
- `QuantizeMAC` に `overflow="saturate"` を追加する。
