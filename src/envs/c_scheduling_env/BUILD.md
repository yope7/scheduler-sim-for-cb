# ビルド手順

## 必要な依存関係

- Python 3.7以上
- C/C++コンパイラ（gcc/clang）
- pybind11
- NumPy

## インストール

```bash
# プロジェクトのルートディレクトリから（推奨）
uv sync
```

個別にビルドする場合：

```bash
cd src/envs/c_scheduling_env
uv pip install -e .
```

## デバッグモードでビルド

```bash
cd src/envs/c_scheduling_env
DEBUG=true uv pip install -e .
```

## テスト

```bash
# 単体テスト
python test_c_implementation.py

# ベンチマーク比較
python benchmark_comparison.py
```

## トラブルシューティング

### コンパイルエラー

- **エラー**: `pybind11/pybind11.h: No such file or directory`
  - **解決**: `uv sync` を実行

- **エラー**: `undefined reference to 'build_cache'`
  - **解決**: C言語ファイル（.c）がコンパイルされているか確認

### インポートエラー

- **エラー**: `ModuleNotFoundError: No module named 'scheduling_env_core'`
  - **解決**: `uv sync` を実行してビルド

### パフォーマンスが期待通りでない場合

- コンパイラ最適化フラグを確認（`-O3`, `-march=native`）
- デバッグモードでビルドしていないか確認

