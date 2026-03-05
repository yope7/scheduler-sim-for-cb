# Scheduler Sim for Cloud Bursting

## セットアップ

[uv](https://docs.astral.sh/uv/) を使用したセットアップ（推奨）:

```bash
# uvのインストール（未導入の場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係とC拡張のビルド・インストール
uv sync
```

これにより以下が行われます:
- 仮想環境（.venv）の作成
- 全Python依存関係のインストール
- scheduling_env_core（C拡張）のビルド
- nsga2_core（C拡張）のビルド

## 実行

```bash
uv run python scripts/main.py
# または
uv run python -m pytest
```

## 従来の pip からの移行

- `pip install -r requirement.txt` → `uv sync`
- `pip install -e .`（C拡張）→ `uv sync` に含まれる
- `requirement.txt` は `pyproject.toml` に統合済み（参照用に残置）
