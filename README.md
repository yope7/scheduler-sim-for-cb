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

## 動作確認

```bash
uv run python scripts/verify.py
```

C拡張と環境が正常に動作するか数秒で確認できます。

## 実行

```bash
uv run python scripts/main.py
# または
uv run python -m pytest
```

## イベントベース観測（ビットマップ撤廃）

ビットマップ（ウィンドウ占有状態）を観測から撤廃し、イベントの開始/終了/継続時間のみで学習するモード:

```bash
# 検証
uv run python scripts/verify_event_env.py

# 分散PCNをイベント観測で実行 (distributed_pcn_event)
uv run python -m src.distributed.distributed_pcn_event
# 短時間テスト
DISTRIBUTED_PCN_QUICK=1 uv run python -m src.distributed.distributed_pcn_event
# または
uv run bash scripts/run_distributed_pcn_event.sh
```

- **SchedulingEnvEventObs**: `src/envs/scheduling_env_event_obs.py`
- **バックアップ**: `src/envs/backup_c_current/README.md`（現在のC環境の復元方法）

## 従来の pip からの移行

- `pip install -r requirement.txt` → `uv sync`
- `pip install -e .`（C拡張）→ `uv sync` に含まれる
- `requirement.txt` は `pyproject.toml` に統合済み（参照用に残置）
