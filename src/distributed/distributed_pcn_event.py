"""
分散PCN - イベント観測の明示エントリポイント (distributed_pcn_event)

本体 (src.distributed.distributed_pcn) は既定でイベント観測 + ラーナー側ビットマップ復元（NN入力）を使う。
このモジュールは互換用にイベント観測を import 前に強制する。

実行:
  uv run python -m src.distributed.distributed_pcn
  uv run python -m src.distributed.distributed_pcn_event  # 同等（USE_EVENT_OBS 強制）
"""
import os

os.environ["DISTRIBUTED_PCN_USE_EVENT_OBS"] = "1"

from src.distributed.distributed_pcn import main

if __name__ == "__main__":
    print("[PCN_EVENT] イベントベース観測で分散PCNを実行します")
    main()
