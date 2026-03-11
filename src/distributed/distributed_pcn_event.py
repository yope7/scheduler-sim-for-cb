"""
分散PCN - イベントベース観測版 (distributed_pcn_event)

ビットマップを撤廃し、イベントの開始/終了/継続時間のみで学習する。
distributed_pcn をベースに、環境を SchedulingEnvEventObs に固定した版。

実行:
  uv run python -m src.distributed.distributed_pcn_event
  DISTRIBUTED_PCN_QUICK=1 uv run python -m src.distributed.distributed_pcn_event  # 短時間テスト
"""
import os

# イベント観測を強制（distributed_pcn の import 前に設定必須）
os.environ["DISTRIBUTED_PCN_USE_EVENT_OBS"] = "1"

from src.distributed.distributed_pcn import main

if __name__ == "__main__":
    print("[PCN_EVENT] イベントベース観測で分散PCNを実行します")
    main()
