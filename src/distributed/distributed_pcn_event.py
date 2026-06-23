"""
分散PCN - イベント観測の明示エントリポイント (distributed_pcn_event)
"""
import os
import sys

os.environ["DISTRIBUTED_PCN_USE_EVENT_OBS"] = "1"
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_NATIVE", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
os.environ.setdefault("DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP", "1")


def main():
    from src.distributed.distributed_pcn import main as distributed_pcn_main

    print("[PCN_EVENT] イベント観測 + イベントネイティブ step で分散PCNを実行します")
    distributed_pcn_main()


if __name__ == "__main__":
    from src.distributed.distributed_pcn_cli import apply_distributed_pcn_cli

    apply_distributed_pcn_cli()
    main()
