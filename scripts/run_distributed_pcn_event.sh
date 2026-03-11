#!/bin/bash
# イベントベース観測で分散PCNを実行 (distributed_pcn_event)
# ビットマップを撤廃し、開始/終了/継続時間のみで学習
exec python -m src.distributed.distributed_pcn_event "$@"
