#!/bin/bash
# イベントベース観測で分散PCNを実行 (distributed_pcn_event)
# 既定: イベント観測のまま NN へ（ビットマップ復元OFF）
export SCHEDULER_LEARNER_BITMAP="${SCHEDULER_LEARNER_BITMAP:-0}"
export DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP="${DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP:-1}"
exec python -m src.distributed.distributed_pcn_event "$@"
