#!/bin/bash
# distributed_pcn.py または指定スクリプトを Singularity で実行するラッパースクリプト
# 使用例:
#   ./singularity/run.sh                    # デフォルト: distributed_pcn
#   ./singularity/run.sh --quick             # distributed_pcn に引数
#   SCHEDULER_SIM_SCRIPT=scripts/run_distributed_pcn_sweep.py ./singularity/run.sh --quick

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE="${SCHEDULER_SIM_IMAGE:-$PROJECT_ROOT/scheduler-sim.sif}"

if [ ! -f "$IMAGE" ]; then
    echo "Error: イメージが見つかりません: $IMAGE"
    echo "先に ./singularity/build.sh でビルドするか、SCHEDULER_SIM_IMAGE でパスを指定してください"
    exit 1
fi

CMD="singularity"
if command -v apptainer &> /dev/null; then
    CMD="apptainer"
fi

# 出力先: デフォルトはプロジェクトルート（-B でマウントしてホストに保存）
OUTPUT_DIR="${DISTRIBUTED_PCN_OUTPUT_DIR:-$PROJECT_ROOT}"
BIND_ARGS="-B $OUTPUT_DIR:/output"

# スクリプト実行時: プロジェクトルートもマウント（sweep 等が execution_times_*.json を /app に書き込むため）
if [ -n "$SCHEDULER_SIM_SCRIPT" ]; then
    BIND_ARGS="$BIND_ARGS -B $PROJECT_ROOT:/app"
fi

# オプション: 設定ファイルをホストから差し替える場合
# DISTRIBUTED_PCN_CONFIG にホストの絶対パスを指定すると、そのファイルをマウント
if [ -n "$DISTRIBUTED_PCN_CONFIG" ] && [ -f "$DISTRIBUTED_PCN_CONFIG" ]; then
    BIND_ARGS="$BIND_ARGS -B $DISTRIBUTED_PCN_CONFIG:/app/config/config.yml:ro"
fi

# オプション: job_trace をホストからマウント（job_type=2 の場合）
if [ -n "$JOB_TRACE_DIR" ] && [ -d "$JOB_TRACE_DIR" ]; then
    BIND_ARGS="$BIND_ARGS -B $JOB_TRACE_DIR:/app/job_trace:ro"
fi

export DISTRIBUTED_PCN_OUTPUT_DIR="/output"
export DISTRIBUTED_PCN_WORKDIR="/app"
# コンテナ内では常に /app/config/config.yml（バインドで差し替え可能）
export DISTRIBUTED_PCN_CONFIG="/app/config/config.yml"

# 実行スクリプト: SCHEDULER_SIM_SCRIPT で指定（例: scripts/run_distributed_pcn_sweep.py）
# 指定時は /app/ をプレフィックスしてコンテナ内パスに変換
if [ -n "$SCHEDULER_SIM_SCRIPT" ]; then
    if [[ "$SCHEDULER_SIM_SCRIPT" != /* ]]; then
        export SCHEDULER_SIM_SCRIPT="/app/$SCHEDULER_SIM_SCRIPT"
    fi
fi

exec $CMD run $BIND_ARGS "$IMAGE" "$@"
