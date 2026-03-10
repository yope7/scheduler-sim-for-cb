#!/bin/bash
#SBATCH --job-name=scheduler-sweep
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# 必要に応じて GPU を有効化:
# #SBATCH --gres=gpu:1

# ジョブ投入時のカレントディレクトリをプロジェクトルートとする
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export SCHEDULER_SIM_SCRIPT=scripts/run_distributed_pcn_sweep.py
export DISTRIBUTED_PCN_OUTPUT_DIR="${DISTRIBUTED_PCN_OUTPUT_DIR:-$PROJECT_ROOT}"

# --quick を外すと本番実行
exec "$SCRIPT_DIR/run.sh" -o "results_$(date +%Y%m%d_%H%M%S).json"
# テスト用（短時間）:
# exec "$SCRIPT_DIR/run.sh" --quick
