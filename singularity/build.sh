#!/bin/bash
# Singularity イメージのビルドスクリプト
# プロジェクトルートで実行: ./singularity/build.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

if ! command -v singularity &> /dev/null && ! command -v apptainer &> /dev/null; then
    echo "Error: singularity または apptainer がインストールされていません"
    exit 1
fi

CMD="singularity"
if command -v apptainer &> /dev/null; then
    CMD="apptainer"
fi

OUTPUT="${1:-scheduler-sim.sif}"
echo "Building $OUTPUT ..."
$CMD build "$OUTPUT" singularity/scheduler-sim.def
echo "Done: $OUTPUT"
