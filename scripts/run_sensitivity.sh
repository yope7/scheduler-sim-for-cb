#!/usr/bin/env bash
# 学習済み方策の入力感度分析(推論のみ): 1モデルを、パラメータを少しずつ変えた合成ジョブで推論し
# 追従corr/spanがどれだけ劣化するかを測る。学習と同じデータ(baseline)からの逸脱=感度。
# eval_b2_compare(greedyのみ KSAMP=1)を各条件で回し results/eval_pf/sensitivity/ に保存。
set -u
cd /home/noguchi/scheduler-sim-for-cb
mkdir -p results/eval_pf/sensitivity
RECIPE="PCN_FILM=1 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4"
NPROC="${NPROC:-24}"
SYNTHCFG=experiments/distributed_pcn/job_synthetic_pcn.yml
TRACECFG=experiments/distributed_pcn/job_trace_256_pcn.yml
# ベースモデル
SYNTH_EXEC=$(ls -d experiments/distributed_pcn/run_synth256_tv05_4/*/ | head -1)
SYNTH_CK=$(find "${SYNTH_EXEC}iteration_"* -name 'model_iter_*.pth' | sort -V | tail -1)
TRACE_EXEC=$(ls -d experiments/distributed_pcn/run_trace256_fourier_lc/*/ | head -1)
TRACE_CK=$(find "${TRACE_EXEC}iteration_"* -name 'model_iter_*.pth' | sort -V | tail -1)
echo "[sens] synth model=$SYNTH_CK"
echo "[sens] trace model=$TRACE_CK"

ev(){ # tag  extra-env  cfg  nj  seeds
  local tag="$1" extra="$2" cfg="$3" nj="$4" seeds="$5"
  env $RECIPE $extra SCHEDULER_OBS_URGENCY=1 CKPT="$6" CFG="$cfg" NJ="$nj" SEEDS="$seeds" NCMD=40 KSAMP=1 NPROC="$NPROC" \
    OUT="results/eval_pf/sensitivity/${tag}.npz" PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py > /tmp/sens_${tag}.out 2>&1
  echo "[sens] $tag exit=$? $(date +%H:%M:%S)"
}

echo "=== A) synthモデル(tv05_4, 学習=L0.5/256/seed0) の感度 ==="
# A1 裾level 微細スイープ (base=0.5)
for L in 0.0 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0; do
  ev "syn_tail_${L}" "SYNTH_TAIL_LEVEL=$L" "$SYNTHCFG" 256 0 "$SYNTH_CK"
done
# A2 ジョブサイズ max_nodes (base=256)
for MN in 3 8 32 128 256; do
  ev "syn_nodes_${MN}" "SYNTH_TAIL_LEVEL=0.5 SYNTH_MAX_NODES=$MN" "$SYNTHCFG" 256 0 "$SYNTH_CK"
done
# A3 未知seed (同一params, 別インスタンス) 一括
ev "syn_seeds" "SYNTH_TAIL_LEVEL=0.5" "$SYNTHCFG" 256 "0,1,2,3,4" "$SYNTH_CK"
# A4 スケール n_jobs (base=256)
for NJ in 128 192 256 384 512; do
  ev "syn_scale_${NJ}" "SYNTH_TAIL_LEVEL=0.5" "$SYNTHCFG" "$NJ" 0 "$SYNTH_CK"
done

echo "=== B) traceモデル(base1, 学習=生trace) の感度 ==="
# B1 生trace(同じデータ) + 構造破壊3種 + 未知seed
ev "tr_raw"     ""                              "$TRACECFG" 256 0 "$TRACE_CK"
ev "tr_shuffle" "SCHEDULER_TRACE_SHUFFLE_PAYLOAD=1" "$TRACECFG" 256 0 "$TRACE_CK"
ev "tr_spread"  "SCHEDULER_TRACE_ARRIVAL_SPREAD=1"  "$TRACECFG" 256 0 "$TRACE_CK"
ev "tr_jitter"  "SCHEDULER_TRACE_JITTER_PT=0.15"    "$TRACECFG" 256 0 "$TRACE_CK"
# B2 traceモデルを synth に当てる(完全OOD)
ev "tr_on_synth" "SYNTH_TAIL_LEVEL=0.5" "$SYNTHCFG" 256 0 "$TRACE_CK"
echo "DONE $(date +%H:%M:%S)" > /tmp/sensitivity.marker
echo "[sens] ALL DONE $(date +%H:%M:%S)"
