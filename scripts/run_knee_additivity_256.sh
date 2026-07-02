#!/usr/bin/env bash
# knee精緻化 + 加法性 (trace256, NITER=60, 正準レシピ=run_trace_final.sh と同一):
#   knee: cmd-track 重み w∈{0.1,0.2,0.4,0.5} × seed{1,2,3} = 12本
#   加法性: ct03(w0.3)+fb2(Fourier帯2) × seed{1,2,3} = 3本
# 学習完了ごとに rich_eval_cell(256較正) を自動実行 → /tmp/knee_{tag}_{i}.json
# GPU pin はしない(CUDA_VISIBLE_DEVICES を渡すと Ray Learner が device 文字列と誤読して即死する既知バグ)。
# MAXJOBS=3 並列(06-25 に MAXJOBS=4 で 256/512 成功実績あり)。
set -u
cd /home/noguchi/scheduler-sim-for-cb
NITER="${NITER:-60}"; MAXJOBS="${MAXJOBS:-3}"
MARK=/tmp/knee256.marker; rm -f "$MARK"
WOFF="PCN_TRAIN_MID_PF_WEIGHT=0 PCN_TRAIN_KNEE_PF_WEIGHT=0 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0 PCN_TRAIN_LOW_WAIT_PF_WEIGHT=0 PCN_TRAIN_COST_ENDPOINT_WEIGHT=0"
DENS="PCN_TRAIN_PF_DENSITY_WEIGHT=8 PCN_TRAIN_PF_DENSITY_K=2 PCN_TRAIN_PF_DENSITY_ALPHA=1.0"
DEFER="SCHEDULER_ALLOW_DEFER=1 SCHEDULER_DEFER_OFFSET=1 DISTRIBUTED_PCN_PHASE1_GIANT_DEFER=0.9"
BASE="PCN_FILM=1 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1 DISTRIBUTED_PCN_N_UPDATES=200 SCHEDULER_OBS_URGENCY=0 DISTRIBUTED_PCN_LIVE_UNIFORM_PF=1 PCN_COST_HOLD=0 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 $WOFF $DENS $DEFER"
CT="DISTRIBUTED_PCN_CMD_OUTCOMES=1"

train_eval_one(){ local tag="$1" lv="$2" i="$3" t0 t1
  t0=$(date +%s)
  env $lv DISTRIBUTED_PCN_EARLYSTOP=0 DISTRIBUTED_PCN_CONFIG=experiments/distributed_pcn/job_trace_256_pcn.yml \
    DISTRIBUTED_PCN_N_ACTORS=8 \
    bash scripts/run_synthetic_urgency.sh "knee_${tag}_${i}" 256 "$NITER" > /tmp/knee_${tag}_${i}.out 2>&1
  t1=$(date +%s); echo "[knee256] train ${tag}_${i} exit=$? sec=$((t1-t0)) $(date +%H:%M:%S)"
  # 学習直後に rich eval を自動チェーン(256デフォルト較正)
  local d=$(ls -dt experiments/distributed_pcn/run_synth256_knee_${tag}_${i}/2* 2>/dev/null | head -1)
  local ck="$d/final_model.pth"
  [ -f "$ck" ] || ck=$(find "$d"/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
  if [ -f "$ck" ]; then
    env CKPT="$ck" NJ=256 CFG=experiments/distributed_pcn/job_trace_256_pcn.yml JOB_TYPE=2 \
        TRACE_PATH="job_trace/FY2024/scacctreq_202412_top1024_jobs.csv" \
        PYTHONPATH=. .venv/bin/python scripts/rich_eval_cell.py > /tmp/knee_${tag}_${i}.json 2>/tmp/knee_${tag}_${i}.err
    echo "[knee256] eval ${tag}_${i} done $(date +%H:%M:%S)"
  else
    echo "[knee256] eval ${tag}_${i} SKIP (no ckpt)"
  fi
}

JOBS=()
for w in 0.1 0.2 0.4 0.5; do
  tag="w${w/./}"   # w01, w02, w04, w05
  for i in 1 2 3; do JOBS+=("${tag}|PCN_CMD_TRACK_WEIGHT=${w} $CT|$i"); done
done
for i in 1 2 3; do JOBS+=("ctfb2|PCN_CMD_TRACK_WEIGHT=0.3 $CT PCN_FOURIER_BANDS=2|$i"); done

echo "[knee256] START $(date +%H:%M:%S) ${#JOBS[@]}runs NITER=$NITER MAXJOBS=$MAXJOBS"
for spec in "${JOBS[@]}"; do
  IFS='|' read -r tag lv i <<< "$spec"
  while [ "$(jobs -rp|wc -l)" -ge "$MAXJOBS" ]; do sleep 10; done
  echo "[knee256] launch ${tag} rep=$i $(date +%H:%M:%S)"
  train_eval_one "$tag" "$BASE $lv" "$i" &
done
wait
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "[knee256] ALL DONE $(date +%H:%M:%S)"
