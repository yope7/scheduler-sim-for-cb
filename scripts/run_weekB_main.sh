#!/usr/bin/env bash
# weekB_sub256(一週間分・間引き567ジョブ・容量256・負荷率1.17)の主実験:
# 丸ごと学習 base×3 + ct03×3 (NITER=60, 確立済み512級レシピ, α=n_jobs自動)。
# 学習後に rich_eval を weekB 較正(ベースライン掃引実測: cost max 1.74e8 / wait max 87,525)で自動実行。
set -u
cd /home/noguchi/scheduler-sim-for-cb
NITER="${NITER:-60}"; MAXJOBS="${MAXJOBS:-3}"
MARK=/tmp/weekB_main.marker; rm -f "$MARK"
CFG=experiments/distributed_pcn/job_trace_weekB_sub256_pcn.yml
EVAL_CAL="COST_SCALE=1.74e8 HV_COST_REF=2.0e8 HV_WAIT_REF=1.0e5"
WOFF="PCN_TRAIN_MID_PF_WEIGHT=0 PCN_TRAIN_KNEE_PF_WEIGHT=0 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0 PCN_TRAIN_LOW_WAIT_PF_WEIGHT=0 PCN_TRAIN_COST_ENDPOINT_WEIGHT=0"
DENS="PCN_TRAIN_PF_DENSITY_WEIGHT=8 PCN_TRAIN_PF_DENSITY_K=2 PCN_TRAIN_PF_DENSITY_ALPHA=1.0"
DEFER="SCHEDULER_ALLOW_DEFER=1 SCHEDULER_DEFER_OFFSET=1 DISTRIBUTED_PCN_PHASE1_GIANT_DEFER=0.9"
BASE="PCN_FILM=1 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1 DISTRIBUTED_PCN_N_UPDATES=200 SCHEDULER_OBS_URGENCY=0 DISTRIBUTED_PCN_LIVE_UNIFORM_PF=1 PCN_COST_HOLD=0 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 $WOFF $DENS $DEFER"
CT03="PCN_CMD_TRACK_WEIGHT=0.3 DISTRIBUTED_PCN_CMD_OUTCOMES=1"

train_eval_one(){ local tag="$1" lv="$2" i="$3" t0 t1
  t0=$(date +%s)
  env $lv DISTRIBUTED_PCN_EARLYSTOP=0 DISTRIBUTED_PCN_CONFIG=$CFG \
    DISTRIBUTED_PCN_N_ACTORS=8 \
    bash scripts/run_synthetic_urgency.sh "weekB_${tag}_${i}" 566 "$NITER" > /tmp/weekB_${tag}_${i}.out 2>&1
  t1=$(date +%s); echo "[weekB] train ${tag}_${i} exit=$? sec=$((t1-t0)) $(date +%H:%M:%S)"
  local d=$(ls -dt experiments/distributed_pcn/run_synth566_weekB_${tag}_${i}/2* 2>/dev/null | head -1)
  local ck="$d/final_model.pth"
  [ -f "$ck" ] || ck=$(find "$d"/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
  if [ -f "$ck" ]; then
    env CKPT="$ck" NJ=566 CFG=$CFG JOB_TYPE=2 $EVAL_CAL \
        TRACE_PATH="job_trace/FY2024/scacctreq_202412_weekB_sub256_jobs.csv" \
        PYTHONPATH=. .venv/bin/python scripts/rich_eval_cell.py > /tmp/weekB_${tag}_${i}.json 2>/tmp/weekB_${tag}_${i}.err
    echo "[weekB] eval ${tag}_${i} done $(date +%H:%M:%S)"
  else
    echo "[weekB] eval ${tag}_${i} SKIP (no ckpt)"
  fi
}

JOBS=()
for i in 1 2 3; do JOBS+=("base|$BASE|$i"); done
for i in 1 2 3; do JOBS+=("ct03|$BASE $CT03|$i"); done

echo "[weekB] START $(date +%H:%M:%S) ${#JOBS[@]}runs NITER=$NITER MAXJOBS=$MAXJOBS"
for spec in "${JOBS[@]}"; do
  IFS='|' read -r tag lv i <<< "$spec"
  while [ "$(jobs -rp|wc -l)" -ge "$MAXJOBS" ]; do sleep 10; done
  echo "[weekB] launch ${tag} rep=$i $(date +%H:%M:%S)"
  train_eval_one "$tag" "$lv" "$i" &
done
wait
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "[weekB] ALL DONE $(date +%H:%M:%S)"
# 集計
.venv/bin/python - <<'PY'
import json, numpy as np
def load(p):
    try: return json.loads([l for l in open(p) if l.strip().startswith("{")][-1])
    except Exception: return {}
for tag in ["base","ct03"]:
    hv=[];cd=[];npf=[]
    for i in [1,2,3]:
        d=load(f"/tmp/weekB_{tag}_{i}.json")
        if "hv" in d: hv.append(d["hv"]);cd.append(d["cmd_dist"]);npf.append(d.get("n_pf"))
    if hv:
        hv=np.array(hv);cd=np.array(cd)
        sd=lambda a: np.std(a,ddof=1) if len(a)>1 else 0
        print(f"{tag}: HV {hv.mean():.4f}±{sd(hv):.4f} 追従 {cd.mean():.4f}±{sd(cd):.4f} 崩壊 {(cd>0.15).sum()}/{len(cd)} n_pf {np.mean(npf):.0f}")
    else:
        print(f"{tag}: (no data)")
PY
