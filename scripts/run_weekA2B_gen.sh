#!/usr/bin/env bash
# 第1段: カンニングなしの汎化実験。
# 前週(weekA)の窓256で base を5seed学習 → 評価週(weekB_sub256, 完全未知567ジョブ)へ
# carry(繰越)ストリーミング適用 → frac掃引で命令追従と週目標誤差を集計。
set -u
cd /home/noguchi/scheduler-sim-for-cb
MARK=/tmp/weekA2B.marker; rm -f "$MARK"
WOFF="PCN_TRAIN_MID_PF_WEIGHT=0 PCN_TRAIN_KNEE_PF_WEIGHT=0 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0 PCN_TRAIN_LOW_WAIT_PF_WEIGHT=0 PCN_TRAIN_COST_ENDPOINT_WEIGHT=0"
DENS="PCN_TRAIN_PF_DENSITY_WEIGHT=8 PCN_TRAIN_PF_DENSITY_K=2 PCN_TRAIN_PF_DENSITY_ALPHA=1.0"
DEFER="SCHEDULER_ALLOW_DEFER=1 SCHEDULER_DEFER_OFFSET=1 DISTRIBUTED_PCN_PHASE1_GIANT_DEFER=0.9"
BASE="PCN_FILM=1 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1 DISTRIBUTED_PCN_N_UPDATES=200 SCHEDULER_OBS_URGENCY=0 DISTRIBUTED_PCN_LIVE_UNIFORM_PF=1 PCN_COST_HOLD=0 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 $WOFF $DENS $DEFER"

# 5seed 学習 (2並列)
for i in 1 2 3 4 5; do
  while [ "$(jobs -rp|wc -l)" -ge 2 ]; do sleep 10; done
  env $BASE DISTRIBUTED_PCN_EARLYSTOP=0 DISTRIBUTED_PCN_CONFIG=experiments/distributed_pcn/job_trace_weekA_win256_pcn.yml \
      DISTRIBUTED_PCN_N_ACTORS=8 \
      bash scripts/run_synthetic_urgency.sh weekAwin_base_${i} 256 60 > /tmp/weekAwin_base_${i}.out 2>&1 &
done
wait
echo "[weekA2B] 学習5seed完了 $(date +%H:%M:%S)"

# 各seedの較正値を学習ログから取り、weekB(未知)へ carry 適用
for i in 1 2 3 4 5; do
  D=$(ls -dt experiments/distributed_pcn/run_synth256_weekAwin_base_${i}/2* 2>/dev/null | head -1)
  CK="$D/final_model.pth"; [ -f "$CK" ] || CK=$(find "$D"/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
  LOG="experiments/distributed_pcn/run_synth256_weekAwin_base_${i}/train.log"
  CAL=$(grep -a "WORKLOAD_CALIB" "$LOG" | head -1)
  CCL=$(echo "$CAL" | grep -oE "cost\(op/cl\)=[0-9]+/[0-9]+" | cut -d/ -f3)
  WOP=$(echo "$CAL" | grep -oE "wait\(op/cl\)=[0-9.]+/[0-9.]+" | sed 's/wait(op\/cl)=//' | cut -d/ -f1)
  WCL=$(echo "$CAL" | grep -oE "wait\(op/cl\)=[0-9.]+/[0-9.]+" | cut -d/ -f2)
  echo "[weekA2B] seed$i 較正: cost_cl=$CCL wait_op=$WOP wait_cl=$WCL"
  CKPT=$CK CFG=experiments/distributed_pcn/job_trace_weekB_sub256_pcn.yml NJ=566 WINDOW=256 MODE=carry \
    WIN_COST_CLOUD=$CCL WIN_WAIT_ONPREM=$WOP WIN_WAIT_CLOUD=$WCL \
    FRACS=0.0,0.15,0.3,0.5,0.7,0.85,1.0 OUT=/tmp/weekA2B_carry_${i}.json \
    PYTHONPATH=. .venv/bin/python scripts/eval_weekly_rolling.py > /tmp/weekA2B_carry_${i}.log 2>&1
  echo "[weekA2B] seed$i 適用完了"
done

# 集計
.venv/bin/python - <<'PY'
import json, numpy as np
def hv2d(pf, ref):
    pts=pf[np.lexsort((pf[:,1],pf[:,0]))]
    nd=[]; best=np.inf
    for c,w in pts:
        if w<best: nd.append((c,w)); best=w
    nd=np.array(nd); hv=0.0; prev_c=ref[0]
    for c,w in nd[::-1]:
        if c>=ref[0] or w>=ref[1]: continue
        hv += (prev_c-c)*(ref[1]-w); prev_c=c
    return hv
REF=np.array([2.0e8,1.0e5])
ns=np.load("/tmp/weekB_nsga2.npz"); hv_true=hv2d(ns["pf"].astype(float),REF)
rows=[]
for i in [1,2,3,4,5]:
    try:
        d=json.load(open(f"/tmp/weekA2B_carry_{i}.json"))
        pts=np.array([[r["cost"],r["avg_wait"]] for r in d["results"]],dtype=float)
        fr=[r["frac"] for r in d["results"]]
        errs=[abs(r["cost"]-r["week_cost_target"])/max(1e6,r["week_cost_target"]) for r in d["results"] if r["week_cost_target"]>1e6]
        rho=np.corrcoef(np.argsort(np.argsort(fr)),np.argsort(np.argsort(pts[:,0])))[0,1]
        hvr=hv2d(pts,REF)/hv_true
        rows.append((i,hvr,rho,np.mean(errs)))
        print(f"seed{i}: HV比 {hvr*100:.1f}%  単調性ρ {rho:.2f}  週目標誤差 {np.mean(errs)*100:.0f}%")
    except Exception as e:
        print(f"seed{i}: FAIL {e}")
if rows:
    a=np.array([[r[1],r[2],r[3]] for r in rows])
    print(f"\n5seedまとめ: HV比 {a[:,0].mean()*100:.1f}±{a[:,0].std(ddof=1)*100:.1f}%  ρ {a[:,1].mean():.2f}±{a[:,1].std(ddof=1):.2f}  誤差 {a[:,2].mean()*100:.0f}%")
PY
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "[weekA2B] ALL DONE $(date +%H:%M:%S)"
