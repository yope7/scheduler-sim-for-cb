#!/usr/bin/env bash
# 第3段(S4=本番): 本物のフル週への汎化実験。
#   学習: weekA(前週)フルの先頭4096ジョブ窓 × cap22,700/90,800 で base 3seed (評価プール併用)
#   参照フロント: weekB_full で ミニNSGA-II (POP64 GEN30) ※フルNSGA-IIは1評価50sで不可能
#   適用: weekB_full(50,278ジョブ, 完全未知) へ carry ストリーミング (frac掃引)
set -u
cd /home/noguchi/scheduler-sim-for-cb
MARK=/tmp/s4chain.marker; rm -f "$MARK"
CFGW=experiments/distributed_pcn/job_trace_weekB_full_pcn.yml
CFGT=experiments/distributed_pcn/job_trace_weekAfull_win4096_pcn.yml
NJW=50278

echo "[S4] 1) ミニNSGA-II (weekB_full) 起動 $(date +%H:%M:%S)"
CFG=$CFGW NJ=$NJW POP=64 GEN=30 MUT=auto NPROC=24 OUT=/tmp/s4_nsga2.npz \
  PYTHONPATH=. .venv/bin/python scripts/run_nsga2_trace512.py > /tmp/s4_nsga2.log 2>&1 &
NSGA_PID=$!

echo "[S4] 2) 窓4096 学習 base 3seed (2並列) $(date +%H:%M:%S)"
WOFF="PCN_TRAIN_MID_PF_WEIGHT=0 PCN_TRAIN_KNEE_PF_WEIGHT=0 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0 PCN_TRAIN_LOW_WAIT_PF_WEIGHT=0 PCN_TRAIN_COST_ENDPOINT_WEIGHT=0"
DENS="PCN_TRAIN_PF_DENSITY_WEIGHT=8 PCN_TRAIN_PF_DENSITY_K=2 PCN_TRAIN_PF_DENSITY_ALPHA=1.0"
DEFER="SCHEDULER_ALLOW_DEFER=1 SCHEDULER_DEFER_OFFSET=1 DISTRIBUTED_PCN_PHASE1_GIANT_DEFER=0.9"
BASE="PCN_FILM=1 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1 DISTRIBUTED_PCN_N_UPDATES=200 SCHEDULER_OBS_URGENCY=0 DISTRIBUTED_PCN_LIVE_UNIFORM_PF=1 PCN_COST_HOLD=0 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 $WOFF $DENS $DEFER PCN_EVAL_ACTOR_POOL=16"
for i in 1 2 3; do
  while [ "$(jobs -rp|wc -l)" -ge 3 ]; do sleep 15; done   # NSGA(1)+学習2並列
  env $BASE DISTRIBUTED_PCN_EARLYSTOP=0 DISTRIBUTED_PCN_CONFIG=$CFGT \
      DISTRIBUTED_PCN_N_ACTORS=8 \
      bash scripts/run_synthetic_urgency.sh s4win_base_${i} 4096 60 > /tmp/s4win_base_${i}.out 2>&1 &
done
wait
echo "[S4] 学習+NSGA-II 完了 $(date +%H:%M:%S)"
tail -3 /tmp/s4_nsga2.log

echo "[S4] 3) carry 適用 (フル週50,278) $(date +%H:%M:%S)"
for i in 1 2 3; do
  D=$(ls -dt experiments/distributed_pcn/run_synth4096_s4win_base_${i}/2* 2>/dev/null | head -1)
  CK="$D/final_model.pth"; [ -f "$CK" ] || CK=$(find "$D"/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
  LOG="experiments/distributed_pcn/run_synth4096_s4win_base_${i}/train.log"
  CCL=$(grep -a "WORKLOAD_CALIB" "$LOG" | head -1 | grep -oE "cost\(op/cl\)=[0-9]+/[0-9]+" | sed 's/.*\///')
  WPAIR=$(grep -a "WORKLOAD_CALIB" "$LOG" | head -1 | grep -oE "wait\(op/cl\)=[0-9.]+/[0-9.]+" | sed 's/.*)=//')
  WOP=${WPAIR%/*}; WCL=${WPAIR#*/}
  echo "[S4] seed$i 較正: cost_cl=$CCL wait_op=$WOP wait_cl=$WCL"
  if [ -f "$CK" ] && [ -n "$CCL" ]; then
    CKPT=$CK CFG=$CFGW NJ=$NJW WINDOW=4096 MODE=carry \
      WIN_COST_CLOUD=$CCL WIN_WAIT_ONPREM=$WOP WIN_WAIT_CLOUD=$WCL \
      FRACS=0.0,0.15,0.3,0.5,0.7,0.85,1.0 OUT=/tmp/s4_carry_${i}.json \
      PYTHONPATH=. .venv/bin/python scripts/eval_weekly_rolling.py > /tmp/s4_carry_${i}.log 2>&1 && echo "  seed$i OK" || tail -2 /tmp/s4_carry_${i}.log
  else
    echo "  seed$i SKIP (ckpt/cal 欠落)"
  fi
done

echo "[S4] 4) 集計 $(date +%H:%M:%S)"
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
# 基準点 = フル週P掃引実測の最大×1.15
REF=np.array([1.9582e10*1.15, 102160.6*1.15])
# 参照フロント = ミニNSGA-II PF ∪ P掃引点 の非支配
psweep=np.array([[0.0,102160.6],[4.6664e9,17065.7],[9.5261e9,1623.7],[1.4656e10,25.8],[1.9582e10,57.2]])
try:
    ns=np.load("/tmp/s4_nsga2.npz"); pool=np.vstack([ns["pf"].astype(float), psweep])
except Exception:
    pool=psweep
hv_ref=hv2d(pool,REF)
print(f"参照フロントHV(NSGA∪P掃引) 基準点=[{REF[0]:.2e},{REF[1]:.0f}]")
rows=[]
for i in [1,2,3]:
    try:
        d=json.load(open(f"/tmp/s4_carry_{i}.json"))
        pts=np.array([[r["cost"],r["avg_wait"]] for r in d["results"]],dtype=float)
        fr=[r["frac"] for r in d["results"]]
        errs=[abs(r["cost"]-r["week_cost_target"])/r["week_cost_target"] for r in d["results"] if r["week_cost_target"]>1e8]
        rho=np.corrcoef(np.argsort(np.argsort(fr)),np.argsort(np.argsort(pts[:,0])))[0,1]
        hvr=hv2d(pts,REF)/hv_ref
        rows.append([hvr,rho,np.mean(errs) if errs else np.nan])
        print(f"seed{i}: HV比 {hvr*100:5.1f}%  単調性ρ {rho:5.2f}  週目標誤差 {np.mean(errs)*100 if errs else -1:4.0f}%  窓数{d['results'][0]['n_windows']}")
    except Exception as e:
        print(f"seed{i}: FAIL {e}")
if rows:
    a=np.array(rows); sd=lambda x: x.std(ddof=1) if len(x)>1 else 0
    print(f"S4まとめ(フル週50,278ジョブ): HV比 {a[:,0].mean()*100:.1f}±{sd(a[:,0])*100:.1f}%  ρ {np.nanmean(a[:,1]):.2f}  誤差 {np.nanmean(a[:,2])*100:.0f}%")
PY
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "[S4] ALL DONE $(date +%H:%M:%S)"
