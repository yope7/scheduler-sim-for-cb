#!/usr/bin/env bash
# 第2段(S2): 4倍規模での週分離汎化実験の全チェーン。
#   データ: weekA_sub1024(学習週2902) / weekB_sub1024(評価週2267) cap1024/4096
#   1) weekB の P掃引(達成レンジ+HV基準点の決定)
#   2) weekB の NSGA-II 真PF
#   3) weekA 窓256 で base 3seed 学習
#   4) weekB へ carry ストリーミング適用 (frac掃引)
#   5) 集計(HV比・単調性・週目標誤差) + ntfy
set -u
cd /home/noguchi/scheduler-sim-for-cb
MARK=/tmp/s2chain.marker; rm -f "$MARK"
CFGW=experiments/distributed_pcn/job_trace_weekB_sub1024_pcn.yml
CFGT=experiments/distributed_pcn/job_trace_weekA1024_win256_pcn.yml
NJW=2267

echo "[S2] 1) P掃引 $(date +%H:%M:%S)"
CFG=$CFGW NJ=$NJW PYTHONPATH=. .venv/bin/python /tmp/claude-1002/-home-noguchi-scheduler-sim-for-cb/8b4624a8-8eec-4cdc-a9e1-691d8c808634/scratchpad/weekly_baseline.py > /tmp/s2_baseline.txt 2>/dev/null
cat /tmp/s2_baseline.txt

echo "[S2] 2) NSGA-II $(date +%H:%M:%S)"
CFG=$CFGW NJ=$NJW POP=200 GEN=150 MUT=auto NPROC=24 OUT=/tmp/s2_nsga2.npz \
  PYTHONPATH=. .venv/bin/python scripts/run_nsga2_trace512.py > /tmp/s2_nsga2.log 2>&1
tail -3 /tmp/s2_nsga2.log

echo "[S2] 3) 窓学習 base 3seed $(date +%H:%M:%S)"
WOFF="PCN_TRAIN_MID_PF_WEIGHT=0 PCN_TRAIN_KNEE_PF_WEIGHT=0 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0 PCN_TRAIN_LOW_WAIT_PF_WEIGHT=0 PCN_TRAIN_COST_ENDPOINT_WEIGHT=0"
DENS="PCN_TRAIN_PF_DENSITY_WEIGHT=8 PCN_TRAIN_PF_DENSITY_K=2 PCN_TRAIN_PF_DENSITY_ALPHA=1.0"
DEFER="SCHEDULER_ALLOW_DEFER=1 SCHEDULER_DEFER_OFFSET=1 DISTRIBUTED_PCN_PHASE1_GIANT_DEFER=0.9"
BASE="PCN_FILM=1 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1 DISTRIBUTED_PCN_N_UPDATES=200 SCHEDULER_OBS_URGENCY=0 DISTRIBUTED_PCN_LIVE_UNIFORM_PF=1 PCN_COST_HOLD=0 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 $WOFF $DENS $DEFER"
for i in 1 2 3; do
  while [ "$(jobs -rp|wc -l)" -ge 2 ]; do sleep 10; done
  env $BASE DISTRIBUTED_PCN_EARLYSTOP=0 DISTRIBUTED_PCN_CONFIG=$CFGT \
      DISTRIBUTED_PCN_N_ACTORS=8 \
      bash scripts/run_synthetic_urgency.sh s2win_base_${i} 256 60 > /tmp/s2win_base_${i}.out 2>&1 &
done
wait
echo "[S2] 学習完了 $(date +%H:%M:%S)"

echo "[S2] 4) carry 適用 $(date +%H:%M:%S)"
for i in 1 2 3; do
  D=$(ls -dt experiments/distributed_pcn/run_synth256_s2win_base_${i}/2* 2>/dev/null | head -1)
  CK="$D/final_model.pth"; [ -f "$CK" ] || CK=$(find "$D"/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
  LOG="experiments/distributed_pcn/run_synth256_s2win_base_${i}/train.log"
  CCL=$(grep -a "WORKLOAD_CALIB" "$LOG" | head -1 | grep -oE "cost\(op/cl\)=[0-9]+/[0-9]+" | sed 's/.*\///')
  WPAIR=$(grep -a "WORKLOAD_CALIB" "$LOG" | head -1 | grep -oE "wait\(op/cl\)=[0-9.]+/[0-9.]+" | sed 's/.*)=//')
  WOP=${WPAIR%/*}; WCL=${WPAIR#*/}
  echo "[S2] seed$i 較正: cost_cl=$CCL wait_op=$WOP wait_cl=$WCL"
  CKPT=$CK CFG=$CFGW NJ=$NJW WINDOW=256 MODE=carry \
    WIN_COST_CLOUD=$CCL WIN_WAIT_ONPREM=$WOP WIN_WAIT_CLOUD=$WCL \
    FRACS=0.0,0.15,0.3,0.5,0.7,0.85,1.0 OUT=/tmp/s2_carry_${i}.json \
    PYTHONPATH=. .venv/bin/python scripts/eval_weekly_rolling.py > /tmp/s2_carry_${i}.log 2>&1 && echo "  seed$i OK" || tail -2 /tmp/s2_carry_${i}.log
done

echo "[S2] 5) 集計 $(date +%H:%M:%S)"
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
# 基準点 = P掃引の全クラウド端×1.15
bl=[l.split() for l in open("/tmp/s2_baseline.txt") if l.startswith("P_cloud")]
costs=[float(x[1].split("=")[1]) for x in bl]; waits=[float(x[2].split("=")[1]) for x in bl]
REF=np.array([max(costs)*1.15, max(waits)*1.15])
print(f"REF={REF}")
ns=np.load("/tmp/s2_nsga2.npz"); hv_true=hv2d(ns["pf"].astype(float),REF)
rows=[]
for i in [1,2,3]:
    try:
        d=json.load(open(f"/tmp/s2_carry_{i}.json"))
        pts=np.array([[r["cost"],r["avg_wait"]] for r in d["results"]],dtype=float)
        fr=[r["frac"] for r in d["results"]]
        errs=[abs(r["cost"]-r["week_cost_target"])/r["week_cost_target"] for r in d["results"] if r["week_cost_target"]>1e6]
        rho=np.corrcoef(np.argsort(np.argsort(fr)),np.argsort(np.argsort(pts[:,0])))[0,1]
        hvr=hv2d(pts,REF)/hv_true
        rows.append([hvr,rho,np.mean(errs)])
        print(f"seed{i}: HV比 {hvr*100:5.1f}%  単調性ρ {rho:5.2f}  週目標誤差 {np.mean(errs)*100:4.0f}%")
    except Exception as e:
        print(f"seed{i}: FAIL {e}")
if rows:
    a=np.array(rows)
    sd=lambda x: x.std(ddof=1) if len(x)>1 else 0
    print(f"S2まとめ(窓リセット~9回): HV比 {a[:,0].mean()*100:.1f}±{sd(a[:,0])*100:.1f}%  ρ {a[:,1].mean():.2f}±{sd(a[:,1]):.2f}  誤差 {a[:,2].mean()*100:.0f}%")
PY
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "[S2] ALL DONE $(date +%H:%M:%S)"
