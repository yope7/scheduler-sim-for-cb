#!/usr/bin/env bash
# trace256スクリーンの rich eval。各 tag×seed の final_model.pth を rich_eval_cell.py で評価
# (trace256=JOB_TYPE2・デフォルト参照=WORKLOAD_CALIBに一致)。→ /tmp/scr256_{tag}_{i}.json
set -u
cd /home/noguchi/scheduler-sim-for-cb
TAGS="${TAGS:-base fb2 wd ct}"; REPS="${REPS:-2}"; PAR="${PAR:-4}"
CFG="experiments/distributed_pcn/job_trace_256_pcn.yml"
run_eval(){ local tag="$1" i="$2" d ck
  d=$(ls -dt experiments/distributed_pcn/run_synth256_scr_${tag}_${i}/2026* 2>/dev/null | head -1)
  ck="$d/final_model.pth"
  [ -f "$ck" ] || ck=$(find "$d"/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
  if [ ! -f "$ck" ]; then echo "{\"tag\":\"$tag\",\"seed\":$i,\"err\":\"no ckpt\"}" > /tmp/scr256_${tag}_${i}.json; return; fi
  env CKPT="$ck" NJ=256 CFG="$CFG" JOB_TYPE=2 \
      TRACE_PATH="job_trace/FY2024/scacctreq_202412_top1024_jobs.csv" \
      PYTHONPATH=. uv run python scripts/rich_eval_cell.py > /tmp/scr256_${tag}_${i}.json 2>/tmp/scr256_${tag}_${i}.err
  echo "[scr256eval] $tag _$i done $(date +%H:%M:%S)"
}
for tag in $TAGS; do for i in $(seq 1 "$REPS"); do
  while [ "$(jobs -rp|wc -l)" -ge "$PAR" ]; do sleep 3; done
  run_eval "$tag" "$i" &
done; done
wait
echo "DONE" > /tmp/scr256eval.marker
echo "[scr256eval] ALL DONE $(date +%H:%M:%S)"
echo "=== 速報 ==="
uv run python - <<'PY'
import json, glob, os
import numpy as np
rows={}
for tag in ["base","fb2","wd","ct"]:
    hv=[]; cd=[]; npf=[]
    for i in [1,2,3]:
        p=f"/tmp/scr256_{tag}_{i}.json"
        if not os.path.exists(p): continue
        try:
            d=json.loads([l for l in open(p) if l.strip().startswith("{")][-1])
            if "hv" in d: hv.append(d["hv"]); cd.append(d.get("cmd_dist")); npf.append(d.get("n_pf"))
        except: pass
    if hv:
        rows[tag]=(np.mean(hv), np.mean([x for x in cd if x is not None]), np.mean([x for x in npf if x]))
print(f"{'tag':6s} {'HV(大=良)':10s} {'追従(小=良)':12s} {'n_pf':6s}")
for t in ["base","fb2","wd","ct"]:
    if t in rows:
        hv,cd,npf=rows[t]; print(f"{t:6s} {hv:<10.4f} {cd:<12.4f} {npf:<6.0f}")
if "base" in rows:
    bh,bc,_=rows["base"]
    print("\n=== vs base 改善 (HV差+/追従差-が良) ===")
    for t in ["fb2","wd","ct"]:
        if t in rows:
            h,c,_=rows[t]; print(f"{t}: HV {h-bh:+.4f}  追従 {c-bc:+.4f}")
PY