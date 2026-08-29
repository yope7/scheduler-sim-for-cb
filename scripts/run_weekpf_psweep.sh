#!/usr/bin/env bash
# フル週(50,278ジョブ)の濃いランダム掃引(参照フロント用)。13点を最大4並列。各点1プロセス。
set -u
cd /home/noguchi/scheduler-sim-for-cb
rm -f /tmp/s4_psweep_*.txt /tmp/weekpf_psweep.marker
cat > /tmp/psweep_one.py <<'PY'
import os, sys
os.environ.update({"DISTRIBUTED_PCN_USE_EVENT_OBS":"1","DISTRIBUTED_PCN_USE_EVENT_NATIVE":"1",
 "SCHEDULER_LEARNER_BITMAP":"0","SCHEDULER_OBS_URGENCY":"0","SCHEDULER_OBS_OCCUPANCY":"1"})
import numpy as np
from scripts.pcn_replay_snapshot import create_eval_env, load_config
NJ=50278; P=float(sys.argv[1])
env=create_eval_env(load_config("experiments/distributed_pcn/job_trace_weekB_full_pcn.yml"), job_seed=0, n_jobs=NJ)
rng=np.random.default_rng(7); env.reset(); done=False; st=0
while not done and st<NJ+50:
    a=1 if rng.random()<P else 0
    _,_,_,_,done=env.step(a); st+=1
cost,_,avgwait=env.calc_objective_values()
open(f"/tmp/s4_psweep_{P:.2f}.txt","w").write(f"{P:.3f} {cost:.6e} {avgwait:.3f}\n")
PY
for P in 0.00 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00; do
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 5; done
  ( PYTHONPATH=. .venv/bin/python /tmp/psweep_one.py $P > /tmp/s4_psweep_${P}.log 2>&1 ) &
done
wait
echo "DONE" > /tmp/weekpf_psweep.marker
echo "PSWEEP_ALL_DONE $(ls /tmp/s4_psweep_*.txt | wc -l)点"
