#!/usr/bin/env bash
# env 配置ロジックの結果不変検証: fast_env=0(原典) と fused / sweep が trajectory hash 完全一致か。
# sweep は PCN_SWEEP_MIN_EVENTS=1 で強制発火（小イベントでも sweep 経路を踏ませる）。
# P=0.0(全オンプレ=非連続経路) / 0.7(混在) / 1.0(全クラウド=連続経路) を網羅。
set -u
cd /home/noguchi/scheduler-sim-for-cb
B=".venv/bin/python scripts/bench_env_alloc.py"
fail=0; n=0
th() { PYTHONPATH=. $1 NJ=$2 P=$3 JOB_SEED=$4 EP=${5:-2} .venv/bin/python scripts/bench_env_alloc.py 2>&1 | grep -aoE 'thash=[0-9a-f]+'; }
run() {  # NJ P SEED EP
  local NJ=$1 P=$2 SEED=$3 EP=$4
  local h0 h1 h2
  h0=$(PYTHONPATH=. PCN_FAST_ENV=0 NJ=$NJ P=$P JOB_SEED=$SEED EP=$EP .venv/bin/python scripts/bench_env_alloc.py 2>&1 | grep -aoE 'thash=[0-9a-f]+')
  h1=$(PYTHONPATH=. PCN_FAST_ENV=1 NJ=$NJ P=$P JOB_SEED=$SEED EP=$EP .venv/bin/python scripts/bench_env_alloc.py 2>&1 | grep -aoE 'thash=[0-9a-f]+')
  h2=$(PYTHONPATH=. PCN_FAST_ENV=1 PCN_FAST_ENV_SWEEP=1 PCN_SWEEP_MIN_EVENTS=1 NJ=$NJ P=$P JOB_SEED=$SEED EP=$EP .venv/bin/python scripts/bench_env_alloc.py 2>&1 | grep -aoE 'thash=[0-9a-f]+')
  n=$((n+1))
  if [ "$h0" = "$h1" ] && [ "$h0" = "$h2" ] && [ -n "$h0" ]; then
    echo "PASS NJ=$NJ P=$P seed=$SEED $h0"
  else
    echo "FAIL NJ=$NJ P=$P seed=$SEED fast0=$h0 fused=$h1 sweep=$h2"; fail=1
  fi
}
for SEED in 0 1 2; do
  for P in 0.0 0.5 0.7 1.0; do run 128 $P $SEED 2; done
done
for SEED in 0 1; do
  for P in 0.0 0.7 1.0; do run 256 $P $SEED 2; done
done
for P in 0.0 1.0; do run 512 $P 0 1; done
echo "=== cells=$n fail=$fail ==="