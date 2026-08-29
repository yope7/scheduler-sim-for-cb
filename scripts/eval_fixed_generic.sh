#!/usr/bin/env bash
# 配備評価バグ修正の取り直し用・汎用ラッパー。
# 学習時に実際に使われたアーキテクチャ設定(pcn_run.log/ckpt state_dictから確認したもの)を
# 呼び出し元が環境変数で明示し、conditioningバグ修正(PCN_COND_ADD_SCALE=0.25 / PCN_COMMAND_BALANCE=1)
# を必須で立てて eval_b2_compare.py を実行する。
# usage: CKPT=... CFG=... CG_MAX=... OUT=... [PCN_FILM=0 PCN_FOURIER_CMD=1 ...] bash scripts/eval_fixed_generic.sh
set -eu
cd /home/noguchi/scheduler-sim-for-cb

CKPT="${CKPT:?CKPT required}"
CFG="${CFG:?CFG required}"
CG_MAX="${CG_MAX:?CG_MAX required}"
OUT="${OUT:?OUT required}"
NJ="${NJ:?NJ required}"
NPROC="${NPROC:-32}"
NCMD="${NCMD:-20}"
KSAMP="${KSAMP:-0}"
SEEDS="${SEEDS:-0}"

mkdir -p "$(dirname "$OUT")"

echo "[eval_fixed_generic] CKPT=$CKPT CFG=$CFG CG_MAX=$CG_MAX OUT=$OUT NJ=$NJ" >&2
echo "[eval_fixed_generic] FILM=${PCN_FILM:-0} FOURIER_CMD=${PCN_FOURIER_CMD:-0} BANDS=${PCN_FOURIER_BANDS:-4} BANDS_COST=${PCN_FOURIER_BANDS_COST:-} FC_DEPTH=${PCN_FC_DEPTH:-2}" >&2
echo "[eval_fixed_generic] OBS_LOG=${PCN_OBS_LOG:-1} URGENCY=${SCHEDULER_OBS_URGENCY:-1} DEFER=${SCHEDULER_ALLOW_DEFER:-0} EFFICIENCY=${SCHEDULER_OBS_EFFICIENCY:-0}" >&2
echo "[eval_fixed_generic] COND_ADD_SCALE=${PCN_COND_ADD_SCALE:-0.25} COMMAND_BALANCE=${PCN_COMMAND_BALANCE:-1} LABEL_G=${PCN_LABEL_G:-0}" >&2

env \
  PCN_FILM="${PCN_FILM:-0}" \
  PCN_FOURIER_CMD="${PCN_FOURIER_CMD:-1}" \
  PCN_FOURIER_BANDS="${PCN_FOURIER_BANDS:-4}" \
  ${PCN_FOURIER_BANDS_COST:+PCN_FOURIER_BANDS_COST="$PCN_FOURIER_BANDS_COST"} \
  PCN_FC_DEPTH="${PCN_FC_DEPTH:-2}" \
  PCN_OBS_LOG="${PCN_OBS_LOG:-1}" \
  SCHEDULER_OBS_URGENCY="${SCHEDULER_OBS_URGENCY:-1}" \
  SCHEDULER_ALLOW_DEFER="${SCHEDULER_ALLOW_DEFER:-0}" \
  SCHEDULER_DEFER_OFFSET="${SCHEDULER_DEFER_OFFSET:-1}" \
  SCHEDULER_OBS_EFFICIENCY="${SCHEDULER_OBS_EFFICIENCY:-0}" \
  PCN_COND_ADD_SCALE="${PCN_COND_ADD_SCALE:-0.25}" \
  PCN_COMMAND_BALANCE="${PCN_COMMAND_BALANCE:-1}" \
  PCN_LABEL_G="${PCN_LABEL_G:-0}" \
  CKPT="$CKPT" CFG="$CFG" NJ="$NJ" SEEDS="$SEEDS" NCMD="$NCMD" KSAMP="$KSAMP" NPROC="$NPROC" \
  CG_MAX="$CG_MAX" OUT="$OUT" \
  PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py

echo "[eval_fixed_generic] done OUT=$OUT" >&2
