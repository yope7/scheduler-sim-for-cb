#!/usr/bin/env python3
"""左上先端 PF のゴール判定（bulge JSON から）。

ゴール（図の意味）:
  - 低コスト帯 (0〜1.2M) で Eval PF が Archive に沿って滑らかに下降（プラトー＋急落を解消）
  - PF 全域は維持（cost_max≈6.5M, wait_min≈4.6k）

数値条件（すべて満たす）:
  - knee_drop <= GOAL_KNEE_DROP
  - low_slope_gap <= GOAL_LOW_SLOPE_GAP
  - cost_max >= MIN_COST_MAX, wait_min <= MAX_WAIT_MIN
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# knee05 基準（比較用）
REF_KNEE_DROP = 4148.5
REF_LOW_SLOPE_GAP = 1108.38  # low 帯 proxy（knee05 に low_slope 帯なし）

GOAL_KNEE_DROP = 3000.0
GOAL_LOW_SLOPE_GAP = 1800.0
MIN_COST_MAX = 5_000_000.0
MAX_WAIT_MIN = 5000.0


def load_bulge(path: Path) -> dict:
    data = json.loads(path.read_text())
    return data[0] if isinstance(data, list) else data


def score(entry: dict) -> tuple[float, dict]:
    cost_hi = float(entry["eval_cost_range"][1])
    wait_lo = float(entry["eval_wait_range"][0])
    knee = entry.get("knee") or {}
    drop = float(knee.get("drop", 1e9))
    knee_cost = float(knee.get("knee_cost", 0))
    cost_step = float(knee.get("cost_step", 1e9))

    ls = entry.get("bands", {}).get("low_slope", {})
    low = entry.get("bands", {}).get("low", {})
    low_gap = float(low.get("mean_gap", 1e9)) if low.get("n", 0) else 1e9
    if ls.get("n", 0):
        ls_gap = float(ls.get("mean_gap", 1e9))
        ls_n = int(ls["n"])
    else:
        ls_gap = low_gap
        ls_n = int(low.get("n", 0))

    ok_range = cost_hi >= MIN_COST_MAX and wait_lo <= MAX_WAIT_MIN
    goal = (
        ok_range
        and drop <= GOAL_KNEE_DROP
        and ls_gap <= GOAL_LOW_SLOPE_GAP
    )

    # 正規化不足（1.0 = ゴールちょうど）の最大値でランク（トレードオフに公平）
    knee_ratio = drop / GOAL_KNEE_DROP
    ls_ratio = ls_gap / GOAL_LOW_SLOPE_GAP
    chebyshev = max(knee_ratio, ls_ratio)
    s = chebyshev * 1000.0 + 0.1 * (drop + 0.45 * ls_gap)
    if not ok_range:
        s += 1e7

    vs_knee05 = drop <= REF_KNEE_DROP and ls_gap <= REF_LOW_SLOPE_GAP * 1.05

    return s, {
        "goal": goal,
        "ok_range": ok_range,
        "knee_drop": drop,
        "knee_cost": knee_cost,
        "knee_cost_step": cost_step,
        "low_slope_gap": ls_gap,
        "low_slope_n": ls_n,
        "low_gap": low_gap,
        "cost_max": cost_hi,
        "wait_min": wait_lo,
        "knee_ratio": knee_ratio,
        "ls_ratio": ls_ratio,
        "beats_knee05_both": vs_knee05,
    }


def main():
    path = Path(sys.argv[1])
    e = load_bulge(path)
    s, m = score(e)
    print(json.dumps({"score": s, **m}, indent=2))
    sys.exit(0 if m["goal"] else 1)


if __name__ == "__main__":
    main()
