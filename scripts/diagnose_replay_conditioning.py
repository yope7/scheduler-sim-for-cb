#!/usr/bin/env python3
"""Replay / Phase1 キャッシュの conditioning 診断（反実仮想ペア・action偏り・PF多様性）。"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def _obs_key(obs: np.ndarray, decimals: int = 4) -> Tuple:
    return tuple(np.round(np.asarray(obs, dtype=np.float64).ravel(), decimals))


def _dr_key(dr: np.ndarray, decimals: int = 3) -> Tuple:
    return tuple(np.round(np.asarray(dr, dtype=np.float64).ravel(), decimals))


def _episode_from_entry(entry):
    if isinstance(entry, (list, tuple)) and len(entry) > 0:
        first = entry[0]
        if hasattr(first, "observation") and hasattr(first, "action"):
            return entry
        if len(entry) >= 3:
            return entry[2]
    return entry


def load_episodes(path: Path) -> List:
    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict):
        raw = payload.get("episodes", [])
    else:
        raw = payload
    return [_episode_from_entry(e) for e in raw]


def episode_step0_command(episode) -> np.ndarray:
    first = episode[0]
    dr = np.asarray(first.reward, dtype=np.float64)
  # step0 の remaining return はエピソード全体報酬と一致しやすい
    return dr


def diagnose(episodes: List, obs_decimals: int = 4) -> Dict[str, Any]:
    n_ep = len(episodes)
    step0_by_obs: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    action_counts = {0: 0, 1: 0}
    phase1_count = 0
    horizons_step0 = []

    for ep_i, episode in enumerate(episodes):
        if not episode:
            continue
        t0 = episode[0]
        obs = np.asarray(t0.observation, dtype=np.float64)
        dr = episode_step0_command(episode)
        act = int(t0.action)
        action_counts[act] = action_counts.get(act, 0) + 1
        hz = float(len(episode))
        horizons_step0.append(hz)
        is_phase1 = getattr(t0, "random_action_prob", None) is not None
        uid = getattr(t0, "_pcn_episode_uid", "")
        if is_phase1 or (isinstance(uid, str) and uid.startswith("phase1:")):
            phase1_count += 1
        ok = _obs_key(obs, obs_decimals)
        step0_by_obs[ok].append(
            {"ep_i": ep_i, "dr": dr, "action": act, "horizon": hz, "phase1": bool(is_phase1)}
        )

    # 反実仮想: 同じ obs_key で異なる dr かつ異なる action
    cf_pairs_dr_action = 0
    cf_pairs_dr_only = 0
    obs_with_multi_dr = 0
    obs_with_multi_action = 0
    for ok, rows in step0_by_obs.items():
        if len(rows) < 2:
            continue
        drs = {_dr_key(r["dr"]) for r in rows}
        acts = {r["action"] for r in rows}
        if len(drs) > 1:
            obs_with_multi_dr += 1
            cf_pairs_dr_only += len(rows) * (len(rows) - 1) // 2
        if len(drs) > 1 and len(acts) > 1:
            obs_with_multi_action += 1
            cf_pairs_dr_action += len(rows) * (len(rows) - 1) // 2

    values = []
    for episode in episodes:
        if not episode:
            continue
        first = episode[0]
        if hasattr(first, "objective_values") and first.objective_values is not None:
            obj = first.objective_values
            values.append([float(obj[0]), float(obj[2])])
    values_np = np.asarray(values, dtype=np.float64) if values else np.zeros((0, 2))
    n_unique_values = len(np.unique(np.round(values_np, 4), axis=0)) if len(values_np) else 0

    return {
        "n_episodes": n_ep,
        "n_unique_obs_keys_step0": len(step0_by_obs),
        "obs_buckets_with_2plus_episodes": sum(1 for v in step0_by_obs.values() if len(v) >= 2),
        "obs_with_multiple_commands": obs_with_multi_dr,
        "obs_with_multiple_commands_and_actions": obs_with_multi_action,
        "counterfactual_pairs_same_obs_diff_dr": cf_pairs_dr_only,
        "counterfactual_pairs_same_obs_diff_dr_diff_action": cf_pairs_dr_action,
        "step0_action_rate_1": float(action_counts.get(1, 0) / max(1, sum(action_counts.values()))),
        "phase1_episode_fraction": float(phase1_count / max(1, n_ep)),
        "n_unique_episode_values": n_unique_values,
        "horizon_step0_mean": float(np.mean(horizons_step0)) if horizons_step0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, help="initial_episodes_cache.pkl.gz")
    parser.add_argument("--output", default=None)
    parser.add_argument("--obs-decimals", type=int, default=4)
    args = parser.parse_args()

    episodes = load_episodes(Path(args.cache))
    report = diagnose(episodes, obs_decimals=args.obs_decimals)
    report["cache_path"] = str(Path(args.cache).resolve())

    out = Path(args.output) if args.output else Path(args.cache).parent / "replay_conditioning_diagnosis.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
