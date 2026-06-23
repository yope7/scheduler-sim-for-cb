#!/usr/bin/env python3
"""全24ジョブ run のsnapshotから非支配エピソード（真のPF~99点）を抽出・結合し、
   initial_episodes_cache 形式で保存。これを DISTRIBUTED_PCN_INITIAL_EPISODE_CACHE_PATH に
   渡すと、24ジョブ run が単発探索(41点)ではなく合体探索(99点)を教師にできる。
   usage: OUT=combined_pf_cache.pkl.gz python scripts/build_combined_pf_cache.py"""
import os, glob, gzip, pickle
import numpy as np
from scripts.pcn_replay_snapshot import load_learner_replay_snapshot
from src.agents.pcn_agent import get_non_dominated_inds_minimize

OUT = os.environ.get("OUT", "combined_pf_cache.pkl.gz")
NJ = int(os.environ.get("NJ", "24"))
PER_POINT = int(os.environ.get("PER_POINT", "4"))   # 1点あたり保持エピソード数(学習多様性)

def ep_obj(episode):
    first = episode[0]
    if hasattr(first, "objective_values") and first.objective_values is not None:
        o = first.objective_values
        return float(o[0]), float(o[2])
    r = np.asarray(first.reward, dtype=np.float64)
    return -float(r[1]), -float(r[0]) / max(1, NJ)

all_eps, all_obj = [], []
snaps = glob.glob(f'experiments/distributed_pcn/run_synth{NJ}_*/20*/learner_replay_snapshot.pkl.gz')
for s in snaps:
    try:
        snap = load_learner_replay_snapshot(s)
    except Exception:
        continue
    for ep in snap.get("episodes", []):
        if not ep:
            continue
        all_eps.append(ep); all_obj.append(ep_obj(ep))
all_obj = np.asarray(all_obj, dtype=np.float64)
print(f"集めた episode={len(all_eps)} from {len(snaps)} runs")

nd = get_non_dominated_inds_minimize(all_obj)
nd_obj = all_obj[nd]
order = np.argsort(nd_obj[:, 0])
nd_idx = [nd[i] for i in order]; nd_obj = nd_obj[order]
# 各distinct点(~200cost,2wait)あたり最大 PER_POINT エピソードを保持
seen = {}
keep = []
for i, o in zip(nd_idx, nd_obj):
    key = (round(o[0] / 200.0), round(o[1] / 2.0))
    c = seen.get(key, 0)
    if c < PER_POINT:
        seen[key] = c + 1; keep.append(i)
curated = [all_eps[i] for i in keep]
uniq_pts = len(seen)
print(f"非支配={len(nd)}, distinct点={uniq_pts}, 保持episode={len(curated)} (PER_POINT={PER_POINT})")

payload = {
    "metadata": {"n_jobs": NJ, "use_event_obs": True, "source": "combined_pf_cache",
                 "total_episodes": len(curated), "distinct_points": uniq_pts},
    "episodes": curated,
}
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with gzip.open(OUT, "wb") as f:
    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"saved {OUT}  size={os.path.getsize(OUT)/1e6:.1f}MB  episodes={len(curated)} distinct={uniq_pts}")
