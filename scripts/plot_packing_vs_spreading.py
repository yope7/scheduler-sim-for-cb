#!/usr/bin/env python3
"""24ジョブ(密集) vs 256ジョブ(分散) の「真のPF・発見・方策再現」を2枚で対比。
   左: 24job 真のPF(全46run合体探索の非支配=99) + 単発run達成(~20)
   右: 256job 発見PF(111) + 達成(72)
   なぜ点数が違うか＝効率点の数ではなく cost レンジの広さ(間隔)を可視化。"""
import glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scripts.pcn_replay_snapshot import load_learner_replay_snapshot, episode_objectives_from_snapshot
from src.agents.pcn_agent import get_non_dominated_inds_minimize
from src.utils.pf_command_eval import dedupe_pf

# --- 24ジョブ: 全run合体探索の真のPF ---
allpts = []
for s in glob.glob('experiments/distributed_pcn/run_synth24_*/20*/learner_replay_snapshot.pkl.gz'):
    try:
        e = episode_objectives_from_snapshot(load_learner_replay_snapshot(s), 24)
        if len(e): allpts.append(e)
    except Exception: pass
allpts = np.vstack(allpts)
nd = get_non_dominated_inds_minimize(allpts); true24 = dedupe_pf(allpts[nd]); true24 = true24[np.argsort(true24[:,0])]
# 24ジョブ単発の達成（FiLM）
d24 = np.load('pf_film_fullpicture.npz'); ach24 = d24['achieved']; arch24 = d24['archive_pf']
nd2 = get_non_dominated_inds_minimize(ach24); apf24 = ach24[nd2]
u24 = len(np.unique(np.round(apf24/1000., 0), axis=0))
# 256ジョブ
d256 = np.load('pf_film256.npz'); ach256 = d256['achieved']; arch256 = d256['archive_pf']; ex256 = d256['explored']
nd3 = get_non_dominated_inds_minimize(ach256); apf256 = ach256[nd3]
u256 = len(np.unique(np.round(apf256/10000., 0), axis=0))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))
# 左: 24ジョブ
ax1.scatter(allpts[::20,0], allpts[::20,1], s=5, c="#dddddd", alpha=0.4, zorder=0, label=f"explored (all runs, {len(allpts)})")
ax1.plot(true24[:,0], true24[:,1], "-", color="#1a73e8", lw=1.6, zorder=2, label=f"TRUE PF, heavy explore ({len(true24)})")
ax1.scatter(arch24[:,0], arch24[:,1], s=18, c="#888888", zorder=3, label=f"single-run found ({len(arch24)})")
ax1.scatter(apf24[:,0], apf24[:,1], s=55, c="#d62728", edgecolor="k", lw=0.5, zorder=5, label=f"policy achieved (~{u24} distinct)")
ax1.set_title(f"24 jobs — cost packed in [0, {true24[:,0].max():.0f}]\n99 efficient pts EXIST, but policy resolves ~{u24}", fontsize=11)
ax1.set_xlabel("Cost"); ax1.set_ylabel("Average Waiting Time"); ax1.grid(alpha=0.3); ax1.legend(fontsize=9, loc="upper right")
# 右: 256ジョブ
ax2.scatter(ex256[::20,0], ex256[::20,1], s=5, c="#dddddd", alpha=0.4, zorder=0, label=f"explored ({len(ex256)})")
ax2.plot(arch256[np.argsort(arch256[:,0]),0], arch256[np.argsort(arch256[:,0]),1], "-", color="#888888", lw=1.3, zorder=2, label=f"discovered PF ({len(arch256)})")
ax2.scatter(apf256[:,0], apf256[:,1], s=40, c="#d62728", edgecolor="k", lw=0.4, zorder=5, label=f"policy achieved ({u256}+ distinct)")
ax2.set_title(f"256 jobs — cost spread over [0, {arch256[:,0].max():.0f}] (~20×)\n111 pts spread out → policy resolves {u256}+", fontsize=11)
ax2.set_xlabel("Cost"); ax2.grid(alpha=0.3); ax2.legend(fontsize=9, loc="upper right")
fig.suptitle("Why 256-job gives more PF points: NOT more efficient points (both ~100), but 20× wider spacing → discoverable + resolvable", fontsize=12, y=1.02)
fig.tight_layout(); fig.savefig("pf_packing_vs_spreading.png", dpi=120, bbox_inches="tight")
print(f"saved pf_packing_vs_spreading.png")
print(f"24job: true PF={len(true24)}, single-run found={len(arch24)}, achieved~{u24}")
print(f"256job: discovered={len(arch256)}, achieved={u256}+")
