#!/usr/bin/env python3
"""ジョブ数粒度実験(J18/J256/J1024/J4096)の評価集計。

判定は「外部評価」= scripts/eval_jscale_c3.sh (=eval_b2_compare.py に学習一致フラグを
明示したもの) の出力 npz を正とする。用途が「1点を指定して出させる」ため、学習中の内部統計
ではなく外部から注文した応答で測る必要があるため。
※ 外部評価は学習と同じ条件付けフラグを設定しないと別ネットワークになる。詳細は
   scripts/eval_jscale_c3.sh の冒頭コメント参照(必ずあのランナー経由で npz を作ること)。

内部記録(cmd_outcomes.jsonl)は CMD_JSONL を渡した時のみ参考値として併記する(判定には使わない)。

主指標(この順):
  1. 等コスト待ち比: pダイヤルの各点と同じコストでPCNの待ちが何倍か。中央値・勝敗数。
  2. 注文精度: 指令(cg,wg)と達成点のズレ(pダイヤルcost/wait幅で正規化したL2)。中央値/p90。符号つき内訳。
  3. カバー範囲: pダイヤルのcostレンジのうちPCNがオーバーラップでカバーする割合。
  4. 出せる点の種類: 指令に対し達成点(cost,wait)をレンジ1%許容で丸めたユニーク数。

usage:
  EVAL_NPZ=<eval_jscale_c3.sh出力.npz> SEED=0 SCALE_TAG=J256 \
    [CMD_JSONL=<run>/cmd_outcomes.jsonl] [TRUEPF_NPZ=... TRUEPF_KEY=pf] \
    OUT_PREFIX=results/eval_pf/jscale_c3/j256 \
    PYTHONPATH=. .venv/bin/python scripts/eval_jscale_c3.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL_NPZ = os.environ["EVAL_NPZ"]
SEED = int(os.environ.get("SEED", "0"))
SCALE_TAG = os.environ.get("SCALE_TAG", "scale")
CMD_JSONL = os.environ.get("CMD_JSONL", "")
TRUEPF_NPZ = os.environ.get("TRUEPF_NPZ", "")
TRUEPF_KEY = os.environ.get("TRUEPF_KEY", "pf")
OUT_PREFIX = os.environ.get("OUT_PREFIX", f"results/eval_pf/jscale_c3/{SCALE_TAG}")
os.makedirs(os.path.dirname(OUT_PREFIX) or ".", exist_ok=True)

data = np.load(EVAL_NPZ)
rp = data[f"rp_{SEED}"].astype(np.float64)          # (40,2) p-dial sweep [cost, wait]
greedy = data[f"greedy_{SEED}"].astype(np.float64)  # (NCMD,2) 指令応答
NCMD = len(greedy)

# eval_b2_compare.py と同じ規則で指令(cg,wg)を再構成(CG_LIST/WG_LIST 未使用時)
cg_hi = float(os.environ.get("CG_MAX", "0")) or float(rp[:, 0].max())
cg = np.linspace(float(rp[:, 0].min()), min(cg_hi, float(rp[:, 0].max())), NCMD)
order = np.argsort(rp[:, 0])
wg = np.interp(cg, rp[order, 0], rp[order, 1])

cost_range = float(rp[:, 0].max() - rp[:, 0].min()) or 1.0
wait_range = float(rp[:, 1].max() - rp[:, 1].min()) or 1.0

# 1. 等コスト待ち比
g_order = np.argsort(greedy[:, 0])
pd_cost_min, pd_cost_max = float(rp[:, 0].min()), float(rp[:, 0].max())
pcn_cost_min, pcn_cost_max = float(greedy[:, 0].min()), float(greedy[:, 0].max())
in_range = (rp[:, 0] >= pcn_cost_min) & (rp[:, 0] <= pcn_cost_max)
pcn_wait_at_pd = np.interp(rp[in_range, 0], greedy[g_order, 0], greedy[g_order, 1])
ratio = pcn_wait_at_pd / np.maximum(rp[in_range, 1], 1e-9)
n_compare = int(in_range.sum())
n_win = int((ratio < 1.0).sum())
equalcost_ratio_median = float(np.median(ratio)) if n_compare else float("nan")

# 2. 注文精度(正規化L2) + 符号つき内訳
dx = (greedy[:, 0] - cg) / cost_range
dy = (greedy[:, 1] - wg) / wait_range
l2 = np.sqrt(dx ** 2 + dy ** 2)

# 3. カバー範囲
overlap = max(0.0, min(pcn_cost_max, pd_cost_max) - max(pcn_cost_min, pd_cost_min))
coverage = overlap / (pd_cost_max - pd_cost_min) if pd_cost_max > pd_cost_min else float("nan")

# 4. 出せる点の種類
grid_c, grid_w = cost_range * 0.01, wait_range * 0.01
n_distinct = len({(round(c / grid_c), round(w / grid_w)) for c, w in greedy})

truepf = None
if TRUEPF_NPZ and os.path.exists(TRUEPF_NPZ):
    tp = np.load(TRUEPF_NPZ)
    arr = np.asarray(tp[TRUEPF_KEY] if TRUEPF_KEY in tp else tp[list(tp.keys())[0]], dtype=np.float64)
    truepf = arr[np.argsort(arr[:, 0])]

result = {
    "scale_tag": SCALE_TAG,
    "eval_npz": EVAL_NPZ,
    "n_cmd": NCMD,
    "equalcost_wait_ratio_median": equalcost_ratio_median,
    "equalcost_n_compare": n_compare,
    "equalcost_n_win": n_win,
    "equalcost_win_frac": n_win / n_compare if n_compare else float("nan"),
    "l2_median": float(np.median(l2)),
    "l2_p90": float(np.percentile(l2, 90)),
    "cost_over_frac": float((dx > 0).mean()),
    "cost_over_mean_norm": float(dx[dx > 0].mean()) if (dx > 0).any() else 0.0,
    "cost_under_mean_norm": float(dx[dx < 0].mean()) if (dx < 0).any() else 0.0,
    "wait_over_frac": float((dy > 0).mean()),
    "wait_over_mean_norm": float(dy[dy > 0].mean()) if (dy > 0).any() else 0.0,
    "wait_under_mean_norm": float(dy[dy < 0].mean()) if (dy < 0).any() else 0.0,
    "coverage": coverage,
    "pcn_cost_range": [pcn_cost_min, pcn_cost_max],
    "pdial_cost_range": [pd_cost_min, pd_cost_max],
    "n_distinct_points": n_distinct,
    "has_truepf": truepf is not None,
}

# 参考: 内部記録(学習中の指令追従統計)。判定には使わない。
if CMD_JSONL and os.path.exists(CMD_JSONL):
    rows = [json.loads(l) for l in open(CMD_JSONL) if l.strip()]
    if rows:
        it = max(r["iter"] for r in rows)
        sel = [r for r in rows if r["iter"] == it]
        icg = np.array([r["command_values"][0] for r in sel])
        iwg = np.array([r["command_values"][1] for r in sel])
        iac = np.array([r["achieved_values"][0] for r in sel])
        iaw = np.array([r["achieved_values"][1] for r in sel])
        il2 = np.sqrt(((iac - icg) / cost_range) ** 2 + ((iaw - iwg) / wait_range) ** 2)
        result["internal_ref"] = {
            "iter": it, "n": len(sel),
            "l2_median": float(np.median(il2)),
            "corr_cost": float(np.corrcoef(icg, iac)[0, 1]),
            "note": "学習中の内部統計。参考値であり判定には使わない",
        }

with open(OUT_PREFIX + "_metrics.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(json.dumps(result, indent=2, ensure_ascii=False))

# PF図
fig, ax = plt.subplots(figsize=(7.5, 6))
if truepf is not None:
    ax.plot(truepf[:, 0], truepf[:, 1], "-", color="#2ca02c", lw=2.2, zorder=2,
            label=f"true PF ({len(truepf)}pt)")
rp_sorted = rp[np.argsort(rp[:, 0])]
ax.plot(rp_sorted[:, 0], rp_sorted[:, 1], "o-", color="#888888", ms=4, lw=1.2, alpha=0.8,
        zorder=1, label="p-dial sweep (40pt)")
ax.scatter(greedy[:, 0], greedy[:, 1], s=34, c="#1a73e8", edgecolor="k", lw=0.4, zorder=4,
           label=f"PCN command response ({n_distinct} distinct)")
ax.set_xlabel("Cost")
ax.set_ylabel("Avg Wait")
ax.set_title(f"{SCALE_TAG}: PCN vs p-dial "
             f"(equalcost wait ratio med={equalcost_ratio_median:.3f}, L2 med={np.median(l2):.3f})",
             fontsize=10)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(OUT_PREFIX + "_pf.png", dpi=130, bbox_inches="tight")
print(f"saved {OUT_PREFIX}_pf.png")
