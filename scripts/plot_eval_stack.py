#!/usr/bin/env python
"""profile_eval_stack.py が吐いた cProfile pstats から、eval(greedy rollout)経路の
「関数スタック時間図」を描く。
  左 : 上位関数の cumulative 時間バー(どこで時間を使うか)
  右 : 呼び出し階層のアイシクル(flame風; 幅=cumulative時間, %は総時間比)
契約ノイズ(他プロセス)に依らないよう全て総時間比(%)で表示。
出力: docs/figures/eval_stack_512.png
usage: PSTATS=/tmp/eval_stack.pstats SCALE=512 .venv/bin/python scripts/plot_eval_stack.py
"""
import os
import pstats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PSTATS = os.environ.get("PSTATS", "/tmp/eval_stack.pstats")
SCALE = os.environ.get("SCALE", "512")
OUT = os.environ.get("OUT", "docs/figures/eval_stack_512.png")

st = pstats.Stats(PSTATS)
total = st.total_tt


def find(sub, func):
    """(filename 部分一致, funcname 完全一致) で (cumtime, tottime) を返す。"""
    for (fn, ln, fname), (cc, nc, tt, ct, callers) in st.stats.items():
        if func == fname and sub in fn:
            return ct, tt
    return 0.0, 0.0


# --- 意味レイヤごとの代表関数(label, filename部分一致, funcname, 色) ---
LAYERS = [
    ("NN forward (policy net)", "pcn_agent.py", "forward", "#4e79a7"),
    ("torch.tensor alloc", "", "<built-in method torch.tensor>", "#a0cbe8"),
    ("tensor .item() sync", "", "<method 'item' of 'torch._C.TensorBase' objects>", "#86bcb6"),
    ("env: allocation sweep", "event_native_env.py", "_find_event_allocation_sweep", "#e15759"),
    ("env: pick_free_from_count", "event_native_env.py", "_pick_free_from_count", "#ff9d9a"),
    ("env: nodes_free_interval", "event_native_env.py", "_find_nodes_free_for_interval", "#fabfd2"),
    ("obs: front_job_urgency", "event_c_env.py", "_front_job_urgency", "#f28e2b"),
    ("numpy roll (obs/env)", "numeric.py", "roll", "#bab0ac"),
]


def builtin(name):
    for (fn, ln, fname), v in st.stats.items():
        if fname == name:
            return v[3], v[2]
    return 0.0, 0.0


rows = []
for label, sub, func, col in LAYERS:
    if func.startswith("<"):
        ct, tt = builtin(func)
    else:
        ct, tt = find(sub, func)
    rows.append((label, ct, tt, col))
rows.sort(key=lambda r: -r[2])  # self time 降順

fig = plt.figure(figsize=(15, 6.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.28)

# ---- 左: self(tottime) バー ----
axL = fig.add_subplot(gs[0])
labels = [r[0] for r in rows]
selfs = [r[2] / total * 100 for r in rows]
cols = [r[3] for r in rows]
y = range(len(rows))
axL.barh(list(y), selfs, color=cols, edgecolor="#333", lw=.6)
for i, (s, r) in enumerate(zip(selfs, rows)):
    axL.text(s + .4, i, f"{s:.0f}%", va="center", fontsize=9)
axL.set_yticks(list(y)); axL.set_yticklabels(labels, fontsize=9.5)
axL.invert_yaxis()
axL.set_xlabel("SELF time (tottime) as % of total eval time")
axL.set_title(f"Where eval CPU time is spent  (trace n_jobs={SCALE})", fontsize=12, fontweight="bold")
axL.grid(axis="x", alpha=.3)

# ---- 右: 呼び出し階層アイシクル(子は親幅を実測 cumtime 比で正規化=各層が親に収まる) ----
axR = fig.add_subplot(gs[1])
rep_ct, _ = find("profile_eval_stack.py", "workload")
ep_ct, _ = find("pcn_agent.py", "_run_episode")
act_ct, _ = find("pcn_agent.py", "_act")
fwd_ct, fwd_tt = find("pcn_agent.py", "forward")
step_ct, _ = find("event_native_env.py", "step")
alloc_ct, _ = find("event_native_env.py", "_find_event_allocation")
sweep_ct, sweep_tt = find("event_native_env.py", "_find_event_allocation_sweep")
obs_ct, _ = find("event_c_env.py", "get_observation")
urg_ct, _ = find("event_c_env.py", "_front_job_urgency")
ttensor_ct, _ = builtin("<built-in method torch.tensor>")
item_ct, _ = builtin("<method 'item' of 'torch._C.TensorBase' objects>")

# 呼び出し木: 各ノード=(label, cumtime, color, [children]). 子の合計が親を超える分は
# "other"(灰)で詰め、親幅(画面%)に正規化して収める。
tree = ("one greedy rollout (_run_episode)", max(ep_ct, 1e-9), "#bab0ac", [
    ("_act → policy NN", act_ct, "#4e79a7", [
        ("forward (MLP math)", fwd_tt, "#3b6694", []),
        ("torch.tensor (per-step alloc)", ttensor_ct, "#a0cbe8", []),
        (".item() (CPU↔ sync)", item_ct, "#86bcb6", []),
    ]),
    ("env.step", step_ct, "#b07aa1", [
        ("_find_event_allocation", alloc_ct, "#e15759", [
            ("_find_event_allocation_sweep", sweep_ct, "#c43d3d", []),
        ]),
        ("get_observation", obs_ct, "#f28e2b", [
            ("_front_job_urgency (urgency obs)", urg_ct, "#d97916", []),
        ]),
    ]),
])
H = 1.0


def draw(node, x0, width, depth):
    label, ct, col, children = node
    axR.add_patch(Rectangle((x0, -depth * H), width, H * 0.9, facecolor=col, edgecolor="white", lw=1.1))
    if width >= 3.2:
        share = ct / max(ep_ct, 1e-9) * 100
        light = col in ("#a0cbe8", "#86bcb6", "#bab0ac")
        axR.text(x0 + width / 2, -depth * H + H * 0.45,
                 f"{label}\n{share:.0f}%", ha="center", va="center",
                 fontsize=8.0, color="#222" if light else "white")
    csum = sum(c[1] for c in children)
    if csum <= 0:
        return
    # 子+other を親実幅に正規化
    scale = width / max(csum, ct, 1e-9)
    cx = x0
    for c in children:
        draw(c, cx, c[1] * scale, depth + 1)
        cx += c[1] * scale


draw(tree, 0, 100, 0)
axR.set_xlim(-1, 101)
axR.set_ylim(-3 * H - 0.1, H + 0.15)
axR.axis("off")
axR.set_title("Call-stack time (icicle; width ∝ cumulative time)", fontsize=12, fontweight="bold")
axR.text(50, H * 0.08, "top = whole rollout · downward = called-by · % of rollout", ha="center", fontsize=8.5, color="#666")

_d = max(ep_ct, 1e-9)
fig.suptitle(f"eval (greedy) function-stack timing — trace {SCALE}J  "
             f"[NN≈{act_ct / _d * 100:.0f}%  env.step≈{step_ct / _d * 100:.0f}% "
             f"(alloc≈{alloc_ct / _d * 100:.0f}%, urgency-obs≈{obs_ct / _d * 100:.0f}%)]",
             fontsize=11, y=1.0)
fig.savefig(OUT, dpi=125, bbox_inches="tight")
print("saved", OUT)
print(f"total={total:.2f}s  _run_episode={ep_ct:.2f}s  _act={act_ct:.2f}s  step={step_ct:.2f}s "
      f"forward={fwd_tt:.2f}s sweep={sweep_tt:.2f}s urgency={urg_ct:.2f}s")
