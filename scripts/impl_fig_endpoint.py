#!/usr/bin/env python3
"""元PCNで「端の点が消える」機序の厳密図(研究用)。
実装 pcn_agent.py:3464-3471 の指令生成を正確に再現:
  r_i ~ Uniform(非支配集合),  r_obj ~ Uniform{0,1},  desired[r_obj] += U(0, sigma_obj * ALPHA)   (ALPHA=0.2)
これを「中間に密集した archive」に適用し、(A)端起点のナッジが内側(中間)を向くこと、
(B)多数サンプルの指令分布が中間に集中し端が枯れることを定量化する。
出力 docs/figures/impl_endpoint.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
ALPHA = 0.2   # _COMMAND_ALPHA 既定

# --- 中間に密集した非支配フロント(自己強化が進んだ状態を模す) ---
# cost を beta(2,2)で中間密に並べ、wait は凸トレードオフ wait = 1-sqrt(cost) 系
u = np.sort(rng.beta(2.0, 2.0, 60))          # 中間に密
u = np.concatenate([[0.0], u, [1.0]])         # 両端を明示的に1点ずつ(疎)
cost = u                                       # 正規化 cost [0,1]
wait = (1.0 - np.sqrt(u)) * 0.9 + 0.05         # 凸フロント(右下がり)
pf = np.column_stack([cost, wait])
sig = pf.std(axis=0)                           # 各目的の標準偏差(ナッジ幅の基準)

# --- 端の定義 ---
i_cost0 = int(np.argmin(pf[:, 0]))   # コスト端(cost=0, wait最大)
i_wait0 = int(np.argmin(pf[:, 1]))   # 待ち端(wait最小, cost最大)


def nudge(point):
    """実装どおり: 1目的をランダムに選び +U(0, sigma*ALPHA)。返り値=ナッジ後の指令。"""
    d = point.copy()
    r_obj = rng.integers(0, 2)
    d[r_obj] += rng.uniform(0, sig[r_obj] * ALPHA)
    return d, r_obj


fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4))

# ===== パネルA: 端起点のナッジベクトル =====
axA.plot(pf[:, 0], pf[:, 1], "-", color="#9aa7b8", lw=1.4, zorder=1)
axA.scatter(pf[:, 0], pf[:, 1], s=14, color="#6b7787", zorder=2, label="non-dominated archive (mid-dense)")
# 端2点から各64本ナッジ(実装式)し、内側に向く割合を測る
for i_end, name, col in [(i_cost0, "cost-end", "#e0556f"), (i_wait0, "wait-end", "#4da3ff")]:
    p = pf[i_end]
    inward = 0
    for _ in range(64):
        d, r_obj = nudge(p)
        axA.annotate("", xy=(d[0], d[1]), xytext=(p[0], p[1]),
                     arrowprops=dict(arrowstyle="->", color=col, alpha=0.35, lw=0.8), zorder=3)
        # 内側(中間方向)=コスト端ならcost増, 待ち端ならwait増
        if (i_end == i_cost0 and d[0] > p[0]) or (i_end == i_wait0 and d[1] > p[1]):
            inward += 1
    axA.scatter([p[0]], [p[1]], s=90, color=col, edgecolor="k", zorder=4,
                label=f"{name}: {inward}/64 nudged inward")
axA.set_title("(A) command nudge from endpoints\n d[r_obj] += U(0, sigma·0.2),  r_obj~Uniform{0,1}", fontsize=11)
axA.set_xlabel("cost (norm)"); axA.set_ylabel("wait (norm)")
axA.grid(alpha=0.3); axA.legend(fontsize=8, loc="upper right")

# ===== パネルB: 多数サンプルの指令分布(件数比例で中間集中) =====
N = 4000
cmd = np.empty((N, 2))
for k in range(N):
    r_i = rng.integers(0, len(pf))     # 非支配集合から一様選択(件数比例で中間が当たりやすい)
    d, _ = nudge(pf[r_i])
    cmd[k] = d
axB.hist(cmd[:, 0], bins=40, color="#f0a050", alpha=0.85)
axB.axvline(pf[i_cost0, 0], color="#e0556f", ls="--", lw=1.6, label="cost-end (cost=0)")
axB.axvline(pf[i_wait0, 0], color="#4da3ff", ls="--", lw=1.6, label="wait-end (cost=max)")
# 端近傍に落ちる指令の割合
edge_lo = np.mean(cmd[:, 0] < 0.05)
edge_hi = np.mean(cmd[:, 0] > 0.95)
mid = 1 - edge_lo - edge_hi
axB.set_title(f"(B) distribution of {N} generated commands (cost)\n"
              f"reach extremes(5%): low {edge_lo*100:.1f}% / high {edge_hi*100:.1f}%  -  middle {mid*100:.1f}%", fontsize=11)
axB.set_xlabel("commanded cost (norm)"); axB.set_ylabel("count")
axB.grid(alpha=0.3, axis="y"); axB.legend(fontsize=8)

fig.suptitle("Why endpoints vanish in vanilla PCN (rigorous): generation never re-creates endpoints (inward nudge + count-proportional pick)  |  pcn_agent.py:_choose_commands :3464-3471",
             fontsize=11.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("docs/figures/impl_endpoint.png", dpi=110, bbox_inches="tight")
print("[SAVED] docs/figures/impl_endpoint.png")
print(f"cost-end inward share & command edge shares: low={edge_lo:.3f} high={edge_hi:.3f} mid={mid:.3f}")
