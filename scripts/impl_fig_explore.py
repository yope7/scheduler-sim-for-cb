#!/usr/bin/env python3
"""探索チューニング(eval-gap feedback)の厳密図(研究用)。実装 pf_eval_gap.py を再現:
  各cost帯で gap = eval達成wait - archive最小wait;  ref_gap = ref_frac * wait_max (ref_frac=0.06);
  帯の増幅率 mult = min(BOOST_MAX=5, 1 + (mean_gap - ref_gap)/ref_gap)  (gap>ref の帯のみ)。
合成の archive PF(端が弱い) と eval PF(端で未達=gap大) を作り、帯ごとの gap→mult を定量化。
出力 docs/figures/impl_explore.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF_FRAC, BOOST_MAX = 0.06, 5.0

cost = np.linspace(0.0, 1.0, 200)
# archive PF(達成済み): 端(低cost/高cost)が弱く中間が良い凸
archive = (1.0 - np.sqrt(cost)) * 0.9 + 0.05
# eval PF(均等格子greedy): 中間はarchiveに近いが両端で未達(waitが上振れ=gap)
gap_shape = 0.18 * (np.exp(-((cost - 0.0) ** 2) / 0.02) + 0.6 * np.exp(-((cost - 1.0) ** 2) / 0.03))
evalpf = archive + gap_shape
wait_max = float(max(archive.max(), evalpf.max()))
ref_gap = REF_FRAC * wait_max

# cost帯(8分割)で gap と mult
nb = 8
edges = np.linspace(0, 1, nb + 1)
band_c, mean_gap, mult = [], [], []
for b in range(nb):
    m = (cost >= edges[b]) & (cost < edges[b + 1] + (1e-9 if b == nb - 1 else 0))
    g = float(np.mean(evalpf[m] - archive[m]))
    band_c.append((edges[b] + edges[b + 1]) / 2); mean_gap.append(g)
    mult.append(min(BOOST_MAX, 1.0 + (g - ref_gap) / ref_gap) if g > ref_gap else 1.0)
band_c, mean_gap, mult = map(np.array, (band_c, mean_gap, mult))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

# A: archive vs eval PF, gap, ref_gap
axA.plot(cost, archive, color="#5ee0a0", lw=2, label="archive PF (achieved)")
axA.plot(cost, evalpf, color="#e0556f", lw=2, ls="--", label="eval PF (uniform-grid greedy)")
axA.fill_between(cost, archive, evalpf, color="#f0a050", alpha=0.3, label="gap = eval - archive")
for b in range(nb):
    axA.axvline(edges[b], color="#444", lw=0.4, alpha=0.4)
axA.set_title(f"(A) gap per cost band  (ref_gap = 0.06·wait_max = {ref_gap:.3f})", fontsize=11)
axA.set_xlabel("cost (norm)"); axA.set_ylabel("wait (norm)"); axA.grid(alpha=0.3); axA.legend(fontsize=8)

# B: 帯ごとの mean_gap と mult(増幅率)
x = np.arange(nb)
axB.bar(x - 0.2, mean_gap, 0.4, color="#f0a050", label="mean gap")
axB.axhline(ref_gap, color="#888", ls="--", lw=1.2, label=f"ref_gap={ref_gap:.3f}")
ax2 = axB.twinx()
ax2.bar(x + 0.2, mult, 0.4, color="#4da3ff", label="boost mult")
ax2.axhline(BOOST_MAX, color="#4da3ff", ls=":", lw=1, alpha=0.6)
ax2.set_ylim(0, BOOST_MAX + 0.6); ax2.set_ylabel("replay boost mult", color="#4da3ff")
for xi, mu in zip(x, mult):
    if mu > 1.0:
        ax2.text(xi + 0.2, mu + 0.1, f"x{mu:.1f}", ha="center", fontsize=8, color="#1a73e8")
axB.set_title("(B) band gap -> boost  mult=min(5, 1+(gap-ref)/ref)\nonly the two ENDS (gap>ref) get amplified", fontsize=11)
axB.set_xlabel("cost band (0=cheap-end ... 7=expensive-end)"); axB.set_ylabel("mean gap")
axB.set_xticks(x); axB.grid(alpha=0.3, axis="y")
h1, l1 = axB.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
axB.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")

fig.suptitle("(4) explore-tuning / eval-gap feedback (rigorous): under-covered bands get replay boost  |  pf_eval_gap.py",
             fontsize=11.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("docs/figures/impl_explore.png", dpi=110, bbox_inches="tight")
print("[SAVED] docs/figures/impl_explore.png")
print(f"ref_gap={ref_gap:.3f} band mults={np.round(mult,2).tolist()}")
