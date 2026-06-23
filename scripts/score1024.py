"""Score 1024 runs (no reference PF): final uniform-command PF size, cost/wait span, 2D hypervolume.
Compares runs for reproducibility. Reads pcn_mo_hv.json (pareto_fronts_per_eval[-1]).

Usage: python -m scripts.score1024 <exec_dir1> [<exec_dir2> ...]
HV uses a fixed nadir reference (cost_ref, wait_ref) so runs are comparable; both objectives minimized.
"""
import json
import sys
from pathlib import Path

import numpy as np

# fixed nadir reference for top1024 (all-cloud cost, all-onprem wait) — worst-case corner
COST_REF = float(__import__("os").environ.get("SCORE_COST_REF", "1.83e9"))
WAIT_REF = float(__import__("os").environ.get("SCORE_WAIT_REF", "1.64e6"))


def _final_pf(exec_dir: Path):
    f = exec_dir / "pcn_mo_hv.json"
    if not f.exists():
        # fall back to any pcn_mo_hv.json under exec_dir
        cand = list(exec_dir.rglob("pcn_mo_hv.json"))
        if not cand:
            return None
        f = cand[0]
    d = json.loads(f.read_text())
    pfs = d.get("pareto_fronts_per_eval") or []
    if not pfs:
        return None
    return np.asarray(pfs[-1], dtype=np.float64)


def _hv2d(pf, cost_ref, wait_ref):
    """2D hypervolume for minimization, columns [cost, wait], ref = nadir (worst)."""
    pts = pf[(pf[:, 0] <= cost_ref) & (pf[:, 1] <= wait_ref)]
    if len(pts) == 0:
        return 0.0
    # keep non-dominated (minimize both)
    order = pts[np.argsort(pts[:, 0])]
    nd = []
    best_w = np.inf
    for c, w in order:
        if w < best_w:
            nd.append((c, w)); best_w = w
    nd = np.array(sorted(nd, key=lambda x: x[0]))
    # sweep by increasing cost; area of dominated rectangles up to ref
    hv = 0.0
    prev_c = nd[0, 0]
    # integrate: for each step, width=(next_cost-cur_cost), height=(wait_ref - cur_wait)
    cs = list(nd[:, 0]) + [cost_ref]
    ws = list(nd[:, 1])
    for i in range(len(ws)):
        width = cs[i + 1] - cs[i]
        height = wait_ref - ws[i]
        if width > 0 and height > 0:
            hv += width * height
    return hv


def score(exec_dir: Path):
    pf = _final_pf(exec_dir)
    if pf is None or pf.size == 0:
        return {"dir": str(exec_dir), "error": "no PF"}
    # detect cost vs wait column by range magnitude (cost has the bigger absolute range for top1024)
    rng = pf.max(0) - pf.min(0)
    cost_col = int(np.argmax(rng))  # cost spans the most for top1024
    wait_col = 1 - cost_col
    pf2 = np.column_stack([pf[:, cost_col], pf[:, wait_col]])
    uniq = np.unique(np.round(pf2, 3), axis=0)
    return {
        "dir": str(exec_dir),
        "n_pf": int(len(uniq)),
        "cost_span": float(pf2[:, 0].max() - pf2[:, 0].min()),
        "wait_span": float(pf2[:, 1].max() - pf2[:, 1].min()),
        "cost_range": [float(pf2[:, 0].min()), float(pf2[:, 0].max())],
        "wait_range": [float(pf2[:, 1].min()), float(pf2[:, 1].max())],
        "hv": _hv2d(pf2, COST_REF, WAIT_REF),
        "hv_frac": _hv2d(pf2, COST_REF, WAIT_REF) / (COST_REF * WAIT_REF),
    }


def main():
    dirs = [Path(a) for a in sys.argv[1:]]
    results = [score(d) for d in dirs]
    for r in results:
        if "error" in r:
            print(f"{r['dir']}: {r['error']}"); continue
        print(f"{Path(r['dir']).parent.name}: n_pf={r['n_pf']} "
              f"cost_span={r['cost_span']:.3g} wait_span={r['wait_span']:.3g} "
              f"cost=[{r['cost_range'][0]:.3g},{r['cost_range'][1]:.3g}] "
              f"wait=[{r['wait_range'][0]:.3g},{r['wait_range'][1]:.3g}] "
              f"HV_frac={r['hv_frac']:.4f}")
    if len(results) >= 2 and all("error" not in r for r in results):
        # reproducibility: relative spread across runs
        import statistics as st
        for k in ("n_pf", "cost_span", "wait_span", "hv_frac"):
            vals = [r[k] for r in results]
            m = st.mean(vals); sd = st.pstdev(vals)
            print(f"  [{k}] mean={m:.4g} rel_spread={sd/m if m else 0:.2%}")


if __name__ == "__main__":
    main()
