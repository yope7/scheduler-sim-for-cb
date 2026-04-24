"""
PCN evaluate() の診断: パレート崩壊（目的点が1種類）の原因切り分け用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _round_key(arr: np.ndarray, decimals: int = 5) -> Tuple:
    a = np.asarray(arr, dtype=np.float64).ravel()
    return tuple(np.round(a, decimals=decimals))


def count_unique_targets(
    returns: np.ndarray,
    horizons: np.ndarray,
) -> Tuple[int, List[Dict[str, Any]]]:
    """評価ループに入る前の (desired_return, horizon) の多様性を数える。"""
    n = int(returns.shape[0])
    rows: List[Dict[str, Any]] = []
    keys = []
    for i in range(n):
        r = np.asarray(returns[i], dtype=np.float64).ravel()
        h = float(horizons[i])
        key = (_round_key(r), round(h, 5))
        keys.append(key)
        rows.append(
            {
                "i": i,
                "return": r.tolist(),
                "horizon": h,
            }
        )
    unique_keys = set(keys)
    return len(unique_keys), rows


def count_unique_values(values: Sequence[Sequence[float]], decimals: int = 5) -> int:
    keys = [_round_key(np.asarray(v, dtype=np.float64), decimals=decimals) for v in values]
    return len(set(keys))


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
