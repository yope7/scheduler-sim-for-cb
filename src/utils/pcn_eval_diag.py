"""PCN 評価診断 JSONL ユーティリティ。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")


def count_unique_targets(returns, horizons) -> Tuple[int, List]:
    keys = set()
    rows: List[Tuple] = []
    for r, h in zip(returns, horizons):
        rk = tuple(np.round(np.asarray(r, dtype=np.float64).ravel(), 5))
        key = (rk, float(h))
        if key not in keys:
            keys.add(key)
            rows.append(key)
    return len(keys), rows


def count_unique_values(e_values) -> int:
    keys = {
        tuple(np.round(np.asarray(v, dtype=np.float64).ravel(), 4))
        for v in e_values
    }
    return len(keys)
