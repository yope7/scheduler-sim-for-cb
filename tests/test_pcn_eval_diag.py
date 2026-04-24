"""pcn_eval_diag のユニットテスト（Ray / 環境不要）。"""

import numpy as np

from src.utils.pcn_eval_diag import count_unique_targets, count_unique_values


def test_count_unique_targets_all_identical():
    returns = np.ones((10, 2), dtype=np.float32)
    horizons = np.full(10, 42.0, dtype=np.float32)
    n, rows = count_unique_targets(returns, horizons)
    assert n == 1
    assert len(rows) == 10


def test_count_unique_targets_diverse():
    returns = np.array([[i, 0.0] for i in range(5)], dtype=np.float32)
    horizons = np.arange(5, dtype=np.float32)
    n, _ = count_unique_targets(returns, horizons)
    assert n == 5


def test_count_unique_values():
    vals = [[0.0, 1.0], [0.0, 1.0], [2.0, 3.0]]
    assert count_unique_values(vals) == 2
