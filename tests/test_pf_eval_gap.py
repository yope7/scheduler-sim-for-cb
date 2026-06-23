"""Eval–Archive ギャップ検出と replay ブースト。"""
import os
import unittest

import numpy as np

from src.utils.pf_eval_gap import (
    DEFAULT_GAP_BANDS,
    gap_band_boosts,
    gap_bands_from_env,
    pf_gap_in_band,
    summarize_band_gaps,
)
from src.utils.pf_command_eval import stratified_sample_pf
from scripts.score_eval_pf_vs_reference import (
    command_achievability_stats,
    score_eval_vs_reference,
)


class TestPfEvalGap(unittest.TestCase):
    def test_pf_gap_in_band(self):
        archive = np.array([[0, 100], [5e5, 120], [1e6, 80]], dtype=np.float64)
        eval_pf = np.array([[0, 200], [5e5, 180], [1e6, 85]], dtype=np.float64)
        st = pf_gap_in_band(eval_pf, archive, 0, 1.2e6)
        self.assertGreater(st["n"], 0)
        self.assertGreater(st["mean_gap"], 0)

    def test_gap_bands_from_env_names(self):
        os.environ["PCN_EVAL_GAP_BAND_NAMES"] = "low,low_slope"
        bands = gap_bands_from_env()
        self.assertEqual(len(bands), 2)
        self.assertEqual(bands[0][0], "low")
        del os.environ["PCN_EVAL_GAP_BAND_NAMES"]

    def test_gap_band_boosts_only_weak_bands(self):
        summary = {
            "low_slope": {"n": 10, "mean_gap": 2500.0, "max_gap": 3000},
            "high": {"n": 5, "mean_gap": 100.0, "max_gap": 200},
        }
        boosts = gap_band_boosts(summary, DEFAULT_GAP_BANDS, ref_gap=1200, max_boost=3.0)
        self.assertEqual(len(boosts), 1)
        self.assertAlmostEqual(boosts[0][2], 2.083333333333333, places=5)

    def test_apply_eval_gap_step_boost(self):
        from src.agents.pcn_agent import PCN

        class _Stub:
            _eval_gap_band_boosts = [(0.0, 1.2e6, 2.0)]

            _command_cost_in_mid_band = PCN._command_cost_in_mid_band

        stub = _Stub()
        dr = np.array([[-1e6, -5e5], [-1e6, -3e5]], dtype=np.float32)
        w = np.ones(2, dtype=np.float64)
        w2 = PCN._apply_eval_gap_step_boost(stub, w.copy(), dr)
        self.assertEqual(w2[0], 2.0)
        self.assertEqual(w2[1], 2.0)

    def test_stratified_sample_pf_reserves_low_wait_commands(self):
        pf = np.array(
            [
                [0.0, 1000.0],
                [100.0, 900.0],
                [200.0, 800.0],
                [300.0, 700.0],
                [400.0, 600.0],
                [500.0, 500.0],
                [600.0, 400.0],
                [700.0, 300.0],
                [800.0, 200.0],
                [900.0, 100.0],
            ],
            dtype=np.float64,
        )
        sample = stratified_sample_pf(
            pf,
            6,
            rng=np.random.default_rng(0),
            low_wait_max=350.0,
            low_wait_quota=3,
        )
        self.assertEqual(sample.shape, (6, 2))
        self.assertTrue(np.any(np.isclose(sample[:, 0], 0.0)))
        self.assertTrue(np.any(np.isclose(sample[:, 1], 100.0)))
        self.assertGreaterEqual(int((sample[:, 1] <= 350.0).sum()), 3)

    def test_low_wait_achievability_can_gate_score(self):
        ref = np.array([[0.0, 1000.0], [100.0, 500.0], [200.0, 200.0]], dtype=np.float64)
        targets = np.array([[100.0, 500.0], [200.0, 200.0]], dtype=np.float64)
        achieved = np.array([[100.0, 650.0], [200.0, 250.0]], dtype=np.float64)
        ach = command_achievability_stats(achieved, targets, low_wait_max=600.0)
        self.assertEqual(ach["low_wait_target_n"], 2)
        self.assertAlmostEqual(ach["low_wait_covered_frac"], 0.5)

        score = score_eval_vs_reference(
            achieved,
            ref,
            eval_is_pf_commands=True,
            command_targets=targets,
            low_wait_max=600.0,
            min_low_wait_covered_frac=0.75,
            mean_gap_max=1e6,
            frac_bad_max=1.0,
            min_unique_pf=1,
        )
        self.assertFalse(score["passed"])


if __name__ == "__main__":
    unittest.main()
