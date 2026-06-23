"""ワークロードキャリブレーションのスモークテスト。"""
from __future__ import annotations

import os

import numpy as np
import yaml

from src.utils.pf_eval_gap import gap_band_boosts, gap_bands_from_env, summarize_band_gaps
from src.utils.workload_calibration import apply_calibration_env, calibrate_from_config


def test_calibrate_trace24_no_outlier():
    root = os.path.dirname(os.path.dirname(__file__))
    cfg_path = os.path.join(root, "experiments/distributed_pcn/job_trace_24_no_outlier_pcn.yml")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    cal = calibrate_from_config(config)
    assert cal.n_jobs == 24
    assert cal.cost_all_cloud > 0
    assert cal.wait_all_onprem > cal.wait_all_cloud
    apply_calibration_env(cal)
    bands = gap_bands_from_env()
    assert len(bands) == 4
    eval_pf = np.array([[cal.cost_max * 0.2, cal.wait_all_onprem * 0.9]], dtype=np.float64)
    arch = np.array([[cal.cost_max * 0.2, cal.wait_all_onprem * 0.5]], dtype=np.float64)
    summary = summarize_band_gaps(eval_pf, arch, bands)
    boosts = gap_band_boosts(summary, bands)
    assert boosts
