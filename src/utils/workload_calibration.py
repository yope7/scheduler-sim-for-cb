"""ジョブセットから PCN 学習・Eval ギャップ FB 用スケールを推定する（ワークロード非依存）。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.envs.scheduling_variants.event_native_env import SchedulingEnvEventNative
from src.utils.job_gen.job_generator import JobGenerator


@dataclass(frozen=True)
class WorkloadCalibration:
    n_jobs: int
    cost_all_cloud: float
    cost_all_onprem: float
    wait_all_onprem: float
    wait_all_cloud: float
    cost_mid_hi: float

    @property
    def cost_max(self) -> float:
        return max(self.cost_all_cloud, 1.0)

    @property
    def wait_max(self) -> float:
        return max(self.wait_all_onprem, self.wait_all_cloud, 1.0)


def _build_jobs_set(config: dict, n_jobs: int) -> list:
    cfg = yaml.safe_load(yaml.dump(config))
    cfg["param_env"]["n_jobs"] = int(n_jobs)
    jg = JobGenerator(
        0,
        cfg["param_simulation"]["nb_steps"],
        cfg["param_env"]["n_window"],
        cfg["param_env"]["n_on_premise_node"],
        cfg["param_env"]["n_cloud_node"],
        cfg,
        int(n_jobs),
        0.2,
        1,
    )
    return jg.generate_jobs_set()


def _run_fixed_assignment(
    config: dict,
    n_jobs: int,
    actions: Tuple[int, ...],
) -> Tuple[float, float]:
    jobs_set = _build_jobs_set(config, n_jobs)
    pe = config["param_env"]
    pa = config["param_agent"]
    env = SchedulingEnvEventNative(
        np.inf,
        pe["n_window"],
        pe["n_on_premise_node"],
        pe["n_cloud_node"],
        pe["n_job_queue_obs"],
        pe["n_job_queue_bck"],
        pa["weight_wt"],
        pa["weight_cost"],
        pe["penalty_not_allocate"],
        pe["penalty_invalid_action"],
        jobs_set,
        None,
        0,
    )
    env.reset()
    done = False
    step = 0
    # 固定割当（全 OP=0 / 全 CL=1）は定数アクション。ジョブがノード不足でキュー
    # 待ちになるとエピソードのステップ数が n_jobs を超えうるため、index を
    # クランプして同じアクションを出し続ける（1024 ジョブ等のスケールで必須）。
    n_actions = len(actions)
    fixed_action = actions[-1] if n_actions else 0
    max_steps = max(n_actions, 1) * 100  # 病的な非終了に対する安全弁
    while not done:
        a = actions[step] if step < n_actions else fixed_action
        _, _, _, _, done = env.step(a)
        step += 1
        if step >= max_steps:
            break
    env.finalize_window_history(build_maps=False)
    cost, _, avg_wt = env.calc_objective_values()
    return float(cost), float(avg_wt)


def calibrate_from_config(config: dict, n_jobs: Optional[int] = None) -> WorkloadCalibration:
    """全 OP / 全 CL の2点から cost・wait スケールを推定。"""
    nj = int(
        n_jobs
        if n_jobs is not None
        else config.get("param_env", {}).get("n_jobs")
        or config.get("param_algorithm_compare", {}).get("nb_jobs", 24)
    )
    all0 = tuple([0] * nj)
    all1 = tuple([1] * nj)
    cost_op, wait_op = _run_fixed_assignment(config, nj, all0)
    cost_cl, wait_cl = _run_fixed_assignment(config, nj, all1)
    return WorkloadCalibration(
        n_jobs=nj,
        cost_all_cloud=float(cost_cl),
        cost_all_onprem=float(cost_op),
        wait_all_onprem=float(wait_op),
        wait_all_cloud=float(wait_cl),
        cost_mid_hi=float(max(cost_cl, cost_op) * 0.65),
    )


def apply_calibration_env(cal: WorkloadCalibration) -> Dict[str, str]:
    """推定値を環境変数へ反映（既存値は上書きしない setdefault ではなく明示 set）。"""
    gap_ref = max(30.0, cal.wait_max * 0.08)
    env_updates = {
        "PCN_WORKLOAD_COST_MAX": f"{cal.cost_max:.6g}",
        "PCN_WORKLOAD_WAIT_MAX": f"{cal.wait_max:.6g}",
        # 全OP / 全CL の両端点を明示露出（anchored command pool 用）。
        "PCN_WORKLOAD_COST_OP": f"{cal.cost_all_onprem:.6g}",
        "PCN_WORKLOAD_COST_CL": f"{cal.cost_all_cloud:.6g}",
        "PCN_WORKLOAD_WAIT_OP": f"{cal.wait_all_onprem:.6g}",
        "PCN_WORKLOAD_WAIT_CL": f"{cal.wait_all_cloud:.6g}",
        "PCN_VALUE_COST_SCALE": f"{cal.cost_max:.6g}",
        "PCN_VALUE_WAIT_SCALE": f"{max(cal.wait_max * cal.n_jobs, 100.0):.6g}",
        "PCN_EVAL_GAP_REF_GAP": f"{gap_ref:.6g}",
        "PCN_EVAL_GAP_REF_FRAC": "0.08",
        "PCN_EVAL_GAP_BANDS_FRAC": "0,0.35,0.35,0.65,0.65,0.85,0.85,1.0",
        "PCN_EVAL_GAP_BAND_NAMES": "low,low_slope,mid,knee",
        "PCN_EVAL_GAP_BOOST_MAX": "3.0",
        "PCN_TRAIN_MID_COST_MIN_FRAC": "0.15",
        "PCN_TRAIN_MID_COST_MAX_FRAC": "0.75",
    }
    for k, v in env_updates.items():
        os.environ[k] = v
    return env_updates


def format_calibration_log(cal: WorkloadCalibration, env_updates: Dict[str, str]) -> str:
    lines = [
        "[WORKLOAD_CALIB] "
        f"n_jobs={cal.n_jobs} cost(op/cl)={cal.cost_all_onprem:.0f}/{cal.cost_all_cloud:.0f} "
        f"wait(op/cl)={cal.wait_all_onprem:.1f}/{cal.wait_all_cloud:.1f}",
        f"[WORKLOAD_CALIB] gap_ref={env_updates['PCN_EVAL_GAP_REF_GAP']} "
        f"value_cost_scale={env_updates['PCN_VALUE_COST_SCALE']}",
    ]
    return "\n".join(lines)
