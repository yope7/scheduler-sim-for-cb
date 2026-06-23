"""分散PCN: オーバーヘッド削減用ヘルパーと既定フラグのテスト。"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _reload_distributed_pcn_env(**env: str):
    """distributed_pcn を指定環境変数で再 import（モジュール定数の検証用）。"""
    for key, value in env.items():
        os.environ[key] = value
    # 未指定の診断フラグは OFF に固定してテストを安定化
    os.environ.setdefault("DISTRIBUTED_PCN_CMD_OUTCOMES", "0")
    os.environ.setdefault("DISTRIBUTED_PCN_LOG_RAY_TRANSFER", "0")
    os.environ.setdefault("DISTRIBUTED_PCN_PHASE2_IMPORTANCE", "0")
    os.environ.setdefault("DISTRIBUTED_PCN_GC_EACH_ITER", "0")
    os.environ.setdefault("DISTRIBUTED_PCN_PROFILE", "0")
    os.environ.setdefault("DISTRIBUTED_PCN_QUICK", "0")
    os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
    os.environ["DEBUG"] = "0"
    import src.distributed.distributed_pcn as dpcn

    return importlib.reload(dpcn)


def test_pareto_value_axis_limits_origin_and_includes_archive():
    from src.distributed.distributed_pcn import _pareto_value_axis_limits

    eval_pts = np.array([[10.0, 5.0], [20.0, 8.0]], dtype=np.float64)
    archive_pf = np.array([[50.0, 30.0], [5.0, 40.0]], dtype=np.float64)
    x_lo, x_hi, y_lo, y_hi = _pareto_value_axis_limits(eval_pts, archive_pf)
    assert x_lo == 0.0 and y_lo == 0.0
    assert x_hi >= 50.0
    assert y_hi >= 40.0


def test_storage_observation_skips_copy_for_float32_contiguous():
    from src.distributed.distributed_pcn import _storage_observation, _storage_reward

    obs = np.arange(10, dtype=np.float32)
    stored = _storage_observation(obs)
    assert stored.dtype == np.float32
    assert np.shares_memory(obs, stored)

    reward = np.array([-1.0, 0.5], dtype=np.float32)
    assert _storage_reward(reward).dtype == np.float32


def test_storage_observation_converts_non_float32():
    from src.distributed.distributed_pcn import _storage_observation

    obs = np.arange(5, dtype=np.float64)
    stored = _storage_observation(obs)
    assert stored.dtype == np.float32
    assert stored.shape == obs.shape


def test_replay_buffer_take_all_moves_list_without_transition_rebuild():
    import ray
    from src.agents.pcn_agent import Transition
    from src.distributed.distributed_pcn import ReplayBuffer

    episode = [
        Transition(
            observation=np.zeros(4, dtype=np.float32),
            action=0,
            reward=np.zeros(2, dtype=np.float32),
            next_observation=np.ones(4, dtype=np.float32),
            terminal=False,
        )
    ]

    ray.init(ignore_reinit_error=True, num_cpus=1, include_dashboard=False)
    try:
        actor = ReplayBuffer.remote(max_size=10)
        ray.get(actor.add_batch.remote([episode]))
        taken = ray.get(actor.take_all_episodes.remote())
        assert len(taken) == 1
        assert np.array_equal(taken[0][0].observation, episode[0].observation)
        assert ray.get(actor.size.remote()) == 0
    finally:
        ray.shutdown()


def test_default_overhead_flags_off():
    dpcn = _reload_distributed_pcn_env()
    assert dpcn._LOG_RAY_TRANSFER is False
    assert dpcn._PHASE2_IMPORTANCE is False
    assert dpcn._GC_EACH_ITER is False
    assert dpcn._COLLECT_CMD_OUTCOMES is False


def test_collect_cmd_outcomes_when_viz_light_disabled():
    dpcn = _reload_distributed_pcn_env(
        DISTRIBUTED_PCN_ENABLE_VISUALIZATION="1",
    )
    # _VIZ_LIGHT 既定 True → 矢印用収集は不要
    assert dpcn._VIZ_LIGHT is True
    assert dpcn._COLLECT_CMD_OUTCOMES is False


def test_collect_cmd_outcomes_when_full_viz():
    dpcn = _reload_distributed_pcn_env(
        DISTRIBUTED_PCN_ENABLE_VISUALIZATION="1",
        DISTRIBUTED_PCN_VIZ_LIGHT="0",
    )
    assert dpcn._COLLECT_CMD_OUTCOMES is True


@pytest.mark.skipif(
    not (REPO_ROOT / "config" / "config.yml").exists(),
    reason="config.yml required",
)
def test_actor_run_skips_command_outcomes_by_default():
    """Actor.run が既定で command_outcomes を返さないこと（Ray 要）。"""
    script = textwrap.dedent(
        """
        import os
        os.environ["DISTRIBUTED_PCN_USE_EVENT_OBS"] = "1"
        os.environ["DISTRIBUTED_PCN_CMD_OUTCOMES"] = "0"
        os.environ["DISTRIBUTED_PCN_LOG_RAY_TRANSFER"] = "0"
        os.environ["DISTRIBUTED_PCN_VIZ_LIGHT"] = "1"
        os.environ["DISTRIBUTED_PCN_ENABLE_VISUALIZATION"] = "1"
        import ray
        from src.distributed.distributed_pcn import Actor, Learner, ReplayBuffer
        import yaml

        with open("config/config.yml") as f:
            config = yaml.safe_load(f)
        config["param_env"]["n_jobs"] = 4

        ray.init(ignore_reinit_error=True, num_cpus=2, num_gpus=0, include_dashboard=False)
        buffer = ReplayBuffer.remote(max_size=100)
        LearnerActor = ray.remote(Learner)
        learner = LearnerActor.remote(config, buffer, device="cpu")
        actor = Actor.remote(config, learner, buffer, actor_id=0)
        ray.get(actor._make_env.remote())
        result = ray.get(actor.run.remote(n_episodes=1, random_actions=True))
        assert "command_outcomes" not in result
        ray.shutdown()
        """
    )
    python = REPO_ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        python = Path(sys.executable)
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    proc = subprocess.run(
        [str(python), "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
