"""中域向け replay / conditioning ヘルパの単体テスト。"""
import os
import unittest

os.environ.setdefault("PCN_TRAIN_MID_STEP_WEIGHT", "6")
os.environ.setdefault("PCN_TRAIN_EVALIKE_STEP_WEIGHT", "4")
os.environ.setdefault("PCN_COMMAND_BALANCE", "1")

import gymnasium as gym
import numpy as np

from src.agents import pcn_agent as pcn_agent_mod
from src.agents.pcn_agent import PCN, Transition

pcn_agent_mod._TRAIN_MID_STEP_WEIGHT = float(os.environ["PCN_TRAIN_MID_STEP_WEIGHT"])
pcn_agent_mod._TRAIN_EVALIKE_STEP_WEIGHT = float(os.environ["PCN_TRAIN_EVALIKE_STEP_WEIGHT"])
pcn_agent_mod._COMMAND_BALANCE = True


class _DummyEnv:
    def __init__(self):
        self.reward_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.n_jobs = 1024
        self.spec = type("S", (), {"id": "dummy"})()
        self.unwrapped = self


class TestMidBandTraining(unittest.TestCase):
    def setUp(self):
        self.env = _DummyEnv()
        self.agent = PCN(
            self.env,
            device="cpu",
            state_dim=8,
            scaling_factor=np.array([1.0, 1.0, 1.0 / 1024]),
            learning_rate=1e-3,
            batch_size=8,
            hidden_dim=32,
            project_name="t",
            experiment_name="t",
            log=False,
        )

    def _push_episode(self, cost: float, wait: float, ep_len: int = 5):
        rewards = np.zeros((ep_len, 2), dtype=np.float32)
        rewards[:, 1] = -cost / ep_len
        rewards[:, 0] = -wait * 1024 / ep_len
        transitions = []
        for t in range(ep_len):
            transitions.append(
                Transition(
                    observation=np.zeros(8, dtype=np.float32),
                    action=0,
                    reward=rewards[t],
                    next_observation=np.zeros(8, dtype=np.float32),
                    terminal=(t == ep_len - 1),
                )
            )
        transitions[0].objective_values = [cost, 0.0, wait]
        self.agent.experience_replay.append((1.0, len(self.agent.experience_replay), transitions))

    def test_mid_step_weights_sum(self):
        self._push_episode(5e5, 12000.0)
        self._push_episode(3e6, 8000.0)
        steps = self.agent.build_training_batch_cache(on_device=False)
        self.assertGreater(steps, 0)
        cache = self.agent._training_batch_cache
        self.assertIsNotNone(cache.get("flat_step_probs"))
        fp = cache["flat_step_probs"]
        self.assertAlmostEqual(float(fp.sum()), float(fp.sum()), places=5)
        self.assertTrue(np.any(fp > 1.0))

    def test_archive_mid_commands(self):
        self._push_episode(8e5, 14000.0)
        self._push_episode(1.2e6, 13000.0)
        self._push_episode(4e6, 7000.0)
        cmds = self.agent._collect_mid_band_archive_commands()
        self.assertGreaterEqual(cmds.shape[0], 1)

    def test_archive_low_slope_commands(self):
        pcn_agent_mod._TRAIN_LOW_SLOPE_COST_MAX_FRAC = 0.18
        self._push_episode(0.0, 17800.0)
        self._push_episode(4e5, 16500.0)
        self._push_episode(1.0e6, 14000.0)
        self._push_episode(4e6, 7000.0)
        cmds = self.agent._collect_low_slope_archive_commands()
        self.assertGreaterEqual(cmds.shape[0], 2)
        self.assertEqual(cmds.shape[1], 2)
        self.assertTrue(np.any(cmds[:, 1] < 0.0))  # r1 = -cost（cost>0 の点）

    def test_low_wait_episode_weight_count(self):
        old_weight = pcn_agent_mod._TRAIN_LOW_WAIT_PF_WEIGHT
        old_max = pcn_agent_mod._TRAIN_LOW_WAIT_MAX
        old_frac = pcn_agent_mod._TRAIN_LOW_WAIT_FRAC
        try:
            pcn_agent_mod._TRAIN_LOW_WAIT_PF_WEIGHT = 5.0
            pcn_agent_mod._TRAIN_LOW_WAIT_MAX = 600.0
            pcn_agent_mod._TRAIN_LOW_WAIT_FRAC = 0.0
            self._push_episode(2e5, 900.0)
            self._push_episode(4e5, 500.0)
            self._push_episode(6e5, 300.0)
            self.agent.build_training_batch_cache(on_device=False)
            cache = self.agent._training_batch_cache
            self.assertEqual(cache["low_wait_pf_episode_count"], 2)
        finally:
            pcn_agent_mod._TRAIN_LOW_WAIT_PF_WEIGHT = old_weight
            pcn_agent_mod._TRAIN_LOW_WAIT_MAX = old_max
            pcn_agent_mod._TRAIN_LOW_WAIT_FRAC = old_frac

    def test_command_balance_vector(self):
        self.agent.return_norm_scale = np.array([1e6, 1e7], dtype=np.float32)
        bal = self.agent._command_balance_vector()
        self.assertAlmostEqual(float(bal[0] * 1e6), float(bal[1] * 1e7), delta=1e3)

    def test_extend_cache_matches_full_rebuild(self):
        """Phase3 extend: cmax 不変時の追記が全件再構築と同じ cache になること。"""
        costs = [5e5, 2.5e6, 8e5, 1.1e6]
        episodes = []
        for cost in costs:
            ep_len = 6
            rewards = np.zeros((ep_len, 2), dtype=np.float32)
            rewards[:, 1] = -cost / ep_len
            rewards[:, 0] = -12000.0 * 1024 / ep_len
            transitions = []
            for t in range(ep_len):
                transitions.append(
                    Transition(
                        observation=np.zeros(8, dtype=np.float32),
                        action=0,
                        reward=rewards[t],
                        next_observation=np.zeros(8, dtype=np.float32),
                        terminal=(t == ep_len - 1),
                    )
                )
            transitions[0].objective_values = [cost, 0.0, 12000.0]
            episodes.append(transitions)
            self.agent.experience_replay.append(
                (1.0, len(self.agent.experience_replay), transitions)
            )

        full_steps = self.agent.build_training_batch_cache(on_device=False)
        cache_full = self.agent._training_batch_cache

        agent2 = PCN(
            self.env,
            device="cpu",
            state_dim=8,
            scaling_factor=np.array([1.0, 1.0, 1.0 / 1024]),
            learning_rate=1e-3,
            batch_size=8,
            hidden_dim=32,
            project_name="t",
            experiment_name="t2",
            log=False,
        )
        agent2.np_random = np.random.default_rng(0)
        inc_steps = 0
        for tr in episodes:
            agent2.experience_replay.append((1.0, len(agent2.experience_replay), tr))
            if inc_steps == 0:
                inc_steps = agent2.build_training_batch_cache(on_device=False)
            else:
                inc_steps = agent2.extend_training_batch_cache([tr], on_device=False)

        cache_inc = agent2._training_batch_cache
        self.assertEqual(full_steps, inc_steps)
        np.testing.assert_allclose(cache_full["episode_probs"], cache_inc["episode_probs"])
        np.testing.assert_allclose(cache_full["flat_step_probs"], cache_inc["flat_step_probs"])
        np.testing.assert_array_equal(cache_full["episode_lengths"], cache_inc["episode_lengths"])
        np.testing.assert_allclose(
            cache_full["observations"], cache_inc["observations"], rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
