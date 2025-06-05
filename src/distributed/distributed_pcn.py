import ray
import numpy as np
import torch as th
import yaml
from tqdm import tqdm
import time

from src.agents.pcn_agent import PCN, Transition
from src.envs.scheduling_env import SchedulingEnv
from src.utils.job_gen.job_generator import JobGenerator

# =========================
# 1. Replay Buffer (Ray Actor)
# =========================
@ray.remote
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size

    def add(self, episode):
        print(f"ReplayBuffer: add called, episode len={len(episode)}")
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(episode)

    def sample(self, batch_size):
        import random
        if len(self.buffer) < batch_size:
            return []
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# =========================
# 2. Actor (Ray Actor)
# =========================
@ray.remote
class Actor:
    def __init__(self, config, learner, buffer, actor_id=0):
        self.config = config
        self.env = self._make_env()
        self.agent = PCN(
            self.env,
            device="cpu",
            state_dim=1,
            scaling_factor=np.array([1, 1, 1]),
            learning_rate=1e-2,
            batch_size=1024,
            hidden_dim=256,
            project_name="temp",
            experiment_name="PCN",
            log=False,
        )
        self.learner = learner
        self.buffer = buffer
        self.actor_id = actor_id

    def _make_env(self):
        job_generator = JobGenerator(
            0, 1,
            self.config['param_env']['n_window'],
            self.config['param_env']['n_on_premise_node'],
            self.config['param_env']['n_cloud_node'],
            self.config, 100, 0.23, 1
        )
        jobs_set = job_generator.generate_jobs_set()
        env = SchedulingEnv(
            np.inf,
            self.config['param_env']['n_window'],
            self.config['param_env']['n_on_premise_node'],
            self.config['param_env']['n_cloud_node'],
            self.config['param_env']['n_job_queue_obs'],
            self.config['param_env']['n_job_queue_bck'],
            self.config['param_agent']['weight_wt'],
            self.config['param_agent']['weight_cost'],
            self.config['param_env']['penalty_not_allocate'],
            self.config['param_env']['penalty_invalid_action'],
            jobs_set,
            None, flag=0
        )
        return env

    def run(self, n_episodes=10):
        for ep in range(n_episodes):
            # 最新重みをLearnerからpull
            weights = ray.get(self.learner.get_weights.remote())
            self.agent.model.load_state_dict(weights)
            # 1エピソード実行
            episode = self._run_episode()
            # 経験をReplayBufferにput
            print(f"[Actor {self.actor_id}] pushing episode of len={len(episode)}")
            self.buffer.add.remote(episode)
            if ep % 10 == 0:
                print(f"[Actor {self.actor_id}] {ep} episodes generated.")

    def _run_episode(self):
        obs = self.env.reset()
        done = False
        transitions = []
        # 目標報酬・ホライズンは適宜設定
        self.agent.set_desired_return_and_horizon(
            desired_return=np.zeros(2, dtype=np.float32),
            desired_horizon=10
        )
        while not done:
            action = self.agent.eval(obs)
            n_obs, reward, scheduled, wt_step, done = self.env.step(action)
            transitions.append(Transition(obs, action, np.float32(reward).copy(), n_obs, done))
            obs = n_obs
        return transitions

# =========================
# 3. Learner (Ray Actor)
# =========================
@ray.remote
class Learner:
    def __init__(self, config, buffer, device='cuda'):
        self.config = config
        self.env = self._make_env()
        self.agent = PCN(
            self.env,
            device=device,
            state_dim=1,
            scaling_factor=np.array([1, 1, 1]),
            learning_rate=1e-2,
            batch_size=1024,
            hidden_dim=256,
            project_name="temp",
            experiment_name="PCN",
            log=False,
        )
        self.buffer = buffer

    def _make_env(self):
        job_generator = JobGenerator(
            0, 1,
            self.config['param_env']['n_window'],
            self.config['param_env']['n_on_premise_node'],
            self.config['param_env']['n_cloud_node'],
            self.config, 100, 0.23, 1
        )
        jobs_set = job_generator.generate_jobs_set()
        env = SchedulingEnv(
            np.inf,
            self.config['param_env']['n_window'],
            self.config['param_env']['n_on_premise_node'],
            self.config['param_env']['n_cloud_node'],
            self.config['param_env']['n_job_queue_obs'],
            self.config['param_env']['n_job_queue_bck'],
            self.config['param_agent']['weight_wt'],
            self.config['param_agent']['weight_cost'],
            self.config['param_env']['penalty_not_allocate'],
            self.config['param_env']['penalty_invalid_action'],
            jobs_set,
            None, flag=0
        )
        return env

    def get_weights(self):
        return self.agent.model.state_dict()

    def learn(self, batch_size=32, n_updates=1000):
        for i in range(n_updates):
            batch = ray.get(self.buffer.sample.remote(batch_size))
            if not batch:
                continue
            self._update(batch)
            if i % 10 == 0:
                print(f"[Learner] {i} updates done. Buffer size: {ray.get(self.buffer.size.remote())}")

    def _update(self, batch):
        # batch: List[List[Transition]]
        # flatten
        transitions = [item for episode in batch for item in episode]
        if len(transitions) == 0:
            return
        # 必要なデータを抽出
        obs = np.stack([t.observation for t in transitions])
        actions = np.array([t.action for t in transitions])
        rewards = np.stack([t.reward for t in transitions])
        next_obs = np.stack([t.next_observation for t in transitions])
        terminals = np.array([t.terminal for t in transitions])

        # PCNのupdateを呼ぶ
        # ここでは既存のupdate()を使う（バッチサイズはPCNのself.batch_sizeで制御）
        # 経験リプレイバッファに追加
        # ここではバッチからエピソードを再構成して追加
        # 1エピソード=done=Trueで区切る
        episode = []
        for t in transitions:
            episode.append(t)
            if t.terminal:
                self.agent._add_episode(episode, max_size=10000, step=0)
                episode = []
        # 学習
        self.agent.update()

# =========================
# 4. 実行スクリプト
# =========================
def main():
    # 設定ファイルの読み込み
    with open('config/config.yml', 'r') as yml:
        config = yaml.safe_load(yml)

    ray.init()

    # Replay Buffer
    buffer = ReplayBuffer.remote(max_size=10000)

    # Learner
    learner = Learner.remote(config, buffer, device='cuda')

    # Actor群
    n_actors = 4
    actors = [Actor.remote(config, learner, buffer, actor_id=i) for i in range(n_actors)]

    # Actorでエピソード生成を並列実行
    ray.get([actor.run.remote(n_episodes=10) for actor in actors])

    # Learnerで学習を開始
    learner.learn.remote(batch_size=128, n_updates=1000000)

    print("Distributed PCN training started.")

    for _ in range(100):
        size = ray.get(buffer.size.remote())
        print(f"ReplayBuffer size: {size}")
        time.sleep(5)

if __name__ == "__main__":
    main()
