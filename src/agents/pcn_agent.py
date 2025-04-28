"""Pareto Conditioned Network. Code adapted from https://github.com/mathieu-reymond/pareto-conditioned-networks ."""
import heapq
import os
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Type, Union
from matplotlib.animation import FuncAnimation

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import wandb
# wandb.init(project="temp")

np.random.seed(42)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from morl_baselines.common.evaluation import log_all_multi_policy_metrics
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.pareto import get_non_dominated_inds
from morl_baselines.common.performance_indicators import hypervolume
from src.utils.map_visualizer import visualize_map


def crowding_distance(points):
    """Compute the crowding distance of a set of points."""
    # first normalize across dimensions
    points = (points - points.min(axis=0)) / (np.ptp(points, axis=0) + 1e-8)
    # sort points per dimension
    dim_sorted = np.argsort(points, axis=0)
    point_sorted = np.take_along_axis(points, dim_sorted, axis=0)
    # compute distances between lower and higher point
    distances = np.abs(point_sorted[:-2] - point_sorted[2:])
    # pad extrema's with 1, for each dimension
    distances = np.pad(distances, ((1,), (0,)), constant_values=1)
    # sum distances of each dimension of the same point
    crowding = np.zeros(points.shape)
    crowding[dim_sorted, np.arange(points.shape[-1])] = distances
    crowding = np.sum(crowding, axis=-1)
    return crowding


@dataclass
class Transition:
    """Transition dataclass."""

    observation: np.ndarray
    action: Union[int, int]
    reward: np.ndarray
    next_observation: np.ndarray
    terminal: bool

class BasePCNModel(nn.Module, ABC):
    """Base Model for the PCN."""

    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, scaling_factor: np.ndarray, hidden_dim: int):
        """Initialize the PCN model."""
        super().__init__()
        self.state_dim = state_dim
        # print("state_dim", self.state_dim)
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.scaling_factor = nn.Parameter(th.tensor(scaling_factor).float(), requires_grad=False)
        self.hidden_dim = hidden_dim



        # self.scaling_factor = scaling_factor
        # self.s_emb = nn.Linear(state_dim, hidden_dim)
        # self.c_emb = nn.Linear(action_dim + reward_dim, hidden_dim)
        # self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, desired_return, desired_horizon):
        """Return log-probabilities of actions or return action directly in case of continuous action space."""
        c = th.cat((desired_return, desired_horizon), dim=-1)
        c = c * self.scaling_factor

        # 並列計算の活用
        with th.cuda.amp.autocast():
            s = self.s_emb(state.float())
            c = self.c_emb(c)
            # 行列乗算を最適化
            prediction = self.fc(s * c)
        # print("prediction.shape", prediction.shape)
        return prediction


class DiscreteActionsDefaultModel(BasePCNModel):
    """Model for the PCN with discrete actions."""

    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, scaling_factor: np.ndarray, hidden_dim: int):
        """Initialize the PCN model for discrete actions."""
        super().__init__(state_dim, action_dim, reward_dim, scaling_factor, hidden_dim)
        # print("kotti")
        self.state_dim = 12790
        self.s_emb = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim), nn.Sigmoid())
        self.c_emb = nn.Sequential(nn.Linear(self.reward_dim + 1, self.hidden_dim), nn.Sigmoid())
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.LogSoftmax(dim=1),
        )


class ContinuousActionsDefaultModel(BasePCNModel):
    """Model for the PCN with continuous actions."""

    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, scaling_factor: np.ndarray, hidden_dim: int):
        """Initialize the PCN model for continuous actions."""
        super().__init__(state_dim, action_dim, reward_dim, scaling_factor, hidden_dim)
        self.s_emb = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim), nn.Sigmoid())
        self.c_emb = nn.Sequential(nn.Linear(self.reward_dim + 1, self.hidden_dim), nn.Sigmoid())
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )


class PCN(MOAgent, MOPolicy):
    """Pareto Conditioned Networks (PCN).

    Reymond, M., Bargiacchi, E., & Nowé, A. (2022, May). Pareto Conditioned Networks.
    In Proceedings of the 21st International Conference on Autonomous Agents
    and Multiagent Systems (pp. 1110-1118).
    https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1110.pdf

    ## Credits

    This code is a refactor of the code from the authors of the paper, available at:
    https://github.com/mathieu-reymond/pareto-conditioned-networks
    """

    def __init__(
        self,
        env: Optional[gym.Env],
        scaling_factor: np.ndarray,
        device: Union[th.device, str],
        
        learning_rate: float,
        state_dim: int = 1,
        gamma: float = 1.0,
        batch_size: int = 1024,
        hidden_dim: int = 64,

        noise: float = 0.1,
        project_name: str = "temp",
        experiment_name: str = "PCN",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        seed: Optional[int] = None,

        model_class: Optional[Type[BasePCNModel]] = None,
    ) -> None:
        """Initialize PCN agent.

        Args:
            env (Optional[gym.Env]): Gym environment.
            scaling_factor (np.ndarray): Scaling factor for the desired return and horizon used in the model.
            learning_rate (float, optional): Learning rate. Defaults to 1e-2.
            gamma (float, optional): Discount factor. Defaults to 1.0.
            batch_size (int, optional): Batch size. Defaults to 32.
            hidden_dim (int, optional): Hidden dimension. Defaults to 64.
            noise (float, optional): Standard deviation of the noise to add to the action in the continuous action case. Defaults to 0.1.
            project_name (str, optional): Name of the project for wandb. Defaults to "MORL-Baselines".
            experiment_name (str, optional): Name of the experiment for wandb. Defaults to "PCN".
            wandb_entity (Optional[str], optional): Entity for wandb. Defaults to None.
            log (bool, optional): Whether to log to wandb. Defaults to True.
            seed (Optional[int], optional): Seed for reproducibility. Defaults to None.
            device (Union[th.device, str], optional): Device to use. Defaults to "auto".
            model_class (Optional[Type[BasePCNModel]], optional): Model class to use. Defaults to None.
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device)

        self.reward_dim = env.reward_space.shape[0]

        self.experience_replay = []  # List of (distance, time_step, transition)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.scaling_factor = scaling_factor
        # self.observation_dim = observation_dim

        self.continuous_action = True if type(self.env.action_space) is gym.spaces.Box else False
        self.noise = noise
        self.e_returns = []
        self.transitions = []
        self.mapmap = []
        

        if model_class and not issubclass(model_class, BasePCNModel):
            raise ValueError("model_class must be a subclass of BasePCNModel")

        if model_class is None:
            if self.continuous_action:
                model_class = ContinuousActionsDefaultModel
            else:
                model_class = DiscreteActionsDefaultModel

        # print("observation_dim", self.env.observation_space.shape[0])

        self.model = model_class(
            self.observation_dim, self.action_dim, self.reward_dim, self.scaling_factor, hidden_dim=self.hidden_dim
        ).to(self.device, non_blocking=True)
        self.opt = th.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.log = log
        if log:
            experiment_name += " continuous action" if self.continuous_action else ""
            self.setup_wandb(project_name, experiment_name, wandb_entity)

        # 評価結果を蓄積するための変数を追加
        self.evaluation_history = []
        self.evaluation_timestamps = []
        self.global_steps_at_evaluation = []

    def get_config(self) -> dict:
        """Get configuration of PCN model."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "hidden_dim": self.hidden_dim,
            "scaling_factor": self.scaling_factor,
            "continuous_action": self.continuous_action,
            "noise": self.noise,
            "seed": self.seed,
        }

    def update(self):
        """Update PCN model."""
        with th.cuda.amp.autocast():
            # experience_replayのインデックスを一括サンプリング
            sample_indices = self.np_random.choice(len(self.experience_replay), size=self.batch_size, replace=True)
            
            observations_list = []
            actions_list = []
            desired_return_list = []
            desired_horizon_list = []
            
            for i in sample_indices:
                # エピソードの取得
                episode = self.experience_replay[i][2]
                episode_length = len(episode)
                
                # エピソード内のランダムな時刻をサンプリング
                t = self.np_random.integers(0, episode_length)
                
                # 対象のTransitionを取得
                transition = episode[t]
                
                # 残りステップ数の計算（float32にキャスト）
                rest_horizon = np.float32(episode_length - t)
                
                observations_list.append(transition.observation)
                actions_list.append(transition.action)
                desired_return_list.append(np.float32(transition.reward))
                desired_horizon_list.append(rest_horizon)
            
            # それぞれのリストをnp.stackでまとめ、torch.from_numpyを用いてGPUへ転送
            obs = th.from_numpy(np.stack(observations_list)).to(self.device)
            actions = th.from_numpy(np.stack(actions_list)).to(self.device)
            desired_return = th.from_numpy(np.stack(desired_return_list)).to(self.device)
            desired_horizon = th.from_numpy(np.stack(desired_horizon_list)).to(self.device)
            
            # 推論と損失計算を単一のGPUストリームで実行
            prediction = self.model(obs, desired_return, desired_horizon.unsqueeze(1))
            
            self.opt.zero_grad(set_to_none=True)  # メモリ効率の改善
            
            if self.continuous_action:
                l = F.mse_loss(actions.float(), prediction)
            else:
                actions = F.one_hot(actions.long(), len(prediction[0]))
                l = th.sum(-actions * prediction, -1).mean()
            
            l.backward()
            self.opt.step()
            
            return l, prediction

    def _add_episode(self, transitions: List[Transition], max_size: int, step: int) -> None:
        # compute return
        for i in reversed(range(len(transitions) - 1)):
            transitions[i].reward += self.gamma * transitions[i + 1].reward
        # pop smallest episode of heap if full, add new episode
        # heap is sorted by negative distance, (updated in nlargest)
        # put positive number to ensure that new item stays in the heap
        if len(self.experience_replay) == max_size:
            heapq.heappushpop(self.experience_replay, (1, step, transitions))
        else:
            heapq.heappush(self.experience_replay, (1, step, transitions))

    def _nlargest(self, n, threshold=0.2):
        """See Section 4.4 of https://arxiv.org/pdf/2204.05036.pdf for details."""
        returns = np.array([e[2][0].reward for e in self.experience_replay])
        # crowding distance of each point, check ones that are too close together
        distances = crowding_distance(returns)
        sma = np.argwhere(distances <= threshold).flatten()

        non_dominated_i = get_non_dominated_inds(returns)
        non_dominated = returns[non_dominated_i]
        # we will compute distance of each point with each non-dominated point,
        # duplicate each point with number of non_dominated to compute respective distance
        returns_exp = np.tile(np.expand_dims(returns, 1), (1, len(non_dominated), 1))
        # distance to closest non_dominated point
        l2 = np.min(np.linalg.norm(returns_exp - non_dominated, axis=-1), axis=-1) * -1
        # all points that are too close together (crowding distance < threshold) get a penalty
        non_dominated_i = np.nonzero(non_dominated_i)[0]
        _, unique_i = np.unique(non_dominated, axis=0, return_index=True)
        unique_i = non_dominated_i[unique_i]
        duplicates = np.ones(len(l2), dtype=bool)
        duplicates[unique_i] = False
        l2[duplicates] -= 1e-5
        l2[sma] *= 2

        sorted_i = np.argsort(l2)
        largest = [self.experience_replay[i] for i in sorted_i[-n:]]
        # before returning largest elements, update all distances in heap
        for i in range(len(l2)):
            self.experience_replay[i] = (l2[i], self.experience_replay[i][1], self.experience_replay[i][2])
        heapq.heapify(self.experience_replay)
        return largest

    def _choose_commands(self, num_episodes: int):
        # get best episodes, according to their crowding distance
        episodes = self._nlargest(num_episodes)
        returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
        # keep only non-dominated returns
        nd_i = get_non_dominated_inds(np.array(returns))
        returns = np.array(returns)[nd_i]
        horizons = np.array(horizons)[nd_i]
        # pick random return from random best episode
        r_i = self.np_random.integers(0, len(returns))
        desired_horizon = np.float32(horizons[r_i] - 2)
        # mean and std per objective
        _, s = np.mean(returns, axis=0), np.std(returns, axis=0)
        # desired return is sampled from [M, M+S], to try to do better than mean return
        desired_return = returns[r_i].copy()
        # random objective
        r_i = self.np_random.integers(0, len(desired_return))
        desired_return[r_i] += self.np_random.uniform(high=s[r_i])
        # 改善の余地あり
        desired_return = np.float32(desired_return)
        return desired_return, desired_horizon

    def _act(self, obs: np.ndarray, desired_return, desired_horizon, eval_mode=False) -> int:
        with th.cuda.amp.autocast():
            obs_tensor = th.tensor(np.array([obs]), device=self.device).float()
            return_tensor = th.tensor(np.array([desired_return]), device=self.device).float()
            # desired_horizonをスカラー値として扱うように修正
            horizon_tensor = th.tensor([[desired_horizon]], device=self.device).float()
            
            # 推論を実行
            prediction = self.model(obs_tensor, return_tensor, horizon_tensor)

        if self.continuous_action:
            action = prediction.detach()[0]
            if not eval_mode:
                # Add Gaussian noise: https://arxiv.org/pdf/2204.05027.pdf
                action = action + th.normal(0.0, self.noise, size=action.shape, device=self.device)
            return action.cpu().numpy()
        else:
            log_probs = prediction.detach()[0]
            if eval_mode:
                action = th.argmax(log_probs).item()
            else:
                probs = th.exp(log_probs)
                action = th.multinomial(probs, 1)[0].item()
            return action

    def _run_episode(self, env, desired_return, desired_horizon, max_return, eval_mode=False):
        transitions = []
        map_snapshots_on_premise = []  # 各stepのオンプレミス配置マップ（2次元リスト）の記録
        map_snapshots_cloud = []       # 各stepのクラウド配置マップ（2次元リスト）の記録
        obs = env.reset()
        done = False
        wt_sum = 0
        while not done:
            # エージェントの行動選択（既存の実装に従う）
            action = self._act(obs, desired_return, desired_horizon, eval_mode)
            n_obs, reward, scheduled, wt_step,done = env.step(action)
            if done:
                env.finalize_window_history()
            # 現在の配置マップを記録
            # ここでは各ウィンドウの "job_id" 部分を使用して数値行列にしています
            on_pre_map = env.on_premise_window["job_id"].tolist()
            cloud_map = env.cloud_window["job_id"].tolist()
            map_snapshots_on_premise.append(on_pre_map)
            map_snapshots_cloud.append(cloud_map)
            # Transition の記録（既存の構造体を利用）
            transitions.append(
                Transition(
                    observation=obs,
                    action=action,
                    reward=np.float32(reward).copy(),
                    next_observation=n_obs,
                    terminal=done,
                )
            )
            obs = n_obs
            wt_sum += wt_step
            desired_return = np.clip(desired_return - reward, None, max_return, dtype=np.float32)
            desired_horizon = np.float32(max(desired_horizon - 1, 1.0))
        onpre_final = env.on_premise_window_history_full
        cloud_final = env.cloud_window_history_full
        value_cost, value_wt = env.calc_objective_values()
        return transitions, map_snapshots_on_premise, map_snapshots_cloud, wt_sum, [onpre_final, cloud_final], [value_cost, value_wt]

    def set_desired_return_and_horizon(self, desired_return: np.ndarray, desired_horizon: int):
        """Set desired return and horizon for evaluation."""
        self.desired_return = desired_return
        self.desired_horizon = desired_horizon

    def eval(self, obs, w=None):
        """Evaluate policy action for a given observation."""
        return self._act(obs, self.desired_return, self.desired_horizon, eval_mode=True)
    
    def select_policy_by_certain_objective(self, e_returns, objective_index):
        """特定の目的関数の値が最大となるようなポリシーを選択する"""
        best_policy_index = np.argmax(np.array([e[objective_index] for e in e_returns]))
        return e_returns[best_policy_index]
    
    def execute_selected_policy(self, env, best_policy):
        """選択されたポリシーを実行する"""
        self.run_episode(env, best_policy, max_return=np.full(2, 100.0, dtype=np.float32), eval_mode=True)
    
    def evaluate_and_execute_selected_policy(self, env, max_return, objective_index, n=10):
        """特定の目的関数の値が最大となるようなポリシーを評価して実行する"""
        n = min(n, len(self.experience_replay))
        # print("len(self.experience_replay)", len(self.experience_replay))
        episodes = self._nlargest(n)
        returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
        returns = np.float32(returns)
        horizons = np.float32(horizons)
        all_transitions = []
        e_returns = []
        for i in range(n):
            transitions, _, _, _, _ , _, value = self._run_episode(env, returns[i], np.float32(horizons[i]), max_return, eval_mode=True)
            all_transitions.append(transitions)
            # compute return
            for j in reversed(range(len(transitions) - 1)):
                transitions[j].reward += self.gamma * transitions[j + 1].reward
            e_returns.append(transitions[0].reward)
            #やってみて、再現可能なデータを集める。

        # 非支配解の取得
        # print("e_returns", e_returns)
        non_dominated_inds = get_non_dominated_inds(np.array(e_returns))
        pareto_front = np.array(e_returns)[non_dominated_inds]
        if self.log:
            wandb.log({"pareto_front_eval_and_execute": wandb.Table(data=pareto_front, columns=["Objective1", "Objective2"])})


        best_policy_index = np.argmax([e[objective_index] for e in e_returns])
        if objective_index == 9:
            "並び替えて，真ん中にあるものを選択する．"
            best_policy_index = len(e_returns) // 2
        # print("best_policy_index", best_policy_index)
        best_transitions = all_transitions[best_policy_index]
        # print("best_transitions", best_transitions[0].action)

        #execute best_transitions
        obs = env.reset()
        done = False
        step = 0
        wt_sum = 0
        culmulative_reward = np.zeros(self.reward_dim)
        while not done and step < len(best_transitions):
            action = best_transitions[step].action
            n_obs, reward, _, wt_step,_,done = env.step(action, exe_mode=1)
            if done:
                env.finalize_window_history()
            culmulative_reward += reward
            wt_sum += wt_step
            step += 1
        cost, mkspan = env.get_cost()
        print("culmulative_reward", culmulative_reward)

        return best_transitions, [wt_sum, mkspan, cost]

    def evaluate(self, env, max_return, n=10, save_history=True):
        """評価結果を履歴に保存し、優れた解を経験再生バッファに追加するよう拡張したevaluate"""
        n = min(n, len(self.experience_replay))
        episodes = self._nlargest(n)
        returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
        returns = np.float32(returns)
        horizons = np.float32(horizons)
        e_returns = []
        e_values = []
        all_transitions = []  # 全てのtransitionsを保存するリスト
        
        for i in range(n):
            transitions, _, _, _, map_fin, value = self._run_episode(env, returns[i], np.float32(horizons[i]), max_return, eval_mode=True)
            # compute return
            for j in reversed(range(len(transitions) - 1)):
                transitions[j].reward += self.gamma * transitions[j + 1].reward
            e_returns.append(transitions[0].reward)
            e_values.append(value)
            all_transitions.append(transitions)  # transitionsを保存
        
        distances = np.linalg.norm(np.array(returns) - np.array(e_returns), axis=-1)

        # 非支配解を抽出
        non_dominated_inds_reward = get_non_dominated_inds(np.array(e_returns))
        non_dominated_inds_values = get_non_dominated_inds(np.array(e_values))
        pareto_front_reward = np.array(e_returns)[non_dominated_inds_reward]
        pareto_front_values = np.array(e_values)[non_dominated_inds_values]
        
        # ======= 非支配解を経験再生バッファに追加（安全に行う） =======
        if len(non_dominated_inds_reward) > 0:
            # 元のexperience_replayのサイズを保存
            original_size = len(self.experience_replay)
            max_size = original_size
            
            for i in non_dominated_inds_reward:
                # 単純に追加するだけにして、heapqの比較問題を回避
                if len(self.experience_replay) < max_size:
                    # ヒープに追加する際は、優先度は解の品質に基づいて設定
                    # ここでは、報酬の合計値を使用（より大きい方が優先）
                    priority = float(np.sum(e_returns[i]))
                    heapq.heappush(self.experience_replay, (priority, self.global_step, all_transitions[i]))
            
            # 必要に応じてヒープを再構築
            if len(self.experience_replay) > max_size:
                # 最も優先度の低い要素を削除してサイズを調整
                self.experience_replay = heapq.nlargest(max_size, self.experience_replay)
                heapq.heapify(self.experience_replay)
        
        # 履歴に保存
        if save_history:
            self.evaluation_history.append({
                'all_returns': np.array(e_returns),
                'pareto_front_reward': pareto_front_reward,
                'pareto_front_values': pareto_front_values,
                'values': e_values
            })
            self.evaluation_timestamps.append("1")
            self.global_steps_at_evaluation.append(self.global_step)
        
        return e_returns, np.array(returns), distances, map_fin

    def plot_rewards(self, rewards):
        waiting_times, cloud_costs = zip(*rewards)
        plt.figure(figsize=(10, 6))
        plt.scatter(waiting_times, cloud_costs, c='blue', alpha=0.5)
        plt.title('Reward Points: Waiting Time vs Cloud Cost')
        plt.xlabel('Waiting Time')
        plt.ylabel('Cloud Cost')
        plt.grid(True)
        plt.show()

    def save(self, filename: str = "PCN_model", savedir: str = "weights"):
        """保存時に一意のファイル名を生成して新規ファイルを作成"""
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        
        # 一意のIDを生成（タイムスタンプとランダム数字の組み合わせ）
        import datetime
        import random
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = f"{timestamp}_{random.randint(1000, 9999)}"
        
        # 一意のファイル名を作成
        unique_filename = f"{filename}_{unique_id}.pt"
        
        # モデルを保存
        model_path = f"{savedir}/{unique_filename}"
        th.save(self.model, model_path)
        
        # 最新モデルとしてもコピーして保存（最新版へのアクセスを簡単にするため）
        latest_path = f"{savedir}/{filename}_latest.pt"
        import shutil
        shutil.copy2(model_path, latest_path)
        
        # print(f"モデルを保存しました: {model_path}")
        # print(f"最新モデルとしても保存: {latest_path}")
        
        return model_path  # 保存したパスを返す（必要に応じて使用可能）

    def load(self, filename: str = "PCN_model", savedir: str = "weights"):
        """Load PCN."""
        self.model = th.load(f"{savedir}/{filename}.pt")

    def train(
        self,
        total_timesteps: int,
        eval_env: gym.Env,
        ref_point: np.ndarray,
        num_er_episodes: int,
        num_step_episodes: int,
        num_model_updates: int,
        max_buffer_size: int,
        num_eval_weights_for_eval: int = 25,
        max_return: np.ndarray = None,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_points_pf: int = 200,
    ):
        """Train PCN.

        Args:
            total_timesteps: total number of time steps to train for
            eval_env: environment for evaluation
            ref_point: reference point for hypervolume calculation
            known_pareto_front: Optimal pareto front for metrics calculation, if known.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            num_er_episodes: number of episodes to fill experience replay buffer
            num_step_episodes: number of steps per episode
            num_model_updates: number of model updates per episode
            max_return: maximum return for clipping desired return. When None, this will be set to 100 for all objectives.
            max_buffer_size: maximum buffer size
            num_points_pf: number of points to sample from pareto front for metrics calculation
        """
        max_return = max_return if max_return is not None else np.full(self.reward_dim, 100.0, dtype=np.float32)

        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist(),
                    "known_front": known_pareto_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "num_er_episodes": num_er_episodes,
                    "num_step_episodes": num_step_episodes,
                    "num_model_updates": num_model_updates,
                    "max_return": max_return.tolist(),
                    "max_buffer_size": max_buffer_size,
                    "num_points_pf": num_points_pf,
                }
            )
        self.global_step = 0
        total_episodes = num_er_episodes
        n_checkpoints = 0
        desired_return = np.zeros((2,), dtype=np.float32)  # データ型をfloat32に統一
        desired_horizon = np.zeros((1,), dtype=np.float32)  # データ型をfloat32に統一
        # max_return = np.array([1.0, 1.0], dtype=np.float32)  # データ型をfloat32に統一
    
        cumulative_rewards = []
        real_values = []

        # JSON用にエピソード中の配置マップを記録するリストを初期化
        episode_maps_on_premise = []
        episode_maps_cloud = []

        # fill buffer with random episodes
        self.experience_replay = []
        for _ in range(num_er_episodes):
            transitions = []
            obs = self.env.reset()
            done = False
            fifo_count = 0
            step_count = 0

            while not done: #1周目のやつ
                # print("obs: ",obs)
                action = self.env.action_space.sample()
                n_obs, reward, scheduled, wt_step, done = self.env.step(action)
                # print("steped")
                if done:
                    # print("terminated")
                    self.env.finalize_window_history()
                # print("fifo", fifo)
                # print("reward", reward)
                transitions.append(Transition(obs, action, np.float32(reward).copy(), n_obs, done))
                obs = n_obs
                self.global_step += 1

            # self.env.render_map()

            # add episode in-place
            self._add_episode(transitions, max_size=max_buffer_size, step=self.global_step)

            # print("env on_premise_window_history_full: ",self.env.on_premise_window_history_full)
            # print("env cloud_window_history_full: ",self.env.cloud_window_history_full)
            # exit()


        while self.global_step < total_timesteps:
            loss = []
            entropy = []
            for _ in range(num_model_updates):
                l, lp = self.update()
                loss.append(l.detach().cpu().numpy())
                if not self.continuous_action:
                    lp = lp.detach().cpu().numpy()
                    ent = np.sum(-np.exp(lp) * lp)
                    entropy.append(ent)

            desired_return, desired_horizon = self._choose_commands(num_er_episodes)

            # get all leaves, contain biggest elements, experience_replay got heapified in choose_commands
            leaves_r = np.array([e[2][0].reward for e in self.experience_replay[len(self.experience_replay) // 2 :]])
            # leaves_h = np.array([len(e[2]) for e in self.experience_replay[len(self.experience_replay) // 2 :]])
            if self.log:
                hv = hypervolume(ref_point, leaves_r)
                hv_est = hv
                wandb.log(
                    {
                        "train/hypervolume": hv_est,
                        "train/loss": np.mean(loss), 
                        "global_step": self.global_step,
                        # "train/entropy": np.mean(entropy),
                    },
                    )

            returns = []
            horizons = []

            for _ in range(num_step_episodes):
                transitions, maps_on_pre, maps_cloud, wt_sum, map_fin, value = self._run_episode(
                    self.env, desired_return, desired_horizon, max_return
                )
                self.global_step += len(transitions)
                self._add_episode(transitions, max_size=max_buffer_size, step=self.global_step)
                returns.append(transitions[0].reward)
                horizons.append(len(transitions))

                # 各エピソードのstepごとの配置マップを累積
                episode_maps_on_premise.extend(maps_on_pre)
                episode_maps_cloud.extend(maps_cloud)

                # エピソードごとの待ち時間とコストを取得
                _, total_cost = self.env.get_episode_metrics()

                if self.log:
                    wandb.log(
                        {
                            "episode/wt_sum": wt_sum,
                            "episode/total_cost": total_cost,
                        },
                    )
                # 累積報酬を計算してリストに追加
                cumulative_rewards.append(transitions[0].reward)
                real_values.append((wt_sum, total_cost))

            total_episodes += num_step_episodes

            if self.log:
                wandb.log(
                    {
                        "train/episode": total_episodes,
                        "train/horizon_desired": desired_horizon,
                        "train/mean_horizon_distance": np.linalg.norm(np.mean(horizons) - desired_horizon),
                    },
                )

                for i in range(self.reward_dim):
                    wandb.log(
                        {
                            f"train/desired_return_{i}": desired_return[i],
                            f"train/mean_return_{i}": np.mean(np.array(returns)[:, i]),
                            f"train/mean_return_distance_{i}": np.linalg.norm(
                                np.mean(np.array(returns)[:, i]) - desired_return[i]
                            ),
                            "global_step": self.global_step,
                        },
                    )
            print(
                f"step {self.global_step} \t episode{total_episodes} return {np.mean(returns, axis=0)}, ({np.std(returns, axis=0)}) \t loss {np.mean(loss):.3E} \t horizons {np.mean(horizons)}"
            )

            if self.global_step >= (n_checkpoints + 1) * total_timesteps / 1000:
                
                n_checkpoints += 1
                # print("self.global_step", self.global_step)

                self.evaluate(eval_env, max_return, n=num_points_pf)
                # e_returns, _, _ = self.evaluate(eval_env, max_return, n=num_points_pf)
                # self.e_returns.append(e_returns)
        self.save()
        self.e_returns, _, _, self.mapmap = self.evaluate(eval_env, max_return, n=num_points_pf)
        
        # 訓練終了時にパレート解集合を保存
        self.save_pareto_solutions_to_txt(mode_name=f"training_complete")
        
    
    def get_e_returns(self):
        return self.e_returns
    
    def get_transitions(self):
        return self.transitions
    
    def get_mapmap(self):
        return self.mapmap

    def visualize_evaluation_history(self, save_dir="evaluation_history"):
        """評価履歴を可視化し、一意のIDを持つファイルとして保存"""
        if not self.evaluation_history:
            print("評価履歴がありません")
            return
        
        # ディレクトリ作成
        os.makedirs(save_dir, exist_ok=True)
        
        # 一意のIDを生成（現在時刻のタイムスタンプとランダム値を組み合わせる）
        import datetime
        import random
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = f"{timestamp}_{random.randint(1000, 9999)}"
        
        # 全データから適切な表示範囲を計算
        all_x_values = []
        all_y_values = []
        for history in self.evaluation_history:
            all_returns = history['values']
            all_x_values.extend([ret[0] for ret in all_returns])
            all_y_values.extend([ret[1] for ret in all_returns])
        
        # 表示範囲の計算（少しマージンを追加）
        x_min, x_max = min(all_x_values), max(all_x_values)
        y_min, y_max = min(all_y_values), max(all_y_values)
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        x_range = [x_min - x_margin, x_max + x_margin]
        y_range = [y_min - y_margin, y_max + y_margin]
        
        # パレートフロントの進化を可視化
        plt.figure(figsize=(15, 10))
        
        # 各評価時点のパレートフロントをプロット
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.evaluation_history)))
        
        for i, (history, step) in enumerate(zip(self.evaluation_history, self.global_steps_at_evaluation)):
            pareto_front_reward = history['pareto_front_reward']
            pareto_front_values = history['pareto_front_values']
            plt.scatter(
                [ret[0] for ret in pareto_front_values], 
                [ret[1] for ret in pareto_front_values],
                color=colors[i], 
                label=f"Step {step}",
                alpha=0.7
            )
        
        plt.title("パレートフロントの進化")
        plt.xlabel("時間報酬")
        plt.ylabel("コスト")
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()
        
        # 一意のIDを含むファイル名で保存
        pareto_png_filename = f"{save_dir}/pareto_evolution_{unique_id}.png"
        plt.savefig(pareto_png_filename)
        plt.close()
        

        
        # アニメーション作成
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def update(frame):
            ax.clear()
            history = self.evaluation_history[frame]
            pareto_front_reward = history['pareto_front_reward']
            pareto_front_values = history['pareto_front_values']
            all_returns = history['values']
            
            ax.scatter([ret[0] for ret in all_returns], [ret[1] for ret in all_returns], alpha=0.3, color='blue')
            ax.scatter([ret[0] for ret in pareto_front_values], [ret[1] for ret in pareto_front_values], color='red', s=80)
            
            ax.set_title(f"Step {self.global_steps_at_evaluation[frame]}でのパレートフロント")
            ax.set_xlabel("時間報酬")
            ax.set_ylabel("コスト")
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.grid(True)
        
        ani = FuncAnimation(fig, update, frames=len(self.evaluation_history), repeat=True)
        
        # 一意のIDを含むファイル名でGIFを保存
        pareto_gif_filename = f"{save_dir}/pareto_animation_{unique_id}.gif"
        ani.save(pareto_gif_filename, writer='pillow', fps=2)
        plt.close()
        
        print(f"評価履歴の可視化を保存しました:")
        print(f" - パレートフロント画像: {pareto_png_filename}")
        print(f" - アニメーションGIF: {pareto_gif_filename}")

    def save_pareto_solutions_to_txt(self, mode_name="default"):
        """パレートフロントの解をテキストファイルに保存"""
        if not self.evaluation_history:
            print("評価履歴がありません。ファイルは作成されませんでした。")
            return
        
        # 保存ディレクトリの作成
        save_dir = "pareto_solutions"
        os.makedirs(save_dir, exist_ok=True)
        
        # 一意のファイル名を作成
        import datetime
        import random
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = f"{timestamp}_{random.randint(1000, 9999)}"
        
        # 最新の評価結果を取得
        latest_eval = self.evaluation_history[-1]
        
        # 結果をテキストファイルに書き込む
        filename = f"{save_dir}/pareto_solutions_{mode_name}_{unique_id}.txt"
        try:
            with open(filename, 'w') as f:
                # ヘッダー情報
                f.write(f"# パレートフロント解 - {mode_name}\n")
                f.write(f"# 日時: {timestamp}\n")
                f.write(f"# ステップ数: {self.global_step}\n")
                f.write("\n")
                
                # パレートフロントデータ
                f.write("## パレートフロント\n")
                if 'pareto_front_values' in latest_eval:
                    pareto_front_values = latest_eval['pareto_front_values']
                    for i, solution in enumerate(pareto_front_values):
                        f.write(f"解 {i+1}: {solution}\n")
                f.write("\n")
                
                # 実際の評価値
                f.write("## 実際の評価値 (コスト, 実行時間)\n")
                if 'values' in latest_eval:
                    values = latest_eval['values']
                    for i, val in enumerate(values):
                        f.write(f"値 {i+1}: {val}\n")
                f.write("\n")
                
                # 全解のデータ
                f.write("## 全ての報酬\n")
                if 'all_returns' in latest_eval:
                    all_returns = latest_eval['all_returns']
                    for i, ret in enumerate(all_returns):
                        f.write(f"解 {i+1}: {ret}\n")
                
                # マップ情報はテキストでは表現しにくいので省略
                f.write("\n## マップデータは別途画像として保存されます\n")
            
            # マップデータの視覚化を別途保存
            try:
                if 'maps' in latest_eval:
                    final_maps = latest_eval['maps']
                    map_image_path = f"{save_dir}/final_schedule_{mode_name}_{unique_id}.png"
                    visualize_map(final_maps[0], final_maps[1], [], map_image_path)
                    f.write(f"マップ画像: {map_image_path}\n")
                    print(f"スケジュールマップを保存しました: {map_image_path}")
            except Exception as map_err:
                print(f"マップ画像の保存中にエラーが発生しました: {map_err}")
            
            print(f"パレートフロントデータをテキストファイルに保存しました: {filename}")
            return filename
        except Exception as e:
            print(f"ファイルの保存中にエラーが発生しました: {e}")
            return None