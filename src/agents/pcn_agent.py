"""Pareto Conditioned Network. Code adapted from https://github.com/mathieu-reymond/pareto-conditioned-networks ."""
import heapq
import os
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Type, Union
from matplotlib.animation import FuncAnimation
import signal
import time
import traceback

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

# macOS用の日本語フォント設定
try:
    # 一般的なmacOS用日本語フォントを指定
    plt.rcParams['font.family'] = 'Hiragino Sans'
    # もし上記で見つからない場合、他の候補として 'Hiragino Kaku Gothic Pro', 'System Font' なども試せます。
    # または、font_managerを使って利用可能なフォントから探すこともできます。
    # import matplotlib.font_manager
    # japanese_font = next((f.name for f in matplotlib.font_manager.fontManager.ttflist if 'hiragino' in f.name.lower()), None)
    # if japanese_font:
    #     plt.rcParams['font.family'] = japanese_font
    # else:
    #     # 最終手段として、システムデフォルトに任せる（文字化けする可能性が高い）
    #     print("Hiraginoフォントが見つかりませんでした。システムフォントを使用します。")
    #     pass # または plt.rcParams['font.family'] = plt.rcParamsDefault['font.family']
except RuntimeError:
    print("日本語フォント (例: Hiragino Sans) の設定中にエラーが発生しました。グラフの日本語が文字化けする可能性があります。")
    # ここでもフォールバック処理を記述できますが、まずは指定フォントで試します。

from morl_baselines.common.evaluation import log_all_multi_policy_metrics
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.pareto import get_non_dominated_inds, get_non_dominated_inds_minimize
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


class CNN1D(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(CNN1D, self).__init__()
        
        # 1次元CNNレイヤー
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        
        # 全結合層
        cnn_output_size = 32 * (input_dim // 2)
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_size, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        # 入力を [batch_size, 1, input_dim] の形に変形
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = self.fc(x)
        return x


class CNNBackedPCN(nn.Module):
    def __init__(self, n_premise_nodes, n_cloud_nodes, window_size, job_feature_dim, hidden_dim=256):
        super(CNNBackedPCN, self).__init__()
        
        # リソースマップ用CNN
        self.map_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # CNNの出力サイズを計算
        cnn_output_size = 32 * min(n_premise_nodes, n_cloud_nodes) * window_size
        
        # ジョブ特徴エンコーダー
        self.job_encoder = nn.Sequential(
            nn.Linear(job_feature_dim, 128),
            nn.ReLU()
        )
        
        # 特徴結合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(cnn_output_size + 128, hidden_dim),
            nn.ReLU()
        )
        
        # 多目的出力レイヤー
        self.output_layer = nn.Linear(hidden_dim, 2)  # [待ち時間, コスト]
    
    def forward(self, obs_dict, lam=None):
        # リソースマップ処理
        on_premise = obs_dict['on_premise_map'].unsqueeze(1)  # [B, 1, H, W]
        cloud = obs_dict['cloud_map'].unsqueeze(1)             # [B, 1, H, W]
        maps = th.cat([on_premise, cloud], dim=1)          # [B, 2, H, W]
        
        # CNN特徴抽出
        map_features = self.map_cnn(maps)
        
        # ジョブキュー処理
        job_features = self.job_encoder(obs_dict['job_queue'])
        
        # 特徴統合
        combined = th.cat([map_features, job_features], dim=1)
        features = self.fusion_layer(combined)
        
        # 出力生成
        output = self.output_layer(features)
        
        return output


class EnhancedPCNModel(nn.Module):
    """スケジューリング環境のための拡張PCNモデル"""
    def __init__(self, 
                 observation_dim, 
                 n_premise_nodes,
                 n_cloud_nodes,
                 window_size,
                 job_feature_dim=40,
                 hidden_dim=256,
                 reward_dim=2,
                 action_dim=2,
                 debug_mode=True):
        super(EnhancedPCNModel, self).__init__()
        
        # 基本パラメータ初期化
        self.observation_dim = observation_dim
        self.n_premise_nodes = n_premise_nodes
        self.n_cloud_nodes = n_cloud_nodes
        self.window_size = window_size
        self.job_feature_dim = job_feature_dim
        self.hidden_dim = hidden_dim
        self.reward_dim = reward_dim
        self.action_dim = action_dim
        self.debug_mode = debug_mode
        
        # デバッグ出力
        if self.debug_mode:
            print(f"==== モデル構築: 次元情報 ====")
            print(f"観測次元数: {observation_dim}")
            print(f"オンプレミスノード数: {n_premise_nodes}")
            print(f"クラウドノード数: {n_cloud_nodes}")
            print(f"ウィンドウサイズ: {window_size}")
            print(f"ジョブ特徴量次元: {job_feature_dim}")
            print(f"隠れ層次元: {hidden_dim}")
            print(f"報酬次元: {reward_dim}")
            print(f"行動次元: {action_dim}")
            print("============================")
        
        # 1. マップデータ用CNN処理部分（より多くの縮小を行う）
        self.map_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),  # ストライド2に変更
            nn.ReLU(),
            nn.MaxPool2d(2),  # 2x2のプーリングで更に縮小
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # ストライド2に変更
            nn.ReLU(),
            nn.MaxPool2d(2),  # さらに縮小
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # サイズを固定の4x4に
            nn.Flatten()
        )
        
        # CNNの出力サイズを計算（固定サイズに）
        self.cnn_output_dim = 64 * 4 * 4  # 固定サイズの出力
        
        if self.debug_mode:
            print(f"CNN出力次元: {self.cnn_output_dim}")
        
        # 2. ジョブキュー処理部分 - シンプルな線形層に変更
        self.job_embedding = nn.Linear(8, 32)  # 各ジョブは8次元
        self.job_encoder = nn.Sequential(
            nn.Linear(32 * 5, 64),  # 5ジョブ分を固定サイズに
            nn.ReLU()
        )
        
        # 3. 特徴結合層 - 次元を圧縮
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.cnn_output_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 4. PCN条件エンコーディング
        self.condition_encoder = nn.Sequential(
            nn.Linear(reward_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 5. ホライゾン処理
        self.horizon_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Sigmoid()
        )
        
        # 6. 統合と出力層
        self.pi_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.v_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, reward_dim)
        )
        
        # 7. 入力前処理メソッド
        self.preprocess = InputPreprocessor(
            n_premise_nodes=n_premise_nodes,
            n_cloud_nodes=n_cloud_nodes,
            window_size=window_size,
            debug_mode=debug_mode
        )
    
    def extract_state_features(self, x):
        """観測データから状態特徴を抽出（シンプル化）"""
        # 入力データの分離と整形
        maps_data, job_data = self.preprocess(x)
        
        # マップデータのCNN処理
        map_features = self.map_cnn(maps_data)
        
        # ジョブデータの処理（シンプル化）
        batch_size, n_jobs, job_dim = job_data.shape
        job_emb = self.job_embedding(job_data)  # [B, n_jobs, 32]
        job_features = job_emb.view(batch_size, -1)  # フラット化
        job_features = self.job_encoder(job_features)
        
        # 特徴結合
        combined_features = th.cat([map_features, job_features], dim=1)
        state_features = self.feature_fusion(combined_features)
        
        return state_features
    
    def encode_condition(self, r, h=None):
        """報酬重みとホライゾンから条件を生成（シンプル化）"""
        condition = self.condition_encoder(r)
        
        if h is not None:
            horizon_cond = self.horizon_encoder(h)
            condition = condition * horizon_cond
        
        return condition
    
    def forward(self, x, r, h=None):
        """
        前方伝播処理
        Args:
            x: 観測データ
            r: 報酬重み
            h: ホライゾン（オプション）
        """
        if self.debug_mode:
            print(f"\n==== モデル前方伝播 ====")
            print(f"バッチ観測データ入力形状: {x.shape}")
            print(f"報酬重み入力形状: {r.shape}")
            if h is not None:
                print(f"ホライゾン入力形状: {h.shape}")
        
        # 状態特徴抽出
        state_features = self.extract_state_features(x)
        
        # 条件エンコーディング
        condition = self.encode_condition(r, h)
        
        if self.debug_mode:
            print(f"状態特徴形状: {state_features.shape}")
            print(f"条件形状: {condition.shape}")
        
        # 条件付き予測
        # PCNのキーとなる部分: 状態特徴と条件の要素積
        conditioned_features = state_features * condition
        
        if self.debug_mode:
            print(f"条件付き特徴形状: {conditioned_features.shape}")
        
        # 方策と価値予測
        pi = self.pi_net(conditioned_features)
        v = self.v_net(conditioned_features)
        
        if self.debug_mode:
            print(f"π出力形状: {pi.shape}")
            print(f"V出力形状: {v.shape}")
            print("============================")
        
        return pi, v


class InputPreprocessor(nn.Module):
    """生の観測データを処理して適切な形式に変換するプリプロセッサ"""
    def __init__(self, n_premise_nodes, n_cloud_nodes, window_size, debug_mode=True):
        super(InputPreprocessor, self).__init__()
        self.n_premise_nodes = n_premise_nodes
        self.n_cloud_nodes = n_cloud_nodes
        self.window_size = window_size
        self.debug_mode = debug_mode
        
    def forward(self, x):
        """
        観測データをマップデータとジョブデータに分離
        Returns:
            maps_data: [B, 2, max_nodes, window_size] - 2チャネル（オンプレとクラウド）
            job_data: [B, n_jobs, job_dim] - 各ジョブの特徴
        """
        batch_size = x.shape[0]
        
        if self.debug_mode:
            print(f"\n==== 入力前処理 ====")
            print(f"入力観測データ形状: {x.shape}")
        
        # マップデータのインデックス計算
        premise_size = self.n_premise_nodes * self.window_size
        cloud_size = self.n_cloud_nodes * self.window_size
        map_total_size = premise_size + cloud_size
        
        if self.debug_mode:
            print(f"オンプレミスマップサイズ: {premise_size}")
            print(f"クラウドマップサイズ: {cloud_size}")
            print(f"合計マップサイズ: {map_total_size}")
        
        # マップデータの抽出と整形
        map_data = x[:, :map_total_size]
        
        if self.debug_mode:
            print(f"抽出済みマップデータ形状: {map_data.shape}")
        
        premise_map = map_data[:, :premise_size].reshape(batch_size, self.n_premise_nodes, self.window_size)
        cloud_map = map_data[:, premise_size:map_total_size].reshape(batch_size, self.n_cloud_nodes, self.window_size)
        
        if self.debug_mode:
            print(f"整形後オンプレミスマップ形状: {premise_map.shape}")
            print(f"整形後クラウドマップ形状: {cloud_map.shape}")
        
        # 最大ノード数に合わせたパディング
        max_nodes = max(self.n_premise_nodes, self.n_cloud_nodes)
        
        if self.debug_mode:
            print(f"最大ノード数: {max_nodes}")
        
        if self.n_premise_nodes < max_nodes:
            padding = th.zeros(batch_size, max_nodes - self.n_premise_nodes, self.window_size, device=x.device)
            premise_map = th.cat([premise_map, padding], dim=1)
            
            if self.debug_mode:
                print(f"パディング後オンプレミスマップ形状: {premise_map.shape}")
        
        if self.n_cloud_nodes < max_nodes:
            padding = th.zeros(batch_size, max_nodes - self.n_cloud_nodes, self.window_size, device=x.device)
            cloud_map = th.cat([cloud_map, padding], dim=1)
            
            if self.debug_mode:
                print(f"パディング後クラウドマップ形状: {cloud_map.shape}")
        
        # 2チャネル形式に変換 [B, 2, max_nodes, window_size]
        maps_data = th.stack([premise_map, cloud_map], dim=1)
        
        if self.debug_mode:
            print(f"最終マップデータ形状: {maps_data.shape}")
        
        # ジョブデータの抽出
        job_data = x[:, map_total_size:]
        
        if self.debug_mode:
            print(f"抽出済みジョブデータ形状: {job_data.shape}")
            print(f"想定される残りサイズ: {x.shape[1] - map_total_size}")
        
        n_jobs = 5  # 固定値
        job_dim = 8  # 固定値
        
        try:
            job_data = job_data.reshape(batch_size, n_jobs, job_dim)
            
            if self.debug_mode:
                print(f"整形後ジョブデータ形状: {job_data.shape}")
        except:
            if self.debug_mode:
                print(f"エラー! 整形できません。入力サイズと期待形状の不一致:")
                print(f"現在のジョブデータサイズ: {job_data.shape}")
                print(f"期待される整形後サイズ: [{batch_size}, {n_jobs}, {job_dim}]")
                print(f"必要な要素数: {batch_size * n_jobs * job_dim}")
                print(f"実際の要素数: {job_data.numel()}")
        
        return maps_data, job_data


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
        use_enhanced_model: bool = False,
        debug_mode: bool = True,
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
            use_enhanced_model (bool, optional): Whether to use the enhanced model. Defaults to False.
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device)

        # 安全終了処理のためのフラグを初期化
        self.terminate_requested = False
        self.original_sigint = None
        self.original_sigterm = None

        # 既存の初期化コード
        self.reward_dim = env.reward_space.shape[0]
        # 環境から観測空間と行動空間の次元を取得
        self.observation_dim = self.env.observation_space.shape[0]
        self.continuous_action = isinstance(self.env.action_space, gym.spaces.Box)
        if self.continuous_action:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = self.env.action_space.n
        
        self.experience_replay = []
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.scaling_factor = scaling_factor
        self.noise = noise
        self.e_returns = []
        self.transitions = []
        self.mapmap = []
        
        self.use_enhanced_model = use_enhanced_model
        self.debug_mode = debug_mode

        if use_enhanced_model:
            if self.debug_mode:
                print("拡張モデルを使用します。")
                print(f"オンプレミスノード数: {env.n_on_premise_node}")
                print(f"クラウドノード数: {env.n_cloud_node}")
                print(f"ウィンドウサイズ: {env.n_window}")
            
            self.network = EnhancedPCNModel(
                observation_dim=self.observation_dim,
                n_premise_nodes=env.n_on_premise_node,
                n_cloud_nodes=env.n_cloud_node,
                window_size=env.n_window,
                hidden_dim=self.hidden_dim,
                reward_dim=self.reward_dim,
                action_dim=self.action_dim,
                debug_mode=self.debug_mode
            ).to(self.device)
            
            self.target_network = EnhancedPCNModel(
                observation_dim=self.observation_dim,
                n_premise_nodes=env.n_on_premise_node,
                n_cloud_nodes=env.n_cloud_node,
                window_size=env.n_window,
                hidden_dim=self.hidden_dim,
                reward_dim=self.reward_dim,
                action_dim=self.action_dim,
                debug_mode=self.debug_mode
            ).to(self.device)
            
            self.target_network.load_state_dict(self.network.state_dict())
            self.opt = th.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        else:
            if model_class is None:
                if self.continuous_action:
                    model_class = ContinuousActionsDefaultModel
                else:
                    model_class = DiscreteActionsDefaultModel
            
            self.model = model_class(
                self.observation_dim, self.action_dim, self.reward_dim, self.scaling_factor, hidden_dim=self.hidden_dim
            ).to(self.device, non_blocking=True)
            self.opt = th.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.log = log
        if log:
            experiment_name_to_log = experiment_name + (" continuous action" if self.continuous_action else "")
            self.setup_wandb(project_name, experiment_name_to_log, wandb_entity)

        self.evaluation_history = []
        self.evaluation_timestamps = []
        self.global_steps_at_evaluation = []

    def register_signal_handlers(self):
        """シグナルハンドラを登録する"""
        # 元のシグナルハンドラを保存
        self.original_sigint = signal.getsignal(signal.SIGINT)
        self.original_sigterm = signal.getsignal(signal.SIGTERM)
        
        def graceful_shutdown_handler(sig, frame):
            print("\n\n中断シグナルを受信しました。安全に終了処理を実行します...")
            self.terminate_requested = True
            # ここでは即終了せず、トレーニングループが終了確認するのを待つ
        
        # シグナルハンドラを設定
        signal.signal(signal.SIGINT, graceful_shutdown_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, graceful_shutdown_handler)  # killコマンド
    
    def restore_signal_handlers(self):
        """元のシグナルハンドラを復元する"""
        if self.original_sigint:
            signal.signal(signal.SIGINT, self.original_sigint)
        if self.original_sigterm:
            signal.signal(signal.SIGTERM, self.original_sigterm)
    
    def save_results_on_termination(self, eval_env, max_return, num_points_pf=200):
        """終了時に結果を保存するメソッド"""
        try:
            print("現在の学習状態を保存しています...")
            
            # 特別なフラグ付きでモデルを保存
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.save(filename=f"PCN_model_interrupted_{timestamp}")
            
            # 最終評価を実行
            print("最終評価を実行しています...")
            self.e_returns, _, _, self.mapmap = self.evaluate(
                eval_env, max_return, n=num_points_pf, save_history=True
            )
            
            # 評価結果の可視化
            self.visualize_evaluation_history(save_dir=f"interrupted_results_{timestamp}")
            
            # パレート解の保存
            self.save_pareto_solutions_to_txt(mode_name=f"interrupted_{timestamp}")
            
            print("\n終了処理が完了しました。結果は以下に保存されました：")
            print(f"- モデル: weights/PCN_model_interrupted_{timestamp}.pt")
            print(f"- 評価結果: interrupted_results_{timestamp}/")
            print(f"- パレート解: pareto_solutions/pareto_solutions_interrupted_{timestamp}_*.txt")
        except Exception as e:
            print(f"終了処理中にエラーが発生しました: {e}")
            traceback.print_exc()

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
            obs = th.from_numpy(np.stack(observations_list)).to(self.device).float()
            actions = th.from_numpy(np.stack(actions_list)).to(self.device)
            desired_return = th.from_numpy(np.stack(desired_return_list)).to(self.device).float()
            desired_horizon = th.from_numpy(np.stack(desired_horizon_list)).to(self.device).float().unsqueeze(1)
            
            self.opt.zero_grad(set_to_none=True)

            if self.use_enhanced_model:
                prediction_output = self.network(obs, desired_return, desired_horizon)
            else:
                prediction_output = self.model(obs, desired_return, desired_horizon)
            
            # モデルの出力がタプルである可能性への対応 (EnhancedPCNModel.forward が単一テンソルを返せば不要)
            if isinstance(prediction_output, tuple):
                prediction_logits = prediction_output[0]
            else:
                prediction_logits = prediction_output
            
            if self.continuous_action:
                # 連続行動の場合、prediction_logits は直接的な行動の値を意味する
                l = F.mse_loss(actions.float(), prediction_logits)
            else:
                # 離散行動の場合
                if self.use_enhanced_model:
                    # EnhancedPCNModelはlogitsを出力するため、CrossEntropyLossを使用
                    l = F.cross_entropy(prediction_logits, actions.long())
                else:
                    # DiscreteActionsDefaultModelはlog_probsを出力するため、NLLLossを使用
                    # (DiscreteActionsDefaultModelの最後がLogSoftmaxであることを前提)
                    l = F.nll_loss(prediction_logits, actions.long())
            
            l.backward()
            self.opt.step()
            
            return l, prediction_logits

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
            horizon_tensor = th.tensor([[desired_horizon]], device=self.device).float()
            
            if self.use_enhanced_model:
                prediction_output = self.network(obs_tensor, return_tensor, horizon_tensor)
            else:
                prediction_output = self.model(obs_tensor, return_tensor, horizon_tensor)

            if isinstance(prediction_output, tuple):
                prediction_scores = prediction_output[0]
            else:
                prediction_scores = prediction_output

        if self.continuous_action:
            action = prediction_scores.detach().cpu().numpy()[0]
            if not eval_mode:
                action = action + self.np_random.normal(0.0, self.noise, size=action.shape)
            return action
        else:
            scores = prediction_scores.detach()[0]
            if eval_mode:
                action = th.argmax(scores).item()
            else:
                if self.use_enhanced_model:
                    probs = F.softmax(scores, dim=-1)
                else:
                    probs = th.exp(scores)
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
            if scheduled:
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
        non_dominated_inds_values = get_non_dominated_inds_minimize(np.array(e_values))
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

        if self.use_enhanced_model:
            th.save(self.network.state_dict(), model_path) # 拡張モデルのstate_dictを保存
        else:
            th.save(self.model.state_dict(), model_path) # 通常モデルのstate_dictを保存
        
        # 最新モデルとしてもコピーして保存（最新版へのアクセスを簡単にするため）
        latest_path = f"{savedir}/{filename}_latest.pt"
        import shutil
        shutil.copy2(model_path, latest_path)
        
        # print(f"モデルを保存しました: {model_path}")
        # print(f"最新モデルとしても保存: {latest_path}")

    def load(self, filename: str = "PCN_model", savedir: str = "weights"):
        """指定されたモデルを読み込み、拡張モデルか通常モデルかを自動判別"""
        model_path = f"{savedir}/{filename}.pt"
        if not os.path.exists(model_path):
            latest_path = f"{savedir}/{filename}_latest.pt"
            if os.path.exists(latest_path):
                model_path = latest_path
            else:
                print(f"モデルファイルが見つかりません: {model_path} および {latest_path}")
                return

        # state_dictを読み込む
        state_dict = th.load(model_path, map_location=self.device)

        try:
            if self.use_enhanced_model:
                # EnhancedPCNModel が __init__ で正しく初期化されている前提
                self.network.load_state_dict(state_dict)
                self.target_network.load_state_dict(state_dict) # ターゲットネットワークも同期
                print(f"拡張モデルを読み込みました: {model_path}")
            else:
                # BasePCNModel のサブクラスが __init__ で正しく初期化されている前提
                self.model.load_state_dict(state_dict)
                print(f"通常モデルを読み込みました: {model_path}")
        except RuntimeError as e:
            print(f"モデルの読み込み中にエラーが発生しました（キーの不一致など）: {e}")
            print("モデルのアーキテクチャが保存時と異なる可能性があります。")
        except AttributeError as e:
            print(f"モデルの読み込み中にエラーが発生しました: {e}")
            print("`self.network` または `self.model` が正しく初期化されていません。")

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
        log_episode_only: bool = True,
    ):
        """Train PCN with support for safe termination."""
        # シグナルハンドラを登録
        self.register_signal_handlers()
        
        try:
            # ユーザーに中断機能について通知
            print("\n=== PCN学習を開始します。Ctrl+Cで安全に終了できます ===\n")
            
            # 既存の初期化コード
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
                        "log_episode_only": log_episode_only,
                    }
                )
                
            self.global_step = 0
            total_episodes = num_er_episodes
            n_checkpoints = 0
            desired_return = np.zeros((2,), dtype=np.float32)
            desired_horizon = np.zeros((1,), dtype=np.float32)
            
            cumulative_rewards = []
            real_values = []
            episode_maps_on_premise = []
            episode_maps_cloud = []

            # fill buffer with random episodes
            self.experience_replay = []
            num_count = 0
            print(f"Starting to fill experience replay buffer for {num_er_episodes} episodes...")
            
            for episode_idx in range(num_er_episodes):
                # 中断要求を確認
                if self.terminate_requested:
                    print("経験バッファ充填中に中断要求を受信しました。")
                    break
                    
                transitions = []
                obs = self.env.reset()
                done = False

                while not done:
                    # 中断要求を確認（長いエピソード内でも）
                    if self.terminate_requested:
                        break
                        
                    action = self.env.action_space.sample()
                    n_obs, reward, scheduled, wt_step, done = self.env.step(action)
                    
                    if done:
                        self.env.finalize_window_history()
                        num_count += 1
                        
                        # 1000エピソードごとに進捗表示
                        if num_count % 1000 == 0:
                            print(f"Completed {num_count} episodes for experience replay buffer.")
                            
                            # log_episode_onlyがTrueの場合は、エピソード数のみをログに記録
                            if log_episode_only and self.log:
                                wandb.log({"episodes": num_count})
                    
                    transitions.append(Transition(obs, action, np.float32(reward).copy(), n_obs, done))
                    obs = n_obs
                    self.global_step += 1

                # 中断されていない場合のみエピソードを追加
                if not self.terminate_requested:
                    self._add_episode(transitions, max_size=max_buffer_size, step=self.global_step)
            
            # バッファ充填の最終状態を表示
            if num_count % 1000 != 0:
                print(f"Total of {num_count} episodes completed for experience replay buffer.")

            # メインの学習ループ
            while self.global_step < total_timesteps and not self.terminate_requested:
                loss = []
                entropy = []
                
                # 中断要求を確認
                if self.terminate_requested:
                    break
                    
                for update_idx in range(num_model_updates):
                    # 頻繁な中断チェック（更新が多い場合）
                    if update_idx % 100 == 0 and self.terminate_requested:
                        break
                        
                    l, lp = self.update()
                    loss.append(l.detach().cpu().numpy())
                    if not self.continuous_action:
                        lp = lp.detach().cpu().numpy()
                        ent = np.sum(-np.exp(lp) * lp)
                        entropy.append(ent)

                # モデル更新中に中断された場合
                if self.terminate_requested:
                    break

                desired_return, desired_horizon = self._choose_commands(num_er_episodes)

                # 既存のログコード
                leaves_r = np.array([e[2][0].reward for e in self.experience_replay[len(self.experience_replay) // 2 :]])
                if self.log and not log_episode_only:
                    hv = hypervolume(ref_point, leaves_r)
                    hv_est = hv
                    wandb.log(
                        {
                            "train/hypervolume": hv_est,
                            "train/loss": np.mean(loss), 
                            "global_step": self.global_step,
                        },
                    )

                returns = []
                horizons = []

                for episode_idx in range(num_step_episodes):
                    # 中断要求を確認
                    if self.terminate_requested:
                        break
                        
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

                    if self.log and not log_episode_only:
                        wandb.log(
                            {
                                "episode/wt_sum": wt_sum,
                                "episode/total_cost": total_cost,
                            },
                        )
                    # 累積報酬を計算してリストに追加
                    cumulative_rewards.append(transitions[0].reward)
                    real_values.append((wt_sum, total_cost))

                # エピソード実行中に中断された場合
                if self.terminate_requested:
                    break

                total_episodes += num_step_episodes
                
                # log_episode_onlyがTrueの場合は、エピソード数のみをログに記録
                if self.log and log_episode_only:
                    wandb.log({"episodes": total_episodes})

                # 既存のログ処理
                if self.log and not log_episode_only and len(returns) > 0:  # 中断時に空の場合に備えて確認
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
                
                # 学習状況のコンソール出力
                if len(returns) > 0:  # 中断時に空の場合に備えて確認
                    print(
                        f"step {self.global_step}/{total_timesteps} ({self.global_step/total_timesteps*100:.1f}%) | "
                        f"episode {total_episodes} | "
                        f"return {np.mean(returns, axis=0)} | "
                        f"loss {np.mean(loss):.3E} | "
                        f"horizons {np.mean(horizons)}"
                    )

                # 定期的な評価とチェックポイント
                if self.global_step >= (n_checkpoints + 1) * total_timesteps / 1000:
                    n_checkpoints += 1
                    
                    # 中断要求を確認
                    if self.terminate_requested:
                        break
                        
                    # 定期評価を実行
                    print(f"Evaluating at step {self.global_step}...")
                    self.evaluate(eval_env, max_return, n=num_points_pf)
                    
                    # 中断処理のために短いスリープを入れる（Ctrl+Cを処理する時間）
                    time.sleep(0.1)

            # 学習ループ終了（通常完了または中断）
            training_status = "中断" if self.terminate_requested else "正常完了"
            print(f"\n=== 学習が{training_status}しました ===")
            
            # 終了処理（通常終了時も中断時も実行）
            if not self.terminate_requested:
                # 通常終了時の保存処理
                self.save()
                self.e_returns, _, _, self.mapmap = self.evaluate(eval_env, max_return, n=num_points_pf)
                self.save_pareto_solutions_to_txt(mode_name="training_complete")
                print("訓練結果を保存しました。")
            else:
                # 中断時は特別な終了処理を実行
                self.save_results_on_termination(eval_env, max_return, num_points_pf)
            
        except Exception as e:
            print(f"学習中に予期しないエラーが発生しました: {e}")
            traceback.print_exc()
            # エラー発生時も安全に終了処理を試みる
            self.save_results_on_termination(eval_env, max_return, num_points_pf)
        
        finally:
            # 元のシグナルハンドラを復元
            self.restore_signal_handlers()
        
        return self.e_returns

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
        plt.xlabel("時間（実数）")
        plt.ylabel("コスト（実数）")
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