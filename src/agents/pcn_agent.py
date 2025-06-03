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
import gymnasium.spaces as spaces
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
from morl_baselines.common.performance_indicators import hypervolume
from src.utils.map_visualizer import visualize_map


# 非支配解を取得する関数をファイル内に直接実装
def get_non_dominated_inds(points):
    """非支配解（最大化問題用）のインデックスを取得する関数
    
    Args:
        points (np.ndarray): 評価点の配列（行が各解、列が各目的関数値）
    
    Returns:
        np.ndarray: 非支配解のインデックス配列
    """
    if len(points) == 0:
        return np.array([])
    
    # データ型を浮動小数点の高精度型に変換して数値安定性を向上
    points = np.array(points, dtype=np.float64)
    is_efficient = np.ones(len(points), dtype=bool)
    
    for i, point in enumerate(points):
        if is_efficient[i]:
            # 他の解と比較して、すべての目的関数において同等以上で、
            # 少なくとも1つの目的関数において厳密に優れている場合、
            # その他の解は支配されていると判定
            is_efficient[is_efficient] = np.any(
                points[is_efficient] > point, axis=1
            ) | np.all(points[is_efficient] == point, axis=1)
            is_efficient[i] = True  # 自分自身を再度非支配解としてマーク
    
    return np.nonzero(is_efficient)[0]

def get_non_dominated_inds_minimize(points):
    """非支配解（最小化問題用）のインデックスを取得する関数
    
    Args:
        points (np.ndarray): 評価点の配列（行が各解、列が各目的関数値）
    
    Returns:
        np.ndarray: 非支配解のインデックス配列
    """
    if len(points) == 0:
        return np.array([])
    
    # データ型を浮動小数点の高精度型に変換して数値安定性を向上
    points = np.array(points, dtype=np.float64)
    is_efficient = np.ones(len(points), dtype=bool)
    
    for i, point in enumerate(points):
        if is_efficient[i]:
            # 最小化問題では、他の解と比較して、すべての目的関数において同等以下で、
            # 少なくとも1つの目的関数において厳密に優れている（値が小さい）場合、
            # その他の解は支配されていると判定
            is_efficient[is_efficient] = np.any(
                points[is_efficient] < point, axis=1
            ) | np.all(points[is_efficient] == point, axis=1)
            is_efficient[i] = True  # 自分自身を再度非支配解としてマーク
    
    return np.nonzero(is_efficient)[0]


def crowding_distance(points):
    """端点特別処理を除去した混雑度計算"""
    # 数値の安定性向上のためfloat64を使用
    points = np.array(points, dtype=np.float64)
    
    # 次元が少ない場合の処理
    if points.shape[0] <= 2:
        return np.ones(points.shape[0])
    
    # first normalize across dimensions
    points = (points - points.min(axis=0)) / (np.ptp(points, axis=0) + 1e-8)
    # sort points per dimension
    dim_sorted = np.argsort(points, axis=0)
    point_sorted = np.take_along_axis(points, dim_sorted, axis=0)
    
    # 全ての点に対して前後の点との距離を計算
    distances_full = np.zeros((points.shape[0], points.shape[1]))
    
    # 中間点の処理（従来通り）
    if points.shape[0] > 4:
        middle_distances = np.abs(point_sorted[1:-1] - point_sorted[2:])
        distances_full[1:-1] = middle_distances
    
    # 端点の処理（特別扱いせず、隣接点との距離で計算）
    if points.shape[0] >= 2:
        # 最初の点は2番目の点との距離
        distances_full[0] = np.abs(point_sorted[0] - point_sorted[1])
        # 最後の点は最後から2番目の点との距離
        distances_full[-1] = np.abs(point_sorted[-1] - point_sorted[-2])
    
    # sum distances of each dimension of the same point
    crowding = np.zeros(points.shape)
    for d in range(points.shape[1]):
        crowding[dim_sorted[:, d], d] = distances_full[:, d]
    
    crowding = np.sum(crowding, axis=1)
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
            nn.LeakyReLU(negative_slope=0.01),
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
        use_wandb: bool = True,
        log_episode_only: bool = True,
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

    def _nlargest(self, n, threshold=0.1):
        """非支配解の選択処理を改善 - ログ表示を追加"""
        returns = np.array([e[2][0].reward for e in self.experience_replay], dtype=np.float64)
        
        print("\n===== 非支配解選択プロセス =====")
        print(f"バッファ内の解の数: {len(returns)}")
        
        # 非支配解の選択
        non_dominated_i = get_non_dominated_inds(returns)
        non_dominated = returns[non_dominated_i]
        # print(f"非支配解の数: {len(non_dominated_i)}")
        
        # if len(non_dominated_i) > 0:
        #     print("非支配解の一部:")
        #     for i, idx in enumerate(non_dominated_i[:min(5, len(non_dominated_i))]):
        #         print(f"  解{i+1}: {returns[idx]}")
        #     if len(non_dominated_i) > 5:
        #         print(f"  ... 他 {len(non_dominated_i)-5} 個")
        
        # 全点間の距離に基づく混雑度計算
        distances = crowding_distance(returns)
        avg_crowding = np.mean(distances)
        print(f"平均混雑度: {avg_crowding:.4f}")
        
        # 混雑している点の特定
        sma = np.argwhere(distances <= threshold).flatten()
        print(f"混雑点の数 (閾値 {threshold}): {len(sma)}")
        
        # 端点をチェック
        edge_points = []
        for d in range(returns.shape[1]):
            edge_points.append(np.argmin(returns[:, d]))
            edge_points.append(np.argmax(returns[:, d]))
        edge_points = np.unique(edge_points)
        print(f"特定された端点の数: {len(edge_points)}")
        
        # 距離計算と優先度設定
        returns_exp = np.tile(np.expand_dims(returns, 1), (1, len(non_dominated), 1))
        diff_array = (returns_exp - non_dominated).astype(np.float64)
        l2 = np.min(np.linalg.norm(diff_array, axis=-1), axis=-1) * -1
        
        print("優先度の付与:")
        print(f"  基本優先度の範囲: {np.min(l2):.4f} 〜 {np.max(l2):.4f}")
        
        # 以下は既存のコードに沿って処理...
        
        # 結果の表示
        sorted_i = np.argsort(l2)
        largest = [self.experience_replay[i] for i in sorted_i[-n:]]
        print(f"\n選択された {n} 個の解の内訳:")
        selected_returns = np.array([e[2][0].reward for e in largest])
        selected_non_dom = get_non_dominated_inds(selected_returns)
        print(f"  - 選択解のうち非支配解の数: {len(selected_non_dom)}")
        print(f"  - 選択解の優先度範囲: {np.min([e[0] for e in largest]):.4f} 〜 {np.max([e[0] for e in largest]):.4f}")
        print("==================================\n")
        
        # ヒープの更新処理...
        
        return largest

    def _choose_commands(self, num_episodes: int):
        """探索方向を決定するメソッド - 偏り修正版"""
        episodes = self._nlargest(num_episodes)
        returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
        
        # 非支配解のみを保持
        returns_array = np.array(returns)
        nd_i = get_non_dominated_inds(returns_array)
        returns = returns_array[nd_i]
        horizons = np.array(horizons)[nd_i]
        
        # 重複する非支配解を除外
        unique_returns, unique_indices = np.unique(returns, axis=0, return_index=True)
        returns = unique_returns
        horizons = horizons[unique_indices]
        
        if len(returns) > 0:
            # パレートフロント上での位置を正規化
            normalized_returns = (returns - returns.min(axis=0)) / (returns.max(axis=0) - returns.min(axis=0) + 1e-8)
            
            # 端点を避け、中央付近の解を優先
            center_dists = np.linalg.norm(normalized_returns - 0.5, axis=1)
            center_weights = 1.0 / (center_dists + 0.1)  # 中央に近いほど重み大きく
            
            # 確率的サンプリング（中央寄りの点が選ばれやすい）
            weights = center_weights / np.sum(center_weights)
            r_i = self.np_random.choice(len(returns), p=weights)
            
            # 報酬設定
            desired_return = returns[r_i].copy()
            desired_horizon = np.float32(horizons[r_i] - 2)
            
            # 合理的な改善量の設定（報酬の範囲の5%程度）
            reward_range = np.ptp(returns, axis=0)
            dim = self.np_random.integers(0, self.reward_dim)
            improvement = self.np_random.uniform(0, reward_range[dim]) 
            desired_return[dim] += improvement
            
            return np.float32(desired_return), desired_horizon
        else:
            # 非支配解がない場合のフォールバック
            return np.zeros(self.reward_dim, dtype=np.float32), np.float32(40)

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
        map_snapshots_on_premise = []
        map_snapshots_cloud = []
        
        obs = env.reset()
        done = False
        wt_sum = 0
        # 
        # print("\n===== エピソード実行 =====")
        # print(f"目標: 報酬={desired_return}, ステップ数={desired_horizon}")
        
        while not done:
            action = self._act(obs, desired_return, desired_horizon, eval_mode)
            n_obs, reward, scheduled, wt_step, done = env.step(action)
            
            if done:
                env.finalize_window_history()
                
            on_pre_map = env.on_premise_window["job_id"].tolist()
            cloud_map = env.cloud_window["job_id"].tolist()
            map_snapshots_on_premise.append(on_pre_map)
            map_snapshots_cloud.append(cloud_map)
            
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
        
        # エピソード完了後の結果表示
        episode_return = transitions[0].reward  # 累積報酬
        for i in reversed(range(len(transitions) - 1)):
            transitions[i].reward += self.gamma * transitions[i + 1].reward
        
        final_return = transitions[0].reward
        onpre_final = env.on_premise_window_history_full
        cloud_final = env.cloud_window_history_full
        value_cost, value_wt = env.calc_objective_values()
        
        # print(f"エピソード完了: 長さ={len(transitions)}")
        # print(f"最終報酬: {final_return}")
        # print(f"実際の値: コスト={value_cost}, 待ち時間={value_wt}")
        # print("=========================\n")
        
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
        e_returns_np = np.array(e_returns, dtype=np.float64)
        non_dominated_inds = get_non_dominated_inds(e_returns_np)
        pareto_front = e_returns_np[non_dominated_inds]
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
        
        # CUDA対応の行列計算
        if th.cuda.is_available():
            # NumPy配列をPyTorchテンソルに変換してGPUに転送
            returns_tensor = th.tensor(np.array(returns), device=self.device)
            e_returns_tensor = th.tensor(np.array(e_returns), device=self.device)
            
            # GPU上で距離計算
            with th.cuda.amp.autocast():
                distances_tensor = th.norm(returns_tensor - e_returns_tensor, dim=1)
            
            # 結果をCPUに戻してNumPy配列に変換
            distances = distances_tensor.cpu().numpy()
        else:
            # 従来通りNumPyで計算（オーバーフロー防止のためfloat64を使用）
            distances = np.linalg.norm(
                np.array(returns, dtype=np.float64) - np.array(e_returns, dtype=np.float64), 
                axis=-1
            )

        # 非支配解を抽出（CPU上で実行）
        e_returns_np = np.array(e_returns, dtype=np.float64)  # float64でキャスト
        e_values_np = np.array(e_values, dtype=np.float64)    # float64でキャスト
        
        non_dominated_inds_reward = get_non_dominated_inds(e_returns_np)
        non_dominated_inds_values = get_non_dominated_inds_minimize(e_values_np)
        pareto_front_reward = e_returns_np[non_dominated_inds_reward]
        pareto_front_values = e_values_np[non_dominated_inds_values]
        
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

    def clean_zero_crowding_points(self, threshold=0.001):
        """混雑度が閾値以下（実質的に重複）の点をバッファから削除する"""
        if len(self.experience_replay) <= 5:  # バッファが少なすぎる場合は何もしない
            return 0
            
        returns = np.array([e[2][0].reward for e in self.experience_replay], dtype=np.float64)
        cd_values = crowding_distance(returns)
        
        # 非支配解のインデックスを取得
        non_dom_indices = get_non_dominated_inds(returns)
        
        # 新しいバッファを準備
        new_buffer = []
        removed_count = 0
        
        for i, (priority, step, transitions) in enumerate(self.experience_replay):
            # 混雑度が閾値より大きいか、非支配解の場合は保持
            if cd_values[i] > threshold or i in non_dom_indices:
                new_buffer.append((priority, step, transitions))
            else:
                removed_count += 1
        
        if removed_count > 0:
            # バッファの更新
            self.experience_replay = new_buffer
            heapq.heapify(self.experience_replay)
            print(f"バッファクリーニング: 混雑度{threshold}以下の{removed_count}個のエピソードを除去。残り{len(self.experience_replay)}個")
        
        return removed_count

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
        num_eval_episodes: int = 30,
        log_episode_only: bool = True,
        use_wandb: bool = True,
        reset_buffer: bool = False,  # バッファをリセットするかどうかの新しいパラメータ
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
            total_episodes = 0  # ヒューリスティックエピソードを含めないよう0から開始
            n_checkpoints = 0
            desired_return = np.zeros((2,), dtype=np.float32)
            desired_horizon = np.zeros((1,), dtype=np.float32)
            
            cumulative_rewards = []
            real_values = []
            episode_maps_on_premise = []
            episode_maps_cloud = []
            
            # クリーニングのカウンター
            cleaning_counter = 0
            
            # バッファにランダムエピソードを追加（既存のバッファ内容は保持）
            existing_buffer_size = len(self.experience_replay)
            
            # バッファがすでにヒューリスティックデータで初期化されているか確認
            if reset_buffer or existing_buffer_size == 0:
                # バッファを空にして新しく埋める
                self.experience_replay = []
                print(f"経験再生バッファをリセットし、{num_er_episodes}エピソードのランダムデータで埋めます...")
                
                num_count = 0
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
                
                total_episodes = num_count
            else:
                # 既存のバッファを使用
                print(f"既存の経験再生バッファを使用します（サイズ: {existing_buffer_size}エピソード）")
                total_episodes = existing_buffer_size

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

                if use_wandb and log_episode_only:
                    wandb.log({
                        "episodes": total_episodes,
                        "global_step": self.global_step,
                        "loss": np.mean(loss),
                    })

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
                    wandb.log({
                        "episodes": total_episodes,
                        "global_step": self.global_step,
                        "loss": np.mean(loss),
                    })

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
                leaves_r = np.array([e[2][0].reward for e in self.experience_replay[len(self.experience_replay) // 2 :]])
                # print("self.experience_replay",self.experience_replay)
                hv = hypervolume(ref_point, leaves_r)
                # print("leaves_r",leaves_r)
                
                # 各点のCrowding Distanceを計算して表示
                if len(leaves_r) > 1:  # 少なくとも2点必要
                    cd_values = crowding_distance(leaves_r)
                    # print("各点のCrowding Distance:")
                    # for i, (point, cd) in enumerate(zip(leaves_r, cd_values)):
                    #     print(f"  点{i+1}: {point} -> 混雑度: {cd:.4f}")
                    
                    # 非支配解を特定して表示
                    non_dom_indices = get_non_dominated_inds(leaves_r)
                    # print(f"非支配解の数: {len(non_dom_indices)}")
                    # if len(non_dom_indices) > 0:
                        # print("非支配解:")
                        # for i, idx in enumerate(non_dom_indices):
                            # print(f"  解{i+1}: {leaves_r[idx]} -> 混雑度: {cd_values[idx]:.4f}")
                
                hv_est = hv
                
                # 定期的に混雑度0の点をクリーニング（20エピソードごと）
                cleaning_counter += 1
                if cleaning_counter >= 20:
                    self.clean_zero_crowding_points(threshold=0.001)
                    cleaning_counter = 0
 
                if len(returns) > 0:  # 中断時に空の場合に備えて確認
                    print(
                        f"step {self.global_step}/{total_timesteps} ({self.global_step/total_timesteps*100:.1f}%) | "
                        f"episode {total_episodes} | "
                        f"return {np.mean(returns, axis=0)} | "
                        f"loss {np.mean(loss):.3E} | "
                        f"hv {hv_est}"
                    )



                # 前回評価時のエピソード数を記録
                if not hasattr(self, 'last_eval_episode'):
                    self.last_eval_episode = 0

                # より厳密な条件で評価実行
                if total_episodes >= self.last_eval_episode + num_eval_episodes:
                    print(f"エピソード{total_episodes}での評価を実行中...")
                    self.evaluate(eval_env, max_return, n=num_points_pf)
                    self.last_eval_episode = total_episodes
                    
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
        """評価履歴を可視化し、一意のIDを持つファイルとして保存。
        報酬（最大化目的）と実数値（最小化目的）の両方のグラフを別々に表示する。
        """
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
        
        # ------ 実数値（最小化目的）のグラフ ------
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
        
        # パレートフロントの進化を可視化（実数値 - 最小化目的）
        plt.figure(figsize=(15, 10))
        
        # 各評価時点のパレートフロントをプロット
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.evaluation_history)))
        
        for i, (history, step) in enumerate(zip(self.evaluation_history, self.global_steps_at_evaluation)):
            pareto_front_values = history['pareto_front_values']
            plt.scatter(
                [ret[0] for ret in pareto_front_values], 
                [ret[1] for ret in pareto_front_values],
                color=colors[i], 
                label=f"Step {step}",
                alpha=0.7
            )
        
        plt.title("パレートフロントの進化（実数値 - 最小化目的）")
        plt.xlabel("コスト（実数）")
        plt.ylabel("makespan（実数）")
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        
        # 左下が最適な方向であることを示す矢印
        arrow_x = x_min + (x_max - x_min) * 0.85
        arrow_y = y_min + (y_max - y_min) * 0.85
        plt.annotate('最適方向', xy=(arrow_x - (x_max - x_min) * 0.2, arrow_y - (y_max - y_min) * 0.2), 
                    xytext=(arrow_x, arrow_y),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                    fontsize=12)
        
        # 一意のIDを含むファイル名で保存
        pareto_values_png_filename = f"{save_dir}/pareto_values_evolution_{unique_id}.png"
        plt.savefig(pareto_values_png_filename)
        plt.close()
        
        # ------ 報酬（最大化目的）のグラフ ------
        # 全データから適切な表示範囲を計算
        all_x_values_reward = []
        all_y_values_reward = []
        for history in self.evaluation_history:
            all_returns = history['all_returns']
            all_x_values_reward.extend([ret[0] for ret in all_returns])
            all_y_values_reward.extend([ret[1] for ret in all_returns])
        
        # 表示範囲の計算（少しマージンを追加）
        x_min_reward, x_max_reward = min(all_x_values_reward), max(all_x_values_reward)
        y_min_reward, y_max_reward = min(all_y_values_reward), max(all_y_values_reward)
        x_margin_reward = (x_max_reward - x_min_reward) * 0.1
        y_margin_reward = (y_max_reward - y_min_reward) * 0.1
        x_range_reward = [x_min_reward - x_margin_reward, x_max_reward + x_margin_reward]
        y_range_reward = [y_min_reward - y_margin_reward, y_max_reward + y_margin_reward]
        
        # パレートフロントの進化を可視化（報酬 - 最大化目的）
        plt.figure(figsize=(15, 10))
        
        # 各評価時点のパレートフロントをプロット
        for i, (history, step) in enumerate(zip(self.evaluation_history, self.global_steps_at_evaluation)):
            pareto_front_reward = history['pareto_front_reward']
            plt.scatter(
                [ret[0] for ret in pareto_front_reward], 
                [ret[1] for ret in pareto_front_reward],
                color=colors[i], 
                label=f"Step {step}",
                alpha=0.7
            )
        
        plt.title("パレートフロントの進化（報酬 - 最大化目的）")
        plt.xlabel("報酬1")
        plt.ylabel("報酬2")
        plt.xlim(x_range_reward)
        plt.ylim(y_range_reward)
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        # 右上が最適な方向であることを示す矢印
        arrow_x_reward = x_min_reward + (x_max_reward - x_min_reward) * 0.15
        arrow_y_reward = y_min_reward + (y_max_reward - y_min_reward) * 0.15
        plt.annotate('最適方向', xy=(arrow_x_reward + (x_max_reward - x_min_reward) * 0.2, 
                                arrow_y_reward + (y_max_reward - y_min_reward) * 0.2), 
                    xytext=(arrow_x_reward, arrow_y_reward),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                    fontsize=12)
        
        # 一意のIDを含むファイル名で保存
        pareto_rewards_png_filename = f"{save_dir}/pareto_rewards_evolution_{unique_id}.png"
        plt.savefig(pareto_rewards_png_filename)
        plt.close()
        
        # ------ アニメーション作成（実数値） ------
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def update_values(frame):
            ax.clear()
            history = self.evaluation_history[frame]
            pareto_front_values = history['pareto_front_values']
            all_returns = history['values']
            
            ax.scatter([ret[0] for ret in all_returns], [ret[1] for ret in all_returns], alpha=0.3, color='blue')
            ax.scatter([ret[0] for ret in pareto_front_values], [ret[1] for ret in pareto_front_values], color='red', s=80)
            
            ax.set_title(f"Step {self.global_steps_at_evaluation[frame]}でのパレートフロント（実数値 - 最小化目的）")
            ax.set_xlabel("コスト（実数）")
            ax.set_ylabel("makespan（実数）")
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.grid(True)
            
            # 左下が最適な方向であることを示す矢印
            arrow_x = x_min + (x_max - x_min) * 0.85
            arrow_y = y_min + (y_max - y_min) * 0.85
            ax.annotate('最適方向', xy=(arrow_x - (x_max - x_min) * 0.2, arrow_y - (y_max - y_min) * 0.2), 
                       xytext=(arrow_x, arrow_y),
                       arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                       fontsize=12)
        
        ani_values = FuncAnimation(fig, update_values, frames=len(self.evaluation_history), repeat=True)
        
        # 一意のIDを含むファイル名でGIFを保存
        pareto_values_gif_filename = f"{save_dir}/pareto_values_animation_{unique_id}.gif"
        ani_values.save(pareto_values_gif_filename, writer='pillow', fps=2)
        plt.close()
        
        # ------ アニメーション作成（報酬） ------
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def update_rewards(frame):
            ax.clear()
            history = self.evaluation_history[frame]
            pareto_front_reward = history['pareto_front_reward']
            all_returns = history['all_returns']
            
            ax.scatter([ret[0] for ret in all_returns], [ret[1] for ret in all_returns], alpha=0.3, color='green')
            ax.scatter([ret[0] for ret in pareto_front_reward], [ret[1] for ret in pareto_front_reward], color='red', s=80)
            
            ax.set_title(f"Step {self.global_steps_at_evaluation[frame]}でのパレートフロント（報酬 - 最大化目的）")
            ax.set_xlabel("報酬1")
            ax.set_ylabel("報酬2")
            ax.set_xlim(x_range_reward)
            ax.set_ylim(y_range_reward)
            ax.grid(True)
            
            # 右上が最適な方向であることを示す矢印
            arrow_x_reward = x_min_reward + (x_max_reward - x_min_reward) * 0.15
            arrow_y_reward = y_min_reward + (y_max_reward - y_min_reward) * 0.15
            ax.annotate('最適方向', xy=(arrow_x_reward + (x_max_reward - x_min_reward) * 0.2, 
                                   arrow_y_reward + (y_max_reward - y_min_reward) * 0.2), 
                       xytext=(arrow_x_reward, arrow_y_reward),
                       arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                       fontsize=12)
        
        ani_rewards = FuncAnimation(fig, update_rewards, frames=len(self.evaluation_history), repeat=True)
        
        # 一意のIDを含むファイル名でGIFを保存
        pareto_rewards_gif_filename = f"{save_dir}/pareto_rewards_animation_{unique_id}.gif"
        ani_rewards.save(pareto_rewards_gif_filename, writer='pillow', fps=2)
        plt.close()
        
        print(f"評価履歴の可視化を保存しました:")
        print(f" - 実数値パレートフロント画像: {pareto_values_png_filename}")
        print(f" - 報酬パレートフロント画像: {pareto_rewards_png_filename}")
        print(f" - 実数値アニメーションGIF: {pareto_values_gif_filename}")
        print(f" - 報酬アニメーションGIF: {pareto_rewards_gif_filename}")

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

    def initialize_buffer_with_heuristics(self, env, num_episodes_per_pattern=5):
        """事前知識ベースの初期探索のためのバッファ初期化"""
        print("ヒューリスティックパターンによるバッファ初期化を開始...")
        
        # 初期化前にリプレイバッファを空にする
        self.experience_replay = []
        
        # 各パターンの実行比率（0と1の選択確率）
        patterns = [
            0.0,   # 常に0を選択（オンプレミス優先）
            1.0,   # 常に1を選択（クラウド優先）
            0.25,  # 25%の確率で1を選択
            0.5,   # 50%の確率で1を選択
            0.75   # 75%の確率で1を選択
        ]
        
        transitions_collected = 0
        episodes_collected = 0
        
        for p_idx, p in enumerate(patterns):
            print(f"パターン {p_idx+1}/5: 1の選択確率 = {p:.2f}")
            
            for ep in range(num_episodes_per_pattern):
                transitions = []
                obs = env.reset()
                done = False
                
                while not done:
                    # パターンに基づいて0または1のスカラー値を選択
                    action = 1 if self.np_random.random() < p else 0
                    
                    n_obs, reward, scheduled, wt_step, done = env.step(action)
                    
                    if done:
                        env.finalize_window_history()
                    
                    # Transitionオブジェクトを作成して保存
                    transitions.append(
                        Transition(
                            observation=obs,
                            action=action,
                            reward=np.float32(reward).copy(),
                            next_observation=n_obs,
                            terminal=done
                        )
                    )
                    
                    obs = n_obs
                    transitions_collected += 1
                
                # 報酬の計算
                for i in reversed(range(len(transitions) - 1)):
                    transitions[i].reward += self.gamma * transitions[i + 1].reward
                
                # ヒープを使って適切に追加
                if len(transitions) > 0:
                    # ヒープに追加するときは priority, step, transitions のタプルを使用
                    # ここでは優先度を高く（1.0）設定して確実に保持されるようにする
                    priority = float(1.0 + np.sum(transitions[0].reward) * 0.1)  # 報酬が高いほど優先度も高く
                    heapq.heappush(self.experience_replay, (priority, self.global_step, transitions))
                    episodes_collected += 1
                    
                    # グローバルステップを更新（学習の進行状況を正確に追跡するため）
                    self.global_step += len(transitions)
        
        # バッファ内のエピソードが十分かチェック
        if episodes_collected < 5:  # 最低でも5エピソードは必要
            print(f"警告: ヒューリスティック初期化で収集されたエピソード数が少なすぎます({episodes_collected})。" +
                  f"num_episodes_per_patternの値を大きくしてください。")
        
        print(f"ヒューリスティック初期化完了: {episodes_collected}エピソード、{transitions_collected}ステップのデータを収集")
        print(f"バッファサイズ: {len(self.experience_replay)}エピソード")
        
        # バッファの内容を確認（デバッグ用）
        if len(self.experience_replay) > 0:
            returns = [e[2][0].reward for e in self.experience_replay]
            avg_return = np.mean(returns, axis=0)
            print(f"バッファ内の平均報酬: {avg_return}")

    def extract_env_info(self, env: Optional[gym.Env]) -> None:
        """Extracts all the features of the environment: observation space, action space, ..."""
        if env is not None:
            self.env = env
            
            # 辞書型観測空間の処理
            if isinstance(self.env.unwrapped.observation_space, spaces.Dict):
                # 辞書型観測空間の場合
                self.observation_shape = None  # 形状は定義しない
                # 全サブスペースの要素数の合計を計算（より安全な方法）
                total_dim = 0
                for space in self.env.unwrapped.observation_space.spaces.values():
                    if hasattr(space, 'shape'):
                        total_dim += int(np.prod(space.shape))
                    elif hasattr(space, 'n'):
                        total_dim += space.n
                self.observation_dim = total_dim
                self.is_dict_obs = True
                
            # 離散型観測空間の処理
            elif isinstance(self.env.unwrapped.observation_space, spaces.Discrete):
                self.observation_shape = (1,)
                self.observation_dim = self.env.unwrapped.observation_space.n
                self.is_dict_obs = False
                
            # 連続型観測空間の処理
            else:
                self.observation_shape = self.env.unwrapped.observation_space.shape
                self.observation_dim = int(np.prod(self.observation_shape))
                self.is_dict_obs = False

            # 行動空間の処理
            self.action_space = env.unwrapped.action_space
            if isinstance(self.env.unwrapped.action_space, (spaces.Discrete, spaces.MultiBinary)):
                self.action_shape = (1,)
                self.action_dim = self.env.unwrapped.action_space.n
            else:
                self.action_shape = self.env.unwrapped.action_space.shape
                self.action_dim = int(np.prod(self.action_shape))
            
            # 報酬次元
            self.reward_dim = self.env.unwrapped.reward_space.shape[0]