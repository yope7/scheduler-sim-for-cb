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
import warnings

# CUDAが利用できない場合の警告を抑制
warnings.filterwarnings('ignore', message="Can't initialize NVML")
warnings.filterwarnings('ignore', message="torch.cuda.amp.GradScaler is enabled, but CUDA is not available")

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import wandb
# wandb.init(project="temp")

# 固定シードを削除して多様性を確保
# np.random.seed(42)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Linuxで使用可能なフォントを自動設定
def setup_linux_fonts():
    """Linuxで使用可能なフォントを自動設定"""
    import matplotlib.font_manager as fm
    import platform
    
    # Linuxシステムでのみ実行
    if platform.system() == 'Linux':
        # 利用可能なフォントを取得
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 優先順位付きでフォントを選択
        preferred_fonts = [
            'DejaVu Sans',
            'Liberation Sans',
            'Ubuntu',
            'Noto Sans CJK JP',
            'Noto Sans',
            'Arial',
            'Helvetica'
        ]
        
        # 利用可能なフォントから選択
        selected_font = None
        for font in preferred_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        # フォントが見つからない場合は、利用可能なフォントから最初のものを使用
        if selected_font is None and available_fonts:
            # sans-serif系のフォントを優先
            sans_serif_fonts = [f for f in available_fonts if 'sans' in f.lower() or 'sans' in f.lower()]
            if sans_serif_fonts:
                selected_font = sans_serif_fonts[0]
            else:
                selected_font = available_fonts[0]
        
        if selected_font:
            plt.rcParams['font.family'] = selected_font
            plt.rcParams['font.sans-serif'] = [selected_font]
            print(f"フォントを設定しました: {selected_font}")
        else:
            print("警告: 利用可能なフォントが見つかりませんでした")

# Linuxフォントを自動設定
# setup_linux_fonts()

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
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.scaling_factor = nn.Parameter(th.tensor(scaling_factor).float(), requires_grad=False)
        self.hidden_dim = hidden_dim

    def forward(self, state, desired_return, desired_horizon):
        """Return log-probabilities of actions or return action directly in case of continuous action space."""
        # 入力値の検証とクリッピング（NaN/Infを防ぐ）
        desired_return = th.clamp(desired_return, min=-1000.0, max=1000.0)
        desired_horizon = th.clamp(desired_horizon, min=0.0, max=1000.0)
        state = th.clamp(state.float(), min=-1000.0, max=1000.0)
        
        c = th.cat((desired_return, desired_horizon), dim=-1)
        c = c * self.scaling_factor
        
        # NaN/Infチェック
        if th.isnan(c).any() or th.isinf(c).any():
            print(f"[BasePCNModel] 警告: 条件ベクトルcにNaN/Infが含まれています")
            print(f"  desired_return範囲: min={desired_return.min()}, max={desired_return.max()}")
            print(f"  desired_horizon範囲: min={desired_horizon.min()}, max={desired_horizon.max()}")
            print(f"  scaling_factor: {self.scaling_factor}")
            c = th.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)

        s = self.s_emb(state)
        c = self.c_emb(c)
        
        # NaN/Infチェック
        if th.isnan(s).any() or th.isinf(s).any():
            print(f"[BasePCNModel] 警告: 状態埋め込みsにNaN/Infが含まれています")
            s = th.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        
        if th.isnan(c).any() or th.isinf(c).any():
            print(f"[BasePCNModel] 警告: 条件埋め込みcにNaN/Infが含まれています")
            c = th.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        
        prediction = self.fc(s * c)
        
        # NaN/Infチェック
        if th.isnan(prediction).any() or th.isinf(prediction).any():
            print(f"[BasePCNModel] 警告: 予測出力にNaN/Infが含まれています")
            print(f"  s範囲: min={s.min()}, max={s.max()}, mean={s.mean()}")
            print(f"  c範囲: min={c.min()}, max={c.max()}, mean={c.mean()}")
            print(f"  s*c範囲: min={(s*c).min()}, max={(s*c).max()}, mean={(s*c).mean()}")
            prediction = th.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        
        return prediction


class DiscreteActionsDefaultModel(BasePCNModel):
    """Model for the PCN with discrete actions."""

    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, scaling_factor: np.ndarray, hidden_dim: int):
        """Initialize the PCN model for discrete actions."""
        super().__init__(state_dim, action_dim, reward_dim, scaling_factor, hidden_dim)
        # TODO: 入力次元の最適化が必要
        # 現在: 205040次元 → 非常に重い
        # 提案: 特徴量選択や次元削減で1000-5000次元程度に削減
        self.state_dim = 76840  # ← この値が性能ボトルネック
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
        # 入力値の検証とクリッピング（NaN/Infを防ぐ）
        r = th.clamp(r, min=-1000.0, max=1000.0)
        if h is not None:
            h = th.clamp(h, min=0.0, max=1000.0)
        
        condition = self.condition_encoder(r)
        
        # NaN/Infチェック
        if th.isnan(condition).any() or th.isinf(condition).any():
            if self.debug_mode:
                print(f"[EnhancedPCNModel] 警告: condition_encoder出力にNaN/Infが含まれています")
                print(f"  r範囲: min={r.min()}, max={r.max()}, mean={r.mean()}")
            condition = th.nan_to_num(condition, nan=0.0, posinf=0.0, neginf=0.0)
        
        if h is not None:
            horizon_cond = self.horizon_encoder(h)
            
            # NaN/Infチェック
            if th.isnan(horizon_cond).any() or th.isinf(horizon_cond).any():
                if self.debug_mode:
                    print(f"[EnhancedPCNModel] 警告: horizon_encoder出力にNaN/Infが含まれています")
                    print(f"  h範囲: min={h.min()}, max={h.max()}, mean={h.mean()}")
                horizon_cond = th.nan_to_num(horizon_cond, nan=0.0, posinf=0.0, neginf=0.0)
            
            condition = condition * horizon_cond
            
            # NaN/Infチェック
            if th.isnan(condition).any() or th.isinf(condition).any():
                if self.debug_mode:
                    print(f"[EnhancedPCNModel] 警告: condition * horizon_condにNaN/Infが含まれています")
                condition = th.nan_to_num(condition, nan=0.0, posinf=0.0, neginf=0.0)
        
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
        
        # 入力値の検証とクリッピング（NaN/Infを防ぐ）
        x = th.clamp(x, min=-1000.0, max=1000.0)
        r = th.clamp(r, min=-1000.0, max=1000.0)
        if h is not None:
            h = th.clamp(h, min=0.0, max=1000.0)
        
        # 状態特徴抽出
        state_features = self.extract_state_features(x)
        
        # NaN/Infチェック
        if th.isnan(state_features).any() or th.isinf(state_features).any():
            if self.debug_mode:
                print(f"[EnhancedPCNModel] 警告: 状態特徴にNaN/Infが含まれています")
            state_features = th.nan_to_num(state_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 条件エンコーディング
        condition = self.encode_condition(r, h)
        
        if self.debug_mode:
            print(f"状態特徴形状: {state_features.shape}")
            print(f"条件形状: {condition.shape}")
        
        # 条件付き予測
        # PCNのキーとなる部分: 状態特徴と条件の要素積
        conditioned_features = state_features * condition
        
        # NaN/Infチェック
        if th.isnan(conditioned_features).any() or th.isinf(conditioned_features).any():
            if self.debug_mode:
                print(f"[EnhancedPCNModel] 警告: 条件付き特徴にNaN/Infが含まれています")
                print(f"  state_features範囲: min={state_features.min()}, max={state_features.max()}, mean={state_features.mean()}")
                print(f"  condition範囲: min={condition.min()}, max={condition.max()}, mean={condition.mean()}")
            conditioned_features = th.nan_to_num(conditioned_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.debug_mode:
            print(f"条件付き特徴形状: {conditioned_features.shape}")
        
        # 方策と価値予測
        pi = self.pi_net(conditioned_features)
        v = self.v_net(conditioned_features)
        
        # NaN/Infチェック
        if th.isnan(pi).any() or th.isinf(pi).any():
            if self.debug_mode:
                print(f"[EnhancedPCNModel] 警告: π出力にNaN/Infが含まれています")
            pi = th.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
        
        if th.isnan(v).any() or th.isinf(v).any():
            if self.debug_mode:
                print(f"[EnhancedPCNModel] 警告: V出力にNaN/Infが含まれています")
            v = th.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        
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
        
        # 混合精度学習の設定
        # CUDAが確実に存在する前提でAMPを有効化
        self.use_amp = True
        self.scaler = th.cuda.amp.GradScaler()
        
        # パフォーマンス監視
        self.update_times = []
        self.last_update_time = None

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

    def update(self, learning_rate=None):
        """Update PCN model - 最適化版
        
        Args:
            learning_rate (float, optional): 一時的な学習率。Noneの場合はデフォルトの学習率を使用
        """
        import time
        start_time = time.time()
        
        # 一時的な学習率の適用
        original_lr = None
        if learning_rate is not None:
            original_lr = self.opt.param_groups[0]['lr']
            self.opt.param_groups[0]['lr'] = learning_rate
        
        # 1. 事前にnumpy配列を準備（メモリアロケーション削減）
        batch_size = self.batch_size
        buffer_size = len(self.experience_replay)
        
        # 2. ベクトル化されたサンプリング
        sample_indices = self.np_random.choice(buffer_size, size=batch_size, replace=True)
        
        # 3. 事前に配列を確保（メモリアロケーション削減）
        obs_shape = self.experience_replay[0][2][0].observation.shape
        reward_shape = self.experience_replay[0][2][0].reward.shape
        
        # 4. 効率的なデータ抽出（メモリ効率化）
        observations = np.empty((batch_size,) + obs_shape, dtype=np.float32)
        actions = np.empty(batch_size, dtype=np.int64)
        desired_returns = np.empty((batch_size,) + reward_shape, dtype=np.float32)
        desired_horizons = np.empty(batch_size, dtype=np.float32)
        
        # 5. ベクトル化されたデータ抽出（PCNの正しい学習方法）
        for i, idx in enumerate(sample_indices):
            episode = self.experience_replay[idx][2]
            episode_length = len(episode)
            
            # エピソード内のランダムなステップを選択（各ステップで学習）
            t = np.random.randint(0, episode_length)
            transition = episode[t]
            
            # コピーを避けて直接代入
            obs_data = transition.observation
            # NaN/Infチェック
            if np.any(np.isnan(obs_data)) or np.any(np.isinf(obs_data)):
                if self.debug_mode:
                    print(f"[PCN] 警告: エピソード {idx}, ステップ {t} の観測にNaN/Infが含まれています")
                    print(f"  観測範囲: min={np.min(obs_data)}, max={np.max(obs_data)}")
                # NaN/Infの場合は0に置き換え
                obs_data = np.nan_to_num(obs_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            observations[i] = obs_data
            actions[i] = transition.action
            
            # 論文に厳密に従った累積報酬計算: R_t = Σ_{i=t}^T γ^i r_i
            remaining_return = np.zeros(reward_shape, dtype=np.float32)
            for j in range(t, episode_length):
                # 論文の式: R_t = Σ_{i=t}^T γ^i r_i
                # ここで episode[j].reward は即時報酬 r_j
                reward = episode[j].reward
                # NaN/Infチェック
                if np.any(np.isnan(reward)) or np.any(np.isinf(reward)):
                    if self.debug_mode:
                        print(f"[PCN] 警告: エピソード {idx}, ステップ {j} の報酬にNaN/Infが含まれています")
                        print(f"  報酬: {reward}")
                    # NaN/Infの場合は0に置き換え
                    reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
                
                remaining_return += (self.gamma ** (j - t)) * reward
            
            # NaN/Infチェックと値のクリッピング
            if np.any(np.isnan(remaining_return)) or np.any(np.isinf(remaining_return)):
                if self.debug_mode:
                    print(f"[PCN] 警告: 累積報酬にNaN/Infが含まれています")
                    print(f"  累積報酬: {remaining_return}")
                # NaN/Infの場合は0に置き換え
                remaining_return = np.nan_to_num(remaining_return, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 値の範囲をクリッピング（異常に大きい値を防ぐ）
            # 累積報酬が異常に大きい場合、モデルの入力が不安定になる可能性がある
            # 注意: scaling_factorが[1, 1, 1]の場合、desired_returnが大きすぎると
            # モデルの内部で数値的不安定性が発生する可能性がある
            # より小さな範囲にクリッピング（-1000から1000の範囲）
            clip_value = 1000.0  # 最大値を1000に制限（scaling_factorを考慮）
            remaining_return = np.clip(remaining_return, -clip_value, clip_value)
            
            if self.debug_mode and (np.abs(remaining_return).max() > 500):
                print(f"[PCN] 警告: 累積報酬が大きすぎます: {remaining_return}")
                print(f"  エピソード長: {episode_length}, 開始ステップ: {t}")
                print(f"  クリッピング後の値: {np.clip(remaining_return, -clip_value, clip_value)}")
            
            desired_returns[i] = remaining_return  # 論文通りの割引累積報酬
            desired_horizons[i] = np.float32(episode_length - t)  # 残りのステップ数
        
        # 6. 一括GPU転送（非同期化 + メモリ効率化）
        # desired_returnの値が異常に大きい場合、モデルの入力が不安定になる可能性があるため、
        # ここでもクリッピングを確認
        # 注意: scaling_factorが[1, 1, 1]の場合、desired_returnが大きすぎると
        # モデルの内部で数値的不安定性が発生する可能性がある
        # より小さな範囲にクリッピング（-1000から1000の範囲）
        if np.any(np.abs(desired_returns) > 1000.0):
            if self.debug_mode:
                print(f"[PCN] 警告: desired_returnsに異常に大きい値が含まれています")
                print(f"  min={np.min(desired_returns)}, max={np.max(desired_returns)}, mean={np.mean(desired_returns)}")
            # クリッピングを再適用（念のため）
            desired_returns = np.clip(desired_returns, -1000.0, 1000.0)
        
        with th.cuda.amp.autocast(enabled=self.use_amp):  # 混合精度学習
            # 非同期転送でCPU-GPU並列化
            obs = th.from_numpy(observations).to(self.device, non_blocking=True)
            actions = th.from_numpy(actions).to(self.device, non_blocking=True)
            desired_return = th.from_numpy(desired_returns).to(self.device, non_blocking=True)
            desired_horizon = th.from_numpy(desired_horizons).to(self.device, non_blocking=True).unsqueeze(1)
            
            # desired_returnの値を正規化（異常に大きい値を防ぐ）
            # モデルの入力が不安定になるのを防ぐため、値を適切な範囲にクリッピング
            # 注意: scaling_factorが[1, 1, 1]の場合、desired_returnが大きすぎると
            # モデルの内部で数値的不安定性が発生する可能性がある
            # より小さな範囲にクリッピング（-1000から1000の範囲）
            desired_return = th.clamp(desired_return, min=-1000.0, max=1000.0)
            
            # desired_horizonもクリッピング（異常に大きい値を防ぐ）
            desired_horizon = th.clamp(desired_horizon, min=0.0, max=1000.0)
            
            # 観測データの値もクリッピング（異常に大きい値を防ぐ）
            obs = th.clamp(obs, min=-1000.0, max=1000.0)
            
            # 7. 最適化された勾配計算
            self.opt.zero_grad(set_to_none=True)
            
            # 8. モデル推論前のデータ検証
            if th.isnan(obs).any() or th.isinf(obs).any():
                print(f"[PCN] 警告: 観測データにNaN/Infが含まれています")
                print(f"  NaN: {th.isnan(obs).any()}, Inf: {th.isinf(obs).any()}")
                print(f"  観測データ範囲: min={obs.min()}, max={obs.max()}, mean={obs.mean()}")
            
            if th.isnan(desired_return).any() or th.isinf(desired_return).any():
                print(f"[PCN] 警告: desired_returnにNaN/Infが含まれています")
                print(f"  NaN: {th.isnan(desired_return).any()}, Inf: {th.isinf(desired_return).any()}")
                print(f"  desired_return範囲: min={desired_return.min()}, max={desired_return.max()}, mean={desired_return.mean()}")
            
            if th.isnan(desired_horizon).any() or th.isinf(desired_horizon).any():
                print(f"[PCN] 警告: desired_horizonにNaN/Infが含まれています")
                print(f"  NaN: {th.isnan(desired_horizon).any()}, Inf: {th.isinf(desired_horizon).any()}")
                print(f"  desired_horizon範囲: min={desired_horizon.min()}, max={desired_horizon.max()}, mean={desired_horizon.mean()}")
            
            # モデル推論
            try:
                if self.use_enhanced_model:
                    prediction_output = self.network(obs, desired_return, desired_horizon)
                else:
                    prediction_output = self.model(obs, desired_return, desired_horizon)
            except Exception as e:
                print(f"[PCN] エラー: モデル推論中にエラーが発生しました: {e}")
                print(f"  観測データ統計: min={obs.min()}, max={obs.max()}, mean={obs.mean()}")
                print(f"  desired_return統計: min={desired_return.min()}, max={desired_return.max()}, mean={desired_return.mean()}")
                print(f"  desired_horizon統計: min={desired_horizon.min()}, max={desired_horizon.max()}, mean={desired_horizon.mean()}")
                import traceback
                traceback.print_exc()
                # エラーの場合は損失を0に設定してスキップ
                l = th.tensor(0.0, device=self.device, requires_grad=False)
                return l, {}
            
            # 9. 損失計算
            if isinstance(prediction_output, tuple):
                prediction_logits = prediction_output[0]
            else:
                prediction_logits = prediction_output
            
            # モデル出力の検証（NaN/Infチェック）
            if th.isnan(prediction_logits).any() or th.isinf(prediction_logits).any():
                print(f"[PCN] 警告: モデル出力にNaN/Infが含まれています")
                print(f"  NaN: {th.isnan(prediction_logits).any()}, Inf: {th.isinf(prediction_logits).any()}")
                print(f"  prediction_logits範囲: min={prediction_logits.min()}, max={prediction_logits.max()}")
                print(f"  観測データ統計: min={obs.min()}, max={obs.max()}, mean={obs.mean()}")
                print(f"  desired_return統計: min={desired_return.min()}, max={desired_return.max()}, mean={desired_return.mean()}")
                print(f"  desired_horizon統計: min={desired_horizon.min()}, max={desired_horizon.max()}, mean={desired_horizon.mean()}")
                
                # モデルの重みを確認
                if self.use_enhanced_model:
                    for name, param in self.network.named_parameters():
                        if th.isnan(param).any() or th.isinf(param).any():
                            print(f"  モデル重み {name} にNaN/Infが含まれています")
                            print(f"    範囲: min={param.min()}, max={param.max()}, mean={param.mean()}")
                else:
                    for name, param in self.model.named_parameters():
                        if th.isnan(param).any() or th.isinf(param).any():
                            print(f"  モデル重み {name} にNaN/Infが含まれています")
                            print(f"    範囲: min={param.min()}, max={param.max()}, mean={param.mean()}")
                print(f"  観測データ統計: min={obs.min()}, max={obs.max()}, mean={obs.mean()}")
                print(f"  desired_return統計: min={desired_return.min()}, max={desired_return.max()}, mean={desired_return.mean()}")
                print(f"  行動統計: {th.bincount(actions.long())}")
                # モデル出力がNaNの場合は、損失を0に設定してスキップ（勾配を計算しない）
                l = th.tensor(0.0, device=self.device, requires_grad=False)
                # 勾配を計算しないため、最適化をスキップ
                return l, {}
            
            if self.continuous_action:
                l = F.mse_loss(actions.float(), prediction_logits)
            else:
                if self.use_enhanced_model:
                    l = F.cross_entropy(prediction_logits, actions.long())
                else:
                    l = F.nll_loss(prediction_logits, actions.long())
            
            # 損失の検証
            if th.isnan(l) or th.isinf(l):
                print(f"[PCN] エラー: 損失がNaN/Infになりました")
                print(f"  損失値: {l.item()}")
                print(f"  観測データ統計: min={obs.min()}, max={obs.max()}, mean={obs.mean()}")
                print(f"  desired_return統計: min={desired_return.min()}, max={desired_return.max()}, mean={desired_return.mean()}")
                print(f"  prediction_logits統計: min={prediction_logits.min()}, max={prediction_logits.max()}, mean={prediction_logits.mean()}")
                print(f"  行動統計: {th.bincount(actions.long())}")
                # NaNの場合は損失を0に設定してスキップ（勾配を計算しない）
                l = th.tensor(0.0, device=self.device, requires_grad=False)
                # 勾配を計算しないため、最適化をスキップ
                return l, {}
            
            # 10. 逆伝播と最適化（混合精度対応）
            if self.use_amp and self.scaler is not None:
                # GradScalerの正しい使用方法:
                # 1. scale(loss).backward() - スケールされた損失で逆伝播
                # 2. unscale_(optimizer) - 勾配をunscale（Inf/NaNチェックも行う）
                # 3. clip_grad_norm_ - 勾配クリッピング（オプション）
                # 4. step(optimizer) - オプティマイザのステップ（unscale_()の後に必ず呼ぶ必要がある）
                # 5. update() - スケーラーの更新
                self.scaler.scale(l).backward()
                
                # unscale_()を呼んで勾配をunscale（Inf/NaNチェックも行う）
                # unscale_()が成功した場合、必ずstep()を呼ぶ必要がある
                try:
                    self.scaler.unscale_(self.opt)
                    # 勾配クリッピングを追加（勾配爆発を防ぐ）
                    th.nn.utils.clip_grad_norm_(self.network.parameters() if self.use_enhanced_model else self.model.parameters(), max_norm=1.0)
                    # unscale_()が成功した場合、必ずstep()を呼ぶ必要がある
                    self.scaler.step(self.opt)
                    self.scaler.update()
                except RuntimeError as e:
                    # unscale_()が既に呼ばれている場合、またはその他のエラーの場合
                    if "unscale_() has already been called" in str(e):
                        if self.debug_mode:
                            print(f"[PCN] 警告: unscale_()が既に呼ばれています。通常の最適化を実行します。")
                        # 勾配クリッピングのみ実行
                        th.nn.utils.clip_grad_norm_(self.network.parameters() if self.use_enhanced_model else self.model.parameters(), max_norm=1.0)
                        # 通常の最適化を実行（GradScalerを使わない）
                        self.opt.step()
                    elif "No inf checks were recorded" in str(e):
                        if self.debug_mode:
                            print(f"[PCN] 警告: Infチェックが記録されていません。通常の最適化を実行します。")
                        # 通常の最適化を実行（GradScalerを使わない）
                        self.opt.step()
                    else:
                        # その他のエラーの場合は再発生
                        raise
            else:
                l.backward()
                # 勾配クリッピングを追加（勾配爆発を防ぐ）
                th.nn.utils.clip_grad_norm_(self.network.parameters() if self.use_enhanced_model else self.model.parameters(), max_norm=1.0)
                self.opt.step()
        
        # 11. メモリクリーンアップ
        del observations, actions, desired_returns, desired_horizons
        if self.device == 'cuda':
            th.cuda.empty_cache()
        
        # 12. パフォーマンス監視
        end_time = time.time()
        update_time = end_time - start_time
        self.update_times.append(update_time)
        
        # 学習率を元に戻す
        if original_lr is not None:
            self.opt.param_groups[0]['lr'] = original_lr
        
        # 詳細なデバッグ情報（100回ごと）
        if len(self.update_times) % 100 == 0 and self.debug_mode:
            avg_time = np.mean(self.update_times[-100:])
            print(f"Update performance: {avg_time:.4f}s per update (avg of last 100)")
            
            # 学習データの統計を表示
            print(f"学習データ統計:")
            print(f"  バッチサイズ: {batch_size}")
            print(f"  バッファサイズ: {buffer_size}")
            print(f"  観測形状: {obs_shape}")
            print(f"  報酬形状: {reward_shape}")
            print(f"  割引率γ: {self.gamma}")
            
            # 行動分布の詳細
            action_counts = np.bincount(actions.cpu().numpy())
            total_actions = len(actions)
            print(f"  行動分布: {action_counts}")
            for i, count in enumerate(action_counts):
                if count > 0:
                    percentage = (count / total_actions) * 100
                    print(f"    行動{i}: {count}回 ({percentage:.1f}%)")
            
            # 報酬とホライズンの詳細統計（論文通りの計算）
            print(f"  論文通りの割引累積報酬統計:")
            print(f"    平均: {desired_returns.mean():.4f}")
            print(f"    標準偏差: {desired_returns.std():.4f}")
            print(f"    範囲: [{desired_returns.min():.4f}, {desired_returns.max():.4f}]")
            
            print(f"  残りステップ数統計:")
            print(f"    平均: {desired_horizons.mean():.1f}")
            print(f"    標準偏差: {desired_horizons.std():.1f}")
            print(f"    範囲: [{desired_horizons.min()}, {desired_horizons.max()}]")
            
            # 学習の質を評価
            if len(self.update_times) >= 200:
                recent_losses = [self.update_times[-100:]]
                loss_trend = np.mean(recent_losses[-50:]) - np.mean(recent_losses[:50])
                if loss_trend > 0:
                    print(f"  ⚠️  損失の傾向: 増加傾向 ({loss_trend:.4f})")
                else:
                    print(f"  ✓ 損失の傾向: 減少傾向 ({loss_trend:.4f})")
            
            # 500回ごとに学習データをファイルに保存
            if len(self.update_times) % 500 == 0:
                print("📊 学習データをファイルに保存中...")
                try:
                    self.save_learning_data_to_file(
                        filename=f"learning_data_update_{len(self.update_times)}.txt",
                        sample_size=200
                    )
                except Exception as e:
                    print(f"⚠️  ファイル保存中にエラーが発生しました: {e}")
        
        return l, prediction_logits

    def _add_episode(self, transitions: List[Transition], max_size: int, step: int) -> None:
        # compute return
        for i in reversed(range(len(transitions) - 1)):
            transitions[i].reward += self.gamma * transitions[i + 1].reward
        # pop smallest episode of heap if full, add new episode
        # heap is sorted by negative distance, (updated in nlargest)
        # put positive number to ensure that new item stays in the heap
        unique_step = (step, id(transitions))
        if len(self.experience_replay) == max_size:
            heapq.heappushpop(self.experience_replay, (1, unique_step, transitions))
        else:
            heapq.heappush(self.experience_replay, (1, unique_step, transitions))

    def _nlargest(self, n, threshold=0.1):
        """経験再生バッファから上位n個のエピソードを取得"""
        if len(self.experience_replay) == 0:
            print("警告: 経験再生バッファが空です。")
            return []
        
        # 全てのエピソードの報酬を取得
        all_returns = []
        all_episodes = []
        
        for priority, step, episode in self.experience_replay:
            if len(episode) > 0:
                return_val = episode[0].reward
                all_returns.append(return_val)
                all_episodes.append((priority, step, episode))
        
        if len(all_returns) == 0:
            print("警告: 有効なエピソードが見つかりません。")
            return []
        
        # 非支配解のインデックスを取得
        returns_array = np.array(all_returns)
        non_dominated_inds = get_non_dominated_inds(returns_array)
        
        # print(f"\n=== PCNエージェント: _nlargestメソッド実行 ===")
        # print(f"経験再生バッファサイズ: {len(self.experience_replay)}")
        # print(f"有効なエピソード数: {len(all_returns)}")
        # print(f"非支配解の数: {len(non_dominated_inds)}")
        
        if len(non_dominated_inds) > 0:
            # 非支配解の報酬範囲を表示
            nd_returns = returns_array[non_dominated_inds]
            # print(f"非支配解の報酬範囲:")
            # for dim in range(nd_returns.shape[1]):
            #     min_val = np.min(nd_returns[:, dim])
            #     max_val = np.max(nd_returns[:, dim])
            #     print(f"  次元{dim}: {min_val:.4f} 〜 {max_val:.4f}")
        
        # 非支配解のみを使用
        nd_episodes = [all_episodes[i] for i in non_dominated_inds]
        nd_returns = returns_array[non_dominated_inds]
        
        if len(nd_episodes) == 0:
            print("非支配解が見つかりませんでした。全てのエピソードを使用します。")
            # 非支配解がない場合は、全てのエピソードを使用
            nd_episodes = all_episodes
            nd_returns = returns_array
        
        # Crowding Distanceを計算
        if len(nd_returns) > 1:
            crowding_distances = crowding_distance(nd_returns)
        else:
            crowding_distances = np.array([1.0])
        
        # 端点を識別（パレートフロントの端点）
        endpoints = []
        for dim in range(nd_returns.shape[1]):
            min_idx = np.argmin(nd_returns[:, dim])
            max_idx = np.argmax(nd_returns[:, dim])
            endpoints.extend([min_idx, max_idx])
        endpoints = list(set(endpoints))  # 重複を除去
        
        # 端点のペナルティを大幅に強化（10倍のペナルティ）
        for idx in endpoints:
            crowding_distances[idx] *= 1  # 10分の1にすることで大幅なペナルティ
        
        # さらに、最近選択された端点を追跡して追加ペナルティを適用
        if not hasattr(self, '_recently_selected_endpoints'):
            self._recently_selected_endpoints = []
        
        # 最近選択された端点に追加ペナルティ
        for idx in self._recently_selected_endpoints:
            if idx < len(crowding_distances):
                crowding_distances[idx] *= 0.05  # さらに20分の1のペナルティ
        
        # 距離ベースの優先度を計算
        distances = []
        for i, (priority, step, episode) in enumerate(nd_episodes):
            # 非支配解からの距離を計算
            dist = np.min(np.linalg.norm(nd_returns - nd_returns[i], axis=1))
            # Crowding Distanceと組み合わせ
            combined_score = crowding_distances[i] / (dist + 1e-8)
            distances.append(combined_score)
        
        distances = np.array(distances)
        
        # 上位n個を選択
        n = min(n, len(nd_episodes))
        top_indices = np.argsort(distances)[-n:]
        
        # 選択された端点を記録
        selected_endpoints = []
        for idx in top_indices:
            if idx in endpoints:
                selected_endpoints.append(idx)
        
        # 最近選択された端点リストを更新（最新の5個を保持）
        self._recently_selected_endpoints = selected_endpoints[:5]
        
        # print(f"選択されたエピソード数: {n}")
        # print("選択されたエピソードの詳細:")
        for i, idx in enumerate(top_indices):
            episode = nd_episodes[idx]
            return_val = nd_returns[idx]
            crowding_dist = crowding_distances[idx]
            distance_score = distances[idx]
            is_endpoint = "端点" if idx in endpoints else "非端点"
            # print(f"  {i+1}. 報酬: {return_val}, Crowding Distance: {crowding_dist:.4f}, 距離スコア: {distance_score:.4f} ({is_endpoint})")
        
        # print("="*60)
        
        # 結果の表示
        sorted_i = np.argsort(distances)
        largest = [nd_episodes[i] for i in sorted_i[-n:]]
        # print(f"\n選択された {n} 個の解の内訳:")
        selected_returns = np.array([e[2][0].reward for e in largest])
        selected_non_dom = get_non_dominated_inds(selected_returns)
        # print(f"  - 選択解のうち非支配解の数: {len(selected_non_dom)}")
        # print(f"  - 選択解の優先度範囲: {np.min([e[0] for e in largest]):.4f} 〜 {np.max([e[0] for e in largest]):.4f}")
        # print("==================================\n")
        
        # ヒープの更新処理...
        
        return largest

    def _choose_commands(self, num_episodes: int):
        """探索方向を決定するメソッド - 論文に沿った修正版"""
        episodes = self._nlargest(num_episodes)
        
        if len(episodes) == 0:
            print("警告: コマンド選択用のエピソードが見つかりませんでした。デフォルト値を返します。")
            return np.zeros(self.reward_dim, dtype=np.float32), np.float32(40)
        
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
            
            # 端点を識別
            endpoints = []
            for dim in range(returns.shape[1]):
                min_idx = np.argmin(returns[:, dim])
                max_idx = np.argmax(returns[:, dim])
                endpoints.extend([min_idx, max_idx])
            endpoints = list(set(endpoints))
            
            # 端点を避け、中央付近の解を優先（より強力なペナルティ）
            center_dists = np.linalg.norm(normalized_returns - 0.5, axis=1)
            center_weights = 1.0 / (center_dists + 0.1)  # 中央に近いほど重み大きく
            
            # 端点に対する大幅なペナルティ
            for idx in endpoints:
                center_weights[idx] *= 0.1  # 100分の1のペナルティ
            
            # 最近選択された端点を追跡して追加ペナルティ
            if not hasattr(self, '_recently_chosen_endpoints'):
                self._recently_chosen_endpoints = []
            
            # 確率的サンプリング（中央寄りの点が選ばれやすい）
            weights = center_weights / np.sum(center_weights)
            r_i = self.np_random.choice(len(returns), p=weights)
            
            # 選択された端点を記録
            if r_i in endpoints:
                self._recently_chosen_endpoints = [r_i] + self._recently_chosen_endpoints[:4]  # 最新5個を保持
            
            # 元の報酬を保存
            original_return = returns[r_i].copy()
            original_horizon = np.float32(horizons[r_i] - 2)
            
            # 論文に沿った目標改善量の設定
            # 1. 目的関数値を[0,1]の範囲に正規化
            normalized_current = (original_return - returns.min(axis=0)) / (returns.max(axis=0) - returns.min(axis=0) + 1e-8)
            
            # 2. 一様分布から改善ベクトルδ ∈ [0,1]^mをサンプリング
            delta = self.np_random.uniform(0, 1, size=self.reward_dim)
            
            # 3. 現在の解xにδを加えて目標x + δを作成（ただし[0,1]の範囲にクリップ）
            target_normalized = np.clip(normalized_current + delta, 0, 1)
            
            # 4. 正規化された目標値を元のスケールに戻す
            desired_return = target_normalized * (returns.max(axis=0) - returns.min(axis=0)) + returns.min(axis=0)
            desired_horizon = original_horizon
            
            # 詳細なログ出力（デバッグ用）
            # print(f"\n=== PCNエージェント: 論文に沿った目標値設定 ===")
            # print(f"選択された元の報酬: {original_return}")
            # print(f"正規化された現在値: {normalized_current}")
            # print(f"サンプリングされた改善ベクトルδ: {delta}")
            # print(f"正規化された目標値: {target_normalized}")
            # print(f"最終的な目標報酬: {desired_return}")
            # print(f"目標ホライズン: {desired_horizon}")
            # print("="*60)
            
            return np.float32(desired_return), desired_horizon
        else:
            # 非支配解がない場合のフォールバック
            print(f"\n=== PCNエージェント: フォールバック ===")
            print("非支配解が見つかりませんでした。デフォルト値を設定します。")
            print("デフォルト目標報酬: [0, 0, 0]")
            print("デフォルト目標ホライズン: 40")
            print("="*60)
            return np.zeros(self.reward_dim, dtype=np.float32), np.float32(40)

    def _act(self, obs: np.ndarray, desired_return, desired_horizon, eval_mode=False) -> int:
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
        # 注意: 累積報酬の計算は_add_episodeメソッドで行われるため、ここでは行わない
        episode_return = transitions[0].reward  # 即座の報酬（累積報酬ではない）
        
        final_return = episode_return
        onpre_final = env.on_premise_window_history_full
        cloud_final = env.cloud_window_history_full
        value_cost, value_wt, value_avg_wt = env.calc_objective_values()
        
        # エピソード完了時の結果を表示
        if eval_mode:
            pass
            # print(f"エピソード完了: 長さ={len(transitions)}")
            # print(f"  累積報酬: {final_return}")
            # print(f"  実際のコスト: {value_cost}, 実際の待ち時間: {value_wt}")
            
            # # actionsetを表示
            # actions = [t.action for t in transitions]
            # print(f"  actionset: {actions}")
            
            # # マップ情報を表示
            # # print(f"  オンプレミスマップ: {onpre_final}")
            # # print(f"  クラウドマップ: {cloud_final}")
            # print("------------------------")
        
        # print(f"エピソード完了: 長さ={len(transitions)}")
        # print(f"最終報酬: {final_return}")
        # print(f"実際の値: コスト={value_cost}, 待ち時間={value_wt}")
        # print("=========================\n")
        
        return transitions, map_snapshots_on_premise, map_snapshots_cloud, wt_sum, [onpre_final, cloud_final], [value_cost, value_avg_wt]

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
        self.run_episode(env, best_policy, max_return=np.full(np.inf, np.inf, dtype=np.float32), eval_mode=True)
    
    def evaluate_and_execute_selected_policy(self, env, max_return, objective_index, n=10):
        """特定の目的関数の値が最大となるようなポリシーを評価して実行する"""
        n = min(n, len(self.experience_replay))
        # print("len(self.experience_replay)", len(self.experience_replay))
        episodes = self._nlargest(n)
        
        # 実際に取得されたエピソード数に基づいてnを調整
        actual_n = len(episodes)
        if actual_n == 0:
            print("警告: 評価用のエピソードが見つかりませんでした。")
            return None, [0, 0, 0]
        
        returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
        returns = np.float32(returns)
        horizons = np.float32(horizons)
        all_transitions = []
        e_returns = []
        
        # print(f"\n===== {actual_n}個のエピソード評価結果（選択実行版）=====")
        
        for i in range(actual_n):
            transitions, _, _, _, map_fin, value = self._run_episode(env, returns[i], np.float32(horizons[i]), max_return, eval_mode=True)
            all_transitions.append(transitions)
            # compute return
            for j in reversed(range(len(transitions) - 1)):
                transitions[j].reward += self.gamma * transitions[j + 1].reward
            e_returns.append(transitions[0].reward)
            
            # 各エピソードの結果を表示
            # print(f"エピソード {i+1}:")
            # print(f"  累積報酬: {transitions[0].reward}")
            # print(f"  実際のコスト: {value[0]}, 実際の待ち時間: {value[1]}")
            
            # actionsetを表示
            # actions = [t.action for t in transitions]
            # print(f"  actionset: {actions}")
            
            # マップ情報を表示
            # print(f"  オンプレミスマップ: {map_fin[0]}")
            # print(f"  クラウドマップ: {map_fin[1]}")
            # print()
            #やってみて、再現可能なデータを集める。

        # print("==========================================\n")

        # 非支配解の取得
        # print("e_returns", e_returns)
        e_returns_np = np.array(e_returns, dtype=np.float64)
        non_dominated_inds = get_non_dominated_inds(e_returns_np)
        pareto_front = e_returns_np[non_dominated_inds]
        if self.log:
            wandb.log({"pareto_front_eval_and_execute": wandb.Table(data=pareto_front, columns=["Objective1", "Objective2"])})


        # objective_indexが範囲内かチェック
        if objective_index >= len(e_returns[0]) if e_returns else 0:
            print(f"警告: objective_index {objective_index} が範囲外です。デフォルト値0を使用します。")
            objective_index = 0
        
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
        
        # 実際に取得されたエピソード数に基づいてnを調整
        actual_n = len(episodes)
        if actual_n == 0:
            print("警告: 評価用のエピソードが見つかりませんでした。")
            return [], [], [], None
        
        returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
        returns = np.float32(returns)
        horizons = np.float32(horizons)
        e_returns = []
        e_values = []
        all_transitions = []  # 全てのtransitionsを保存するリスト
        
        # print(f"\n===== {actual_n}個のエピソード評価結果 =====")
        
        for i in range(actual_n):
            transitions, _, _, _, map_fin, value = self._run_episode(env, returns[i], np.float32(horizons[i]), max_return, eval_mode=True)
            
            # 累積報酬を計算（表示用のみ）
            transitions_copy = []
            for t in transitions:
                transitions_copy.append(Transition(
                    observation=t.observation,
                    action=t.action,
                    reward=np.array(t.reward, copy=True),
                    next_observation=t.next_observation,
                    terminal=t.terminal
                ))
            
            for j in reversed(range(len(transitions_copy) - 1)):
                transitions_copy[j].reward += self.gamma * transitions_copy[j + 1].reward
            
            e_returns.append(transitions_copy[0].reward)
            e_values.append(value)
            all_transitions.append(transitions)  # 元のtransitionsを保存
            
            # 各エピソードの結果を表示
            # print(f"エピソード {i+1}:")
            # print(f"  累積報酬: {transitions[0].reward}")
            # print(f"  実際のコスト: {value[0]}, 実際の待ち時間: {value[1]}")
            
            # actionsetを表示
            # actions = [t.action for t in transitions]
            # print(f"  actionset: {actions}")
            # print()
        
        # print("===============================\n")
        


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
        
        return e_returns, e_values, distances, map_fin

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
            max_return = max_return if max_return is not None else np.full(self.reward_dim, np.inf, dtype=np.float32)

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
                            
                        # より多様なランダムアクションを生成
                        # 各エピソードで異なるシードを使用
                        episode_seed = int(time.time() * 1000) + episode_idx + self.global_step
                        np.random.seed(episode_seed)
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
                    
                    # 累積報酬を計算（表示用のみ）
                    transitions_copy = []
                    for t in transitions:
                        transitions_copy.append(Transition(
                            observation=t.observation,
                            action=t.action,
                            reward=np.array(t.reward, copy=True),
                            next_observation=t.next_observation,
                            terminal=t.terminal
                        ))
                    
                    for j in reversed(range(len(transitions_copy) - 1)):
                        transitions_copy[j].reward += self.gamma * transitions_copy[j + 1].reward
                    
                    returns.append(transitions_copy[0].reward)
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
        """評価履歴を可視化し、固定ファイル名で上書き保存。
        報酬（最大化目的）と実数値（最小化目的）の両方のグラフを別々に表示する。
        """
        if not self.evaluation_history:
            print("評価履歴がありません")
            return
        
        # ディレクトリ作成
        os.makedirs(save_dir, exist_ok=True)
        
        # 固定ファイル名を使用（更新時に上書き）
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
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
        
        # 固定ファイル名で保存（上書き）
        pareto_values_png_filename = f"{save_dir}/pareto_values_evolution.png"
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
        
        # 固定ファイル名で保存（上書き）
        pareto_rewards_png_filename = f"{save_dir}/pareto_rewards_evolution.png"
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
        
        # 固定ファイル名でGIFを保存（上書き）
        pareto_values_gif_filename = f"{save_dir}/pareto_values_animation.gif"
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
        
        # 固定ファイル名でGIFを保存（上書き）
        pareto_rewards_gif_filename = f"{save_dir}/pareto_rewards_animation.gif"
        ani_rewards.save(pareto_rewards_gif_filename, writer='pillow', fps=2)
        plt.close()
        
        print(f"評価履歴の可視化を保存しました（最終更新: {timestamp}）:")
        print(f" - 実数値パレートフロント画像: {pareto_values_png_filename}")
        print(f" - 報酬パレートフロント画像: {pareto_rewards_png_filename}")
        print(f" - 実数値アニメーションGIF: {pareto_values_gif_filename}")
        print(f" - 報酬アニメーションGIF: {pareto_rewards_gif_filename}")

    def save_pareto_solutions_to_txt(self, mode_name="default"):
        """パレートフロントの解をテキストファイルに保存（単一ファイルで更新）"""
        if not self.evaluation_history:
            print("評価履歴がありません。ファイルは作成されませんでした。")
            return
        
        # 保存ディレクトリの作成
        save_dir = "pareto_solutions"
        os.makedirs(save_dir, exist_ok=True)
        
        # 固定ファイル名を使用（更新時に上書き）
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 最新の評価結果を取得
        latest_eval = self.evaluation_history[-1]
        
        # 結果をテキストファイルに書き込む（固定ファイル名で上書き）
        filename = f"{save_dir}/pareto_solutions_{mode_name}.txt"
        try:
            with open(filename, 'w') as f:
                # ヘッダー情報
                f.write(f"# パレートフロント解 - {mode_name}\n")
                f.write(f"# 最終更新日時: {timestamp}\n")
                f.write(f"# ステップ数: {self.global_step}\n")
                f.write(f"# 評価履歴数: {len(self.evaluation_history)}\n")
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
            
            # マップデータの視覚化を別途保存（固定ファイル名で上書き）
            try:
                if 'maps' in latest_eval:
                    final_maps = latest_eval['maps']
                    map_image_path = f"{save_dir}/final_schedule_{mode_name}.png"
                    visualize_map(final_maps[0], final_maps[1], [], map_image_path)
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
                
                # 各エピソードで異なるシードを使用して多様性を確保
                episode_seed = int(time.time() * 1000) + p_idx * 1000 + ep + self.global_step
                np.random.seed(episode_seed)
                
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

    def cleanup_memory(self):
        """明示的にメモリをクリーンアップ"""
        import gc
        
        # 経験再生バッファの古いエピソードを削除
        if len(self.experience_replay) > 5000:  # バッファが大きすぎる場合
            # 古いエピソードを削除（下位50%を削除）
            sorted_buffer = sorted(self.experience_replay, key=lambda x: x[0])
            self.experience_replay = sorted_buffer[len(sorted_buffer)//2:]
            heapq.heapify(self.experience_replay)
            print(f"メモリクリーンアップ: 経験再生バッファを {len(sorted_buffer)} から {len(self.experience_replay)} に削減")
        
        # 評価履歴をクリア
        if len(self.evaluation_history) > 100:  # 履歴が多すぎる場合
            self.evaluation_history = self.evaluation_history[-50:]  # 最新50件のみ保持
            self.evaluation_timestamps = self.evaluation_timestamps[-50:]
            self.global_steps_at_evaluation = self.global_steps_at_evaluation[-50:]
            print(f"メモリクリーンアップ: 評価履歴を最新50件に削減")
        
        # ガベージコレクションを強制実行
        gc.collect()

    def __del__(self):
        """デストラクタ: メモリを確実に解放"""
        try:
            # 大きな配列を明示的に解放
            if hasattr(self, 'experience_replay'):
                del self.experience_replay
            if hasattr(self, 'evaluation_history'):
                del self.evaluation_history
            if hasattr(self, 'evaluation_timestamps'):
                del self.evaluation_timestamps
            if hasattr(self, 'global_steps_at_evaluation'):
                del self.global_steps_at_evaluation
        except:
            pass  # デストラクタでのエラーは無視

    # PCNクラスに追加
    def check_overfitting(self, test_data_size=100):
        """同じデータでoverfittingチェック"""
        if len(self.experience_replay) < test_data_size:
            return False
        
        # 小さな固定データセットで学習
        test_episodes = self.experience_replay[:test_data_size]
        
        # 同じデータで複数回学習
        losses = []
        for epoch in range(10):
            # 同じデータで学習
            loss, _ = self._update_on_fixed_data(test_episodes)
            losses.append(loss.item())
        
        # 損失が減少するかチェック
        if losses[-1] < losses[0] * 0.8:  # 20%以上減少
            print("✓ Overfitting チェック: 学習可能")
            return True
        else:
            print("⚠️  Overfitting チェック: 学習が困難")
            return False

    def _update_on_fixed_data(self, episodes):
        """固定データでの学習"""
        # 既存のupdateメソッドの簡略版
        # ... 実装 ...

    def save_learning_data_to_file(self, filename="learning_data_debug.txt", sample_size=100):
        """学習データの詳細をファイルに書き込む
        
        Args:
            filename (str): 出力ファイル名
            sample_size (int): 分析するサンプル数
        """
        import os
        from datetime import datetime
        
        if len(self.experience_replay) == 0:
            print("⚠️  経験バッファが空です。データを収集してから実行してください。")
            return
        
        # ファイルパスを設定
        debug_dir = "debug_learning_data"
        os.makedirs(debug_dir, exist_ok=True)
        filepath = os.path.join(debug_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PCN学習データ詳細分析\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # 1. 全体統計
            f.write("1. 全体統計\n")
            f.write("-" * 40 + "\n")
            total_episodes = len(self.experience_replay)
            total_transitions = sum(len(episode[2]) for episode in self.experience_replay)
            f.write(f"総エピソード数: {total_episodes}\n")
            f.write(f"総遷移数: {total_transitions}\n")
            f.write(f"バッチサイズ: {self.batch_size}\n")
            f.write(f"割引率γ: {self.gamma}\n")
            f.write(f"学習率: {self.opt.param_groups[0]['lr']}\n")
            f.write(f"デバイス: {self.device}\n")
            f.write(f"モデルタイプ: {'EnhancedPCNModel' if self.use_enhanced_model else 'DefaultModel'}\n\n")
            
            # 2. サンプルデータの詳細分析
            f.write("2. サンプルデータ詳細分析\n")
            f.write("-" * 40 + "\n")
            
            # サンプルサイズを調整
            actual_sample_size = min(sample_size, total_episodes)
            sample_indices = np.random.choice(total_episodes, actual_sample_size, replace=False)
            
            all_observations = []
            all_actions = []
            all_desired_returns = []
            all_desired_horizons = []
            all_episode_lengths = []
            
            for i, idx in enumerate(sample_indices):
                episode = self.experience_replay[idx][2]
                episode_length = len(episode)
                all_episode_lengths.append(episode_length)
                
                # エピソード内のランダムなステップを選択
                t = np.random.randint(0, episode_length)
                transition = episode[t]
                
                # データを収集
                all_observations.append(transition.observation)
                all_actions.append(transition.action)
                
                # 論文通りの累積報酬計算
                remaining_return = 0.0
                for j in range(t, episode_length):
                    remaining_return += (self.gamma ** (j - t)) * episode[j].reward
                
                all_desired_returns.append(remaining_return)
                all_desired_horizons.append(episode_length - t)
                
                # 最初の10サンプルの詳細を記録
                if i < 10:
                    f.write(f"\nサンプル {i+1}:\n")
                    f.write(f"  エピソード長: {episode_length}\n")
                    f.write(f"  選択ステップ: {t}\n")
                    f.write(f"  観測形状: {transition.observation.shape}\n")
                    f.write(f"  観測値（最初の10要素）: {transition.observation}\n")
                    f.write(f"  行動: {transition.action}\n")
                    f.write(f"  即時報酬: {transition.reward}\n")
                    f.write(f"  累積報酬（論文通り）: {remaining_return}\n")
                    f.write(f"  残りステップ数: {episode_length - t}\n")
            
            # 3. 統計分析
            f.write("\n3. 統計分析\n")
            f.write("-" * 40 + "\n")
            
            # 観測データの統計
            obs_array = np.array(all_observations)
            f.write("観測データ統計:\n")
            f.write(f"  形状: {obs_array.shape}\n")
            f.write(f"  平均: {obs_array.mean():.6f}\n")
            f.write(f"  標準偏差: {obs_array.std():.6f}\n")
            f.write(f"  最小値: {obs_array.min():.6f}\n")
            f.write(f"  最大値: {obs_array.max():.6f}\n")
            f.write(f"  NaN数: {np.isnan(obs_array).sum()}\n")
            f.write(f"  Inf数: {np.isinf(obs_array).sum()}\n\n")
            
            # 行動分布
            actions_array = np.array(all_actions)
            unique_actions, action_counts = np.unique(actions_array, return_counts=True)
            f.write("行動分布:\n")
            for action, count in zip(unique_actions, action_counts):
                percentage = (count / len(actions_array)) * 100
                f.write(f"  行動{action}: {count}回 ({percentage:.1f}%)\n")
            
            # 不均衡チェック
            max_action_ratio = np.max(action_counts) / len(actions_array)
            f.write(f"  最大行動比率: {max_action_ratio:.3f}")
            if max_action_ratio > 0.8:
                f.write(" ⚠️  不均衡検出（80%以上が同じ行動）\n")
            else:
                f.write(" ✓ バランス良好\n")
            f.write("\n")
            
            # 累積報酬の統計
            returns_array = np.array(all_desired_returns)
            f.write("累積報酬統計（論文通り）:\n")
            f.write(f"  平均: {returns_array.mean():.6f}\n")
            f.write(f"  標準偏差: {returns_array.std():.6f}\n")
            f.write(f"  最小値: {returns_array.min():.6f}\n")
            f.write(f"  最大値: {returns_array.max():.6f}\n")
            f.write(f"  範囲: {returns_array.max() - returns_array.min():.6f}\n")
            f.write(f"  NaN数: {np.isnan(returns_array).sum()}\n")
            f.write(f"  Inf数: {np.isinf(returns_array).sum()}\n\n")
            
            # ホライゾンの統計
            horizons_array = np.array(all_desired_horizons)
            f.write("残りステップ数統計:\n")
            f.write(f"  平均: {horizons_array.mean():.1f}\n")
            f.write(f"  標準偏差: {horizons_array.std():.1f}\n")
            f.write(f"  最小値: {horizons_array.min()}\n")
            f.write(f"  最大値: {horizons_array.max()}\n\n")
            
            # エピソード長の統計
            episode_lengths_array = np.array(all_episode_lengths)
            f.write("エピソード長統計:\n")
            f.write(f"  平均: {episode_lengths_array.mean():.1f}\n")
            f.write(f"  標準偏差: {episode_lengths_array.std():.1f}\n")
            f.write(f"  最小値: {episode_lengths_array.min()}\n")
            f.write(f"  最大値: {episode_lengths_array.max()}\n\n")
            
            # 4. 学習データの品質チェック
            f.write("4. 学習データ品質チェック\n")
            f.write("-" * 40 + "\n")
            
            # データの有効性チェック
            issues = []
            if np.isnan(obs_array).any():
                issues.append("観測データにNaNが含まれています")
            if np.isinf(obs_array).any():
                issues.append("観測データにInfが含まれています")
            if np.isnan(returns_array).any():
                issues.append("累積報酬にNaNが含まれています")
            if np.isinf(returns_array).any():
                issues.append("累積報酬にInfが含まれています")
            if max_action_ratio > 0.9:
                issues.append("行動分布が極端に偏っています（90%以上が同じ行動）")
            if returns_array.std() < 1e-6:
                issues.append("累積報酬の分散が極めて小さいです")
            
            if issues:
                f.write("⚠️  検出された問題:\n")
                for issue in issues:
                    f.write(f"  - {issue}\n")
            else:
                f.write("✓ データ品質に問題は検出されませんでした\n")
            
            f.write("\n")
            
            # 5. 推奨事項
            f.write("5. 推奨事項\n")
            f.write("-" * 40 + "\n")
            
            if max_action_ratio > 0.8:
                f.write("- 行動分布の不均衡を解決するため、重み付き損失関数の使用を検討してください\n")
                f.write("- より多様な行動を生成するため、探索戦略の見直しを検討してください\n")
            
            if returns_array.std() < 1e-3:
                f.write("- 累積報酬の分散が小さいため、報酬設計の見直しを検討してください\n")
            
            if obs_array.std() > 100:
                f.write("- 観測データのスケールが大きいため、正規化の適用を検討してください\n")
            
            if len(unique_actions) < 2:
                f.write("- 行動の多様性が不足しています。環境設定の見直しを検討してください\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("分析完了\n")
            f.write("=" * 80 + "\n")
        
        print(f"✓ 学習データ分析結果を保存しました: {filepath}")
        return filepath

    def export_learning_samples_to_csv(self, filename="learning_samples.csv", num_samples=1000):
        """学習サンプルをCSVファイルにエクスポート
        
        Args:
            filename (str): 出力ファイル名
            num_samples (int): エクスポートするサンプル数
        """
        import pandas as pd
        import os
        from datetime import datetime
        
        if len(self.experience_replay) == 0:
            print("⚠️  経験バッファが空です。データを収集してから実行してください。")
            return
        
        # ファイルパスを設定
        debug_dir = "debug_learning_data"
        os.makedirs(debug_dir, exist_ok=True)
        filepath = os.path.join(debug_dir, filename)
        
        # サンプルデータを収集
        total_episodes = len(self.experience_replay)
        actual_samples = min(num_samples, total_episodes * 10)  # エピソードあたり最大10サンプル
        
        data_rows = []
        sample_count = 0
        
        for episode_idx in range(total_episodes):
            if sample_count >= actual_samples:
                break
                
            episode = self.experience_replay[episode_idx][2]
            episode_length = len(episode)
            
            # エピソードから複数サンプルを抽出
            num_episode_samples = min(10, episode_length)
            step_indices = np.random.choice(episode_length, num_episode_samples, replace=False)
            
            for t in step_indices:
                if sample_count >= actual_samples:
                    break
                    
                transition = episode[t]
                
                # 論文通りの累積報酬計算
                remaining_return = 0.0
                for j in range(t, episode_length):
                    remaining_return += (self.gamma ** (j - t)) * episode[j].reward
                
                # 観測データをフラット化
                obs_flat = transition.observation.flatten()
                
                # データ行を作成
                row = {
                    'sample_id': sample_count,
                    'episode_id': episode_idx,
                    'step_in_episode': t,
                    'episode_length': episode_length,
                    'action': transition.action,
                    'immediate_reward': transition.reward,
                    'cumulative_return': remaining_return,
                    'remaining_steps': episode_length - t,
                }
                
                # 観測データの各要素を追加
                for i, obs_val in enumerate(obs_flat):
                    row[f'observation_{i}'] = obs_val
                
                data_rows.append(row)
                sample_count += 1
        
        # DataFrameを作成してCSVに保存
        df = pd.DataFrame(data_rows)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"✓ 学習サンプルをCSVにエクスポートしました: {filepath}")
        print(f"  サンプル数: {len(df)}")
        print(f"  列数: {len(df.columns)}")
        
        return filepath