import numpy as np
from typing import Tuple, Optional, List
import gymnasium as gym

class HeuristicAgent:
    """ヒューリスティックによるスケジューリングエージェント
    
    オンプレミス優先戦略：
    1. オンプレミスに配置を試行
    2. 待ち時間が閾値を超える場合はクラウドに配置
    3. オンプレミスが空いたら再びオンプレミス優先
    """
    
    def __init__(self, 
                 base_wait_time_threshold: int = 5,
                 width_factor: float = 0.3,
                 use_cloud_fallback: bool = True):
        """
        Args:
            base_wait_time_threshold: 基本の待ち時間閾値（ステップ数）
            width_factor: ジョブ幅の影響度（0.3 = 幅の30%）
            use_cloud_fallback: クラウドへのフォールバックを許可するか
        """
        self.base_wait_time_threshold = base_wait_time_threshold
        self.width_factor = width_factor
        self.use_cloud_fallback = use_cloud_fallback
        
        # 統計情報
        self.stats = {
            'on_premise_allocations': 0,
            'cloud_allocations': 0,
            'fallback_to_cloud': 0,
            'total_jobs': 0
        }
    
    def calculate_wait_time_threshold(self, job_width: int) -> float:
        """ジョブの幅（処理時間）に応じて閾値を調整
        
        Args:
            job_width: ジョブの幅（処理時間）
            
        Returns:
            調整された待ち時間閾値
        """
        # 幅の30%を待ち時間の許容範囲として追加
        width_adjustment = job_width * self.width_factor
        return self.base_wait_time_threshold + width_adjustment
    
    def select_action(self, env) -> Tuple[int, bool]:
        """ジョブの配置先を決定
        
        Args:
            env: スケジューリング環境
            
        Returns:
            (action, is_valid): 選択した行動と有効性
        """
        # ジョブキューが空の場合はスキップ
        if env.rear_job_queue <= 0:
            return 0, False
        
        # 現在のジョブを取得
        job = env.job_queue[0]
        
        # ジョブが無効な場合はスキップ
        if job[0] == 0 and job[1] == 0:
            return 0, False
        
        # オンプレミスへの配置を試行
        on_premise_action = [0, 0]  # オンプレミス
        position_on_premise, wait_time_on_premise = env.find_allocation_position(on_premise_action)
        is_valid_on_premise = position_on_premise is not None
        
        if is_valid_on_premise:
            # ジョブの幅に応じた閾値を計算
            job_width = int(job[0])
            threshold = self.calculate_wait_time_threshold(job_width)
            
            # 待ち時間が閾値以下ならオンプレミスに配置
            if wait_time_on_premise <= threshold:
                return 0, True  # オンプレミス
        
        # オンプレミスが溢れている場合はクラウドに配置
        if self.use_cloud_fallback:
            cloud_action = [0, 1]  # クラウド
            position_cloud, wait_time_cloud = env.find_allocation_position(cloud_action)
            is_valid_cloud = position_cloud is not None
            
            if is_valid_cloud:
                self.stats['fallback_to_cloud'] += 1
                return 1, True  # クラウド
        
        # どちらも配置できない場合
        return 0, False
    
    def schedule_job(self, env) -> Optional[Tuple[float, float, Tuple]]:
        """ジョブをスケジュール
        
        Args:
            env: スケジューリング環境
            
        Returns:
            (wait_time, cost, position) または None（配置できない場合）
        """
        action, is_valid = self.select_action(env)
        
        if not is_valid:
            return None
        
        # 実際の配置を実行
        if action == 0:  # オンプレミス
            action_array = [0, 0]
            position, wait_time = env.find_allocation_position(action_array)
            if position is not None:
                wait_time = env.do_schedule(action_array, env.job_queue[0], position)
                cost = env.compute_cost(action_array, env.job_queue[0], True)
                self.stats['on_premise_allocations'] += 1
                return wait_time, cost, position
        else:  # クラウド
            action_array = [0, 1]
            position, wait_time = env.find_allocation_position(action_array)
            if position is not None:
                wait_time = env.do_schedule(action_array, env.job_queue[0], position)
                cost = env.compute_cost(action_array, env.job_queue[0], True)
                self.stats['cloud_allocations'] += 1
                return wait_time, cost, position
        
        return None
    
    def get_stats(self) -> dict:
        """統計情報を取得"""
        total = self.stats['on_premise_allocations'] + self.stats['cloud_allocations']
        if total > 0:
            self.stats['on_premise_ratio'] = self.stats['on_premise_allocations'] / total
            self.stats['cloud_ratio'] = self.stats['cloud_allocations'] / total
            self.stats['fallback_ratio'] = self.stats['fallback_to_cloud'] / total
        
        return self.stats.copy()
    
    def reset_stats(self):
        """統計情報をリセット"""
        self.stats = {
            'on_premise_allocations': 0,
            'cloud_allocations': 0,
            'fallback_to_cloud': 0,
            'total_jobs': 0
        }
    
    def print_stats(self):
        """統計情報を表示"""
        stats = self.get_stats()
        print("=== ヒューリスティックエージェント統計 ===")
        print(f"オンプレミス配置: {stats['on_premise_allocations']}")
        print(f"クラウド配置: {stats['cloud_allocations']}")
        print(f"クラウドフォールバック: {stats['fallback_to_cloud']}")
        if 'on_premise_ratio' in stats:
            print(f"オンプレミス比率: {stats['on_premise_ratio']:.2%}")
            print(f"クラウド比率: {stats['cloud_ratio']:.2%}")
            print(f"フォールバック比率: {stats['fallback_ratio']:.2%}")
        print("=====================================") 