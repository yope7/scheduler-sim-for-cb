import gym
import numpy as np
from collections import deque, defaultdict
import sys
from sklearn.preprocessing import MinMaxScaler
import csv
from numba import jit, njit
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from gym.envs.registration import EnvSpec
from gym import spaces
from src.utils.map_visualizer import visualize_map

class EventType(Enum):
    """イベントタイプの定義"""
    JOB_ARRIVAL = "job_arrival"      # ジョブ到着
    JOB_COMPLETION = "job_completion"  # ジョブ完了
    TIME_STEP = "time_step"          # 時間ステップ
    RESOURCE_AVAILABLE = "resource_available"  # リソース利用可能

@dataclass
class Event:
    """イベントクラス"""
    event_type: EventType
    timestamp: int
    job_id: Optional[int] = None
    data: Optional[dict] = None

@dataclass
class Job:
    """ジョブクラス"""
    job_id: int
    width: int          # 処理時間
    height: int         # ノード数
    arrival_time: int   # 到着時刻
    priority: int = 1   # 優先度
    deadline: Optional[int] = None  # デッドライン
    user_id: Optional[int] = None   # ユーザーID

# numbaで高速化するための関数
@njit
def time_transition_njit(on_premise_window_status, on_premise_window_job_id,
                        cloud_window_status, cloud_window_job_id,
                        slide_on_premise, slide_cloud):
    if slide_on_premise:
        # オンプレミスのスライドウィンドウをシフト（手動実装）
        for i in range(on_premise_window_status.shape[0]):
            for j in range(on_premise_window_status.shape[1]-1):
                on_premise_window_status[i, j] = on_premise_window_status[i, j+1]
                on_premise_window_job_id[i, j] = on_premise_window_job_id[i, j+1]
            # 最後の列をクリア
            on_premise_window_status[i, -1] = 0
            on_premise_window_job_id[i, -1] = -1

    if slide_cloud:
        # クラウドのスライドウィンドウをシフト（手動実装）
        for i in range(cloud_window_status.shape[0]):
            for j in range(cloud_window_status.shape[1]-1):
                cloud_window_status[i, j] = cloud_window_status[i, j+1]
                cloud_window_job_id[i, j] = cloud_window_job_id[i, j+1]
            # 最後の列をクリア
            cloud_window_status[i, -1] = 0
            cloud_window_job_id[i, -1] = -1
        
    return on_premise_window_status, on_premise_window_job_id, cloud_window_status, cloud_window_job_id

@njit
def do_schedule_njit(on_premise_window_status, on_premise_window_job_id, 
                    cloud_window_status, cloud_window_job_id,
                    job_width, job_height, job_id, when_submitted, use_cloud, 
                    position, current_time):
    # 位置情報を解析して注意深く処理
    if len(position) == 2:
        # 従来の連続した割り当て
        i, a = position
        if not use_cloud:  # オンプレミスに割り当てる場合
            on_premise_window_status[i:i + job_height, a:a + job_width] = 1
            on_premise_window_job_id[i:i + job_height, a:a + job_width] = job_id
        else:  # クラウドに割り当てる場合
            cloud_window_status[i:i + job_height, a:a + job_width] = 1
            cloud_window_job_id[i:i + job_height, a:a + job_width] = job_id
    else:
        # 分散した割り当て（node_allocationがリストの場合）
        i, a, node_allocation = position
        if not use_cloud:
            for col_offset in range(len(node_allocation)):
                col = a + col_offset
                for node_idx in range(len(node_allocation[col_offset])):
                    node = node_allocation[col_offset][node_idx]
                    on_premise_window_status[node, col] = 1
                    on_premise_window_job_id[node, col] = job_id
        else:
            for col_offset in range(len(node_allocation)):
                col = a + col_offset
                for node_idx in range(len(node_allocation[col_offset])):
                    node = node_allocation[col_offset][node_idx]
                    cloud_window_status[node, col] = 1
                    cloud_window_job_id[node, col] = job_id
        
    return current_time - when_submitted

class EventDrivenSchedulingEnv(gym.core.Env):
    """イベント駆動のジョブスケジューラ環境"""
    
    def __init__(self, max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck, 
                 weight_wt, weight_cost, penalty_not_allocate, penalty_invalid_action, jobs_set=None,
                 job_type=0, flag=0):
        
        # 基本パラメータ
        self.step_count = 0
        self.episode = 0
        self.time = 0
        self.max_step = max_step
        self.n_window = n_window
        self.n_on_premise_node = n_on_premise_node
        self.n_cloud_node = n_cloud_node
        self.n_job_queue_obs = n_job_queue_obs
        self.n_job_queue_bck = n_job_queue_bck
        self.weight_wt = weight_wt
        self.weight_cost = weight_cost
        self.penalty_not_allocate = penalty_not_allocate
        self.penalty_invalid_action = penalty_invalid_action
        self.flag = flag
        self.cost = 0
        self.done_flag = False
        
        # イベント駆動システム
        self.event_queue = deque()  # イベントキュー
        self.scheduled_events = defaultdict(list)  # スケジュール済みイベント
        self.completed_jobs = set()  # 完了したジョブ
        self.running_jobs = {}  # 実行中のジョブ {job_id: (start_time, end_time, use_cloud)}
        
        # ジョブ管理
        self.jobs_set = jobs_set
        self.jobs = []
        self.job_queue = deque()  # 待機中のジョブ
        self.job_id_counter = 0
        
        # リソース管理
        self.dtype = [('status', 'i4'), ('job_id', 'i4')]
        self.on_premise_window = np.zeros((self.n_on_premise_node, self.n_window), dtype=self.dtype)
        self.cloud_window = np.zeros((self.n_cloud_node, self.n_window), dtype=self.dtype)
        self.obs_window_size = 10
        
        # 履歴管理
        self.on_premise_window_history_full = np.full((self.n_on_premise_node, 1), -1)
        self.cloud_window_history_full = np.full((self.n_cloud_node, 1), -1)
        
        # 統計情報
        self.total_waiting_time = 0
        self.total_cost = 0
        self.waiting_times = []
        self.completed_jobs_count = 0
        self.total_jobs_count = 0
        
        # 行動空間と観測空間
        self.n_action = 2
        self.action_space = gym.spaces.Discrete(self.n_action)
        obs_space_size = (self.n_on_premise_node * self.obs_window_size +
                         self.n_cloud_node * self.obs_window_size +
                         8 * 5)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(obs_space_size,), 
            dtype=np.float32
        )
        self.reward_space = spaces.Box(low=-100, high=100, shape=(2,), dtype=np.float32)
        self.spec = EnvSpec(id='EventDrivenScheduling-v0', entry_point='schedulingEventEnv:EventDrivenSchedulingEnv')
        
        # 標準化
        self.mm = MinMaxScaler()
        self.multi_algorithm = True
        self.job_type = job_type
        
        # 初期化
        self.init_window()

    def __del__(self):
        """デストラクタ: メモリを確実に解放"""
        try:
            if hasattr(self, 'on_premise_window_history_full'):
                del self.on_premise_window_history_full
            if hasattr(self, 'cloud_window_history_full'):
                del self.cloud_window_history_full
            if hasattr(self, 'on_premise_window'):
                del self.on_premise_window
            if hasattr(self, 'cloud_window'):
                del self.cloud_window
            if hasattr(self, 'job_queue'):
                del self.job_queue
            if hasattr(self, 'jobs'):
                del self.jobs
        except:
            pass

    def init_window(self):
        """ウィンドウの初期化"""
        self.on_premise_window = np.zeros((self.n_on_premise_node, self.n_window), dtype=self.dtype)
        self.cloud_window = np.zeros((self.n_cloud_node, self.n_window), dtype=self.dtype)
        self.on_premise_window['status'] = 0
        self.on_premise_window['job_id'] = -1
        self.cloud_window['status'] = 0
        self.cloud_window['job_id'] = -1

    def add_event(self, event: Event):
        """イベントをキューに追加"""
        self.event_queue.append(event)
        # タイムスタンプでソート
        self.event_queue = deque(sorted(self.event_queue, key=lambda x: x.timestamp))

    def schedule_job_completion(self, job_id: int, start_time: int, duration: int, use_cloud: bool):
        """ジョブ完了イベントをスケジュール"""
        completion_time = start_time + duration
        completion_event = Event(
            event_type=EventType.JOB_COMPLETION,
            timestamp=completion_time,
            job_id=job_id,
            data={'use_cloud': use_cloud, 'start_time': start_time}
        )
        self.add_event(completion_event)

    def process_job_arrival(self, job: Job):
        """ジョブ到着イベントの処理"""
        self.job_queue.append(job)
        self.total_jobs_count += 1

    def process_job_completion(self, job_id: int, use_cloud: bool):
        """ジョブ完了イベントの処理"""
        if job_id in self.running_jobs:
            start_time, end_time, _ = self.running_jobs[job_id]
            
            # リソースを解放
            if use_cloud:
                # クラウドウィンドウからジョブを削除
                self.cloud_window['status'][self.cloud_window['job_id'] == job_id] = 0
                self.cloud_window['job_id'][self.cloud_window['job_id'] == job_id] = -1
            else:
                # オンプレミスウィンドウからジョブを削除
                self.on_premise_window['status'][self.on_premise_window['job_id'] == job_id] = 0
                self.on_premise_window['job_id'][self.on_premise_window['job_id'] == job_id] = -1
            
            # 完了済みとしてマーク
            self.completed_jobs.add(job_id)
            self.completed_jobs_count += 1
            del self.running_jobs[job_id]
            
            # リソース利用可能イベントを生成
            resource_event = Event(
                event_type=EventType.RESOURCE_AVAILABLE,
                timestamp=self.time,
                data={'use_cloud': use_cloud}
            )
            self.add_event(resource_event)

    def process_time_step(self):
        """時間ステップイベントの処理"""
        # 時間を進める
        self.time += 1
        self.update_window_history()
        
        # 新しいジョブの到着をチェック
        self.check_job_arrivals()

    def check_job_arrivals(self):
        """新しいジョブの到着をチェック"""
        if self.multi_algorithm and self.jobs_set:
            jobs = self.jobs_set[self.episode] if self.episode < len(self.jobs_set) else []
            
            for job_data in jobs:
                # job_dataの構造: [処理時間, ノード数, job_id, 到着時刻, 優先度, デッドライン, ユーザーID, その他]
                arrival_time = int(job_data[3])  # 到着時刻はインデックス3
                if arrival_time == self.time:  # 到着時刻が現在時刻と一致
                    job = Job(
                        job_id=self.job_id_counter,
                        width=int(job_data[0]),    # 処理時間
                        height=int(job_data[1]),   # ノード数
                        arrival_time=arrival_time,
                        priority=int(job_data[4]) if len(job_data) > 4 else 1
                    )
                    self.job_id_counter += 1
                    
                    arrival_event = Event(
                        event_type=EventType.JOB_ARRIVAL,
                        timestamp=self.time,
                        job_id=job.job_id,
                        data={'job': job}
                    )
                    self.add_event(arrival_event)

    def check_pending_jobs(self):
        """待機中のジョブをチェックして、スケジュール可能なものを自動スケジュール"""
        if not self.job_queue:
            return
        
        # 待機中のジョブを順番にチェック
        for i in range(len(self.job_queue)):
            job = self.job_queue[i]
            
            # オンプレミスとクラウドの両方を試す
            on_premise_valid, on_premise_wt, on_premise_position = self.check_is_valid_action([0, 0])
            cloud_valid, cloud_wt, cloud_position = self.check_is_valid_action([0, 1])
            
            # より良い選択肢を選ぶ
            if on_premise_valid and cloud_valid:
                if on_premise_wt <= cloud_wt:
                    action = [0, 0]  # オンプレミス
                    position = on_premise_position
                else:
                    action = [0, 1]  # クラウド
                    position = cloud_position
                is_valid = True
            elif on_premise_valid:
                action = [0, 0]  # オンプレミス
                position = on_premise_position
                is_valid = True
            elif cloud_valid:
                action = [0, 1]  # クラウド
                position = cloud_position
                is_valid = True
            else:
                is_valid = False
            
            if is_valid:
                # ジョブをスケジュール
                self.do_schedule(action, job, position)
                
                # ジョブ完了イベントをスケジュール
                job_width = int(job.width)
                use_cloud = action[1]
                self.schedule_job_completion(job.job_id, self.time, job_width, use_cloud)
                
                # 実行中のジョブとして記録
                self.running_jobs[job.job_id] = (self.time, self.time + job_width, use_cloud)
                
                # 待ち時間を記録
                waiting_time = self.time - job.arrival_time
                self.waiting_times.append(waiting_time)
                self.total_waiting_time += waiting_time
                
                # コスト計算
                cost = self.compute_cost(action, job, is_valid)
                self.total_cost += cost
                
                # キューから削除
                self.job_queue.remove(job)
                break  # 1つのジョブをスケジュールしたら終了

    def get_next_event(self) -> Optional[Event]:
        """次のイベントを取得"""
        if self.event_queue:
            return self.event_queue.popleft()
        return None

    def step(self, action_raw):
        """環境のステップ実行（イベント駆動）"""
        scheduled = False
        time_reward_new = 0
        wt_step = 0
        cost = 0  # cost変数を初期化
        
        # 次のイベントを取得
        next_event = self.get_next_event()
        
        if next_event is None:
            # イベントがない場合は時間を進める
            self.process_time_step()
            next_event = self.get_next_event()
        
        if next_event is None:
            # まだイベントがない場合
            observation = self.get_observation()
            rewards = np.array([0, 0], dtype=np.float64)
            done = self.check_is_done()
            return observation, rewards, scheduled, wt_step, done
        
        # イベントを処理
        if next_event.event_type == EventType.JOB_ARRIVAL:
            job = next_event.data['job']
            self.process_job_arrival(job)
            
        elif next_event.event_type == EventType.JOB_COMPLETION:
            self.process_job_completion(next_event.job_id, next_event.data['use_cloud'])
            
        elif next_event.event_type == EventType.TIME_STEP:
            self.process_time_step()
            
        elif next_event.event_type == EventType.RESOURCE_AVAILABLE:
            # リソースが利用可能になった場合、待機中のジョブをチェック
            self.check_pending_jobs()
        
        # イベント処理後に、まだ処理すべきイベントがある場合は続行
        while self.event_queue and self.event_queue[0].timestamp <= self.time:
            next_event = self.get_next_event()
            if next_event.event_type == EventType.JOB_ARRIVAL:
                job = next_event.data['job']
                self.process_job_arrival(job)
            elif next_event.event_type == EventType.JOB_COMPLETION:
                self.process_job_completion(next_event.job_id, next_event.data['use_cloud'])
            elif next_event.event_type == EventType.TIME_STEP:
                self.process_time_step()
            elif next_event.event_type == EventType.RESOURCE_AVAILABLE:
                self.check_pending_jobs()
        
        # ジョブキューにジョブがある場合はスケジューリングを試行
        if self.job_queue:
            allocated_job = self.job_queue[0]
            action = self.get_converted_action(action_raw)
            
            # イベント駆動システムでは、スケジュールできるまでリソースを横に見ていく
            # オンプレミスとクラウドの両方を試す
            on_premise_valid, on_premise_wt, on_premise_position = self.check_is_valid_action([0, 0])
            cloud_valid, cloud_wt, cloud_position = self.check_is_valid_action([0, 1])
            
            # より良い選択肢を選ぶ（待ち時間が短い方）
            if on_premise_valid and cloud_valid:
                if on_premise_wt <= cloud_wt:
                    is_valid, wt_real, position = True, on_premise_wt, on_premise_position
                    action = [0, 0]  # オンプレミス
                else:
                    is_valid, wt_real, position = True, cloud_wt, cloud_position
                    action = [0, 1]  # クラウド
            elif on_premise_valid:
                is_valid, wt_real, position = True, on_premise_wt, on_premise_position
                action = [0, 0]  # オンプレミス
            elif cloud_valid:
                is_valid, wt_real, position = True, cloud_wt, cloud_position
                action = [0, 1]  # クラウド
            else:
                # どちらも無効な場合（リソース不足）は待機
                is_valid, wt_real, position = False, np.inf, None
            
            if is_valid:
                # ジョブをスケジュール
                wt_step = self.do_schedule(action, allocated_job, position)
                scheduled = True
                self.job_queue.popleft()  # キューから削除
                
                # ジョブ完了イベントをスケジュール
                job_width = int(allocated_job.width)
                use_cloud = action[1]
                self.schedule_job_completion(allocated_job.job_id, self.time, job_width, use_cloud)
                
                # 実行中のジョブとして記録
                self.running_jobs[allocated_job.job_id] = (self.time, self.time + job_width, use_cloud)
                
                # 待ち時間を記録
                waiting_time = self.time - allocated_job.arrival_time
                self.waiting_times.append(waiting_time)
                self.total_waiting_time += waiting_time
                
                # コスト計算
                cost = self.compute_cost(action, allocated_job, is_valid)
                self.total_cost += cost
                
                # 待ち時間に基づく報酬（短いほど良い）
                time_reward_new = max(0, 10 - wt_real) / 10  # 0-1の範囲
                
                # ジョブがスケジュールされた後、時間を進める
                self.time += 1
                self.update_window_history()
                self.check_job_arrivals()
            else:
                # リソース不足で待機（ペナルティなし）
                scheduled = False
                time_reward_new = 0
                cost = 0
                wt_step = 0
        
        observation = self.get_observation()
        done = self.check_is_done()
        rewards = np.array([time_reward_new, -cost], dtype=np.float64)
        
        self.step_count += 1
        return observation, rewards, scheduled, wt_step, done

    def reset(self):
        """環境のリセット"""
        # 変数を初期化
        self.time = 0
        self.step_count = 0
        self.event_queue.clear()
        self.scheduled_events.clear()
        self.completed_jobs.clear()
        self.running_jobs.clear()
        self.job_queue.clear()
        self.job_id_counter = 0
        
        # ジョブセットの読み込み
        if self.multi_algorithm and self.jobs_set:
            self.jobs = self.jobs_set[self.episode] if self.episode < len(self.jobs_set) else []
        else:
            self.jobs = []
        
        # ウィンドウの初期化
        self.init_window()
        
        # 履歴の初期化
        if hasattr(self, 'on_premise_window_history_full'):
            del self.on_premise_window_history_full
        if hasattr(self, 'cloud_window_history_full'):
            del self.cloud_window_history_full
        
        self.on_premise_window_history_full = np.full((self.n_on_premise_node, 1), -1)
        self.cloud_window_history_full = np.full((self.n_cloud_node, 1), -1)
        
        # 統計情報の初期化
        self.total_waiting_time = 0
        self.total_cost = 0
        self.waiting_times = []
        self.completed_jobs_count = 0
        self.total_jobs_count = 0
        
        # 初期イベントの生成
        self.add_event(Event(EventType.TIME_STEP, 0))
        
        # 初期ジョブの到着をチェック
        self.check_job_arrivals()
        
        observation = self.get_observation()
        return observation

    def get_converted_action(self, a):
        """スカラーのactionをリストに変換"""
        if a == 0:
            method = 0
            use_cloud = 0
        elif a == 1:
            method = 0
            use_cloud = 1
        else:
            print('a is invalid')
            exit()
        return [method, use_cloud]

    def update_window_history(self):
        """ウィンドウ履歴の更新"""
        # オンプレミスの履歴を更新
        left_column_on_premise = self.on_premise_window['job_id'][:, 0:1]
        old_on_premise = self.on_premise_window_history_full
        self.on_premise_window_history_full = np.hstack((old_on_premise, left_column_on_premise.copy()))
        del old_on_premise
        
        # クラウドの履歴を更新
        left_column_cloud = self.cloud_window['job_id'][:, 0:1]
        old_cloud = self.cloud_window_history_full
        self.cloud_window_history_full = np.hstack((old_cloud, left_column_cloud.copy()))
        del old_cloud

    def get_observation(self):
        """観測データ(状態)を取得"""
        # 現在時刻を基準とした固定長の最新部分のみを取得
        current_time_col = self.time % self.n_window
        
        # オンプレミスの観測範囲を計算
        on_premise_start = current_time_col
        on_premise_end = min(current_time_col + self.obs_window_size, self.n_window)
        
        # クラウドの観測範囲を計算
        cloud_start = current_time_col
        cloud_end = min(current_time_col + self.obs_window_size, self.n_window)
        
        # 観測範囲がウィンドウの右端を超える場合の処理
        if on_premise_end < current_time_col + self.obs_window_size:
            available_slots = on_premise_end - on_premise_start
            obs_on_premise_window_status = self.on_premise_window['status'][:, on_premise_start:on_premise_end].flatten()
            remaining_slots = self.obs_window_size - available_slots
            padding = np.zeros(remaining_slots * self.n_on_premise_node, dtype=np.float32)
            obs_on_premise_window_status = np.concatenate([obs_on_premise_window_status, padding])
        else:
            obs_on_premise_window_status = self.on_premise_window['status'][:, on_premise_start:on_premise_end].flatten()
        
        if cloud_end < current_time_col + self.obs_window_size:
            available_slots = cloud_end - cloud_start
            obs_cloud_window_status = self.cloud_window['status'][:, cloud_start:cloud_end].flatten()
            remaining_slots = self.obs_window_size - available_slots
            padding = np.zeros(remaining_slots * self.n_cloud_node, dtype=np.float32)
            obs_cloud_window_status = np.concatenate([obs_cloud_window_status, padding])
        else:
            obs_cloud_window_status = self.cloud_window['status'][:, cloud_start:cloud_end].flatten()
        
        # ジョブキューを観測用に変換
        job_queue_obs = np.zeros((5, 8))  # 最大5ジョブ、8属性
        if self.job_queue:  # ジョブキューが空でない場合のみ処理
            for i, job in enumerate(list(self.job_queue)[:5]):
                job_queue_obs[i] = [job.width, job.height, job.job_id, job.arrival_time, 
                                   job.priority, 0, 0, 0]  # 残りは0で埋める
        
        # 1次元配列として連結
        observation = np.concatenate([
            obs_on_premise_window_status,
            obs_cloud_window_status,
            job_queue_obs.flatten()
        ]).astype(np.float32)
        
        return observation

    def do_schedule(self, action, job, position):
        """ジョブのスケジューリング実行"""
        job_width = int(job.width)
        job_height = int(job.height)
        job_id = int(job.job_id)
        when_submitted = int(job.arrival_time)
        use_cloud = action[1]
        
        # 位置情報を解析
        if len(position) == 2:
            # 従来の連続した割り当て
            i, a = position
            if not use_cloud:  # オンプレミスに割り当てる場合
                self.on_premise_window['status'][i:i + job_height, a:a + job_width] = 1
                self.on_premise_window['job_id'][i:i + job_height, a:a + job_width] = job_id
            else:  # クラウドに割り当てる場合
                self.cloud_window['status'][i:i + job_height, a:a + job_width] = 1
                self.cloud_window['job_id'][i:i + job_height, a:a + job_width] = job_id
        else:
            # 分散した割り当て
            i, a, node_allocation = position
            window = self.on_premise_window if not use_cloud else self.cloud_window
            
            for col_offset, nodes in enumerate(node_allocation):
                col = a + col_offset
                for node in nodes:
                    window['status'][node, col] = 1
                    window['job_id'][node, col] = job_id
        
        waiting_time = self.time - when_submitted
        return waiting_time

    def check_is_valid_action(self, action):
        """actionが有効かどうかを判定"""
        method = action[0]
        use_cloud = action[1]
        
        if not self.job_queue:
            return False, -1, None
            
        job = self.job_queue[0]
        job_width = int(job.width)
        job_height = int(job.height)
        when_submitted = int(job.arrival_time)
        time = self.time

        # job が無効なら早期リターン
        if job_width == 0 and job_height == 0:
            return False, -1, None

        # 使用するウィンドウの選択
        if not use_cloud:
            window = self.on_premise_window['status']
            max_h, max_w = self.n_on_premise_node, self.n_window
        else:
            window = self.cloud_window['status']
            max_h, max_w = self.n_cloud_node, self.n_window

        # ジョブサイズが大きすぎる場合は早期リターン
        if job_width > max_w or job_height > max_h:
            return False, -1, None
    
        # 列ごとに利用可能なリソースを確認する方法
        for a in range(max_w - job_width + 1):  # 横方向（時間軸）の開始位置を探索
            can_allocate = True
            
            # この開始位置から job_width 分の幅で配置できるか確認
            for col_offset in range(job_width):
                col = a + col_offset
                available_nodes = max_h - np.count_nonzero(window[:, col])
                
                # この列に必要なノード数が確保できるか
                if available_nodes < job_height:
                    can_allocate = False
                    break
            
            if can_allocate:
                # 利用可能な連続したノード領域を探す（優先的に上から詰める）
                for i in range(max_h - job_height + 1):
                    if np.all(window[i:i + job_height, a:a + job_width] == 0):
                        return True, time + a - when_submitted, (i, a)
                
                # 連続した領域がない場合、分散して割り当てる
                node_allocation = []
                for col_offset in range(job_width):
                    col = a + col_offset
                    free_nodes = [i for i in range(max_h) if window[i, col] == 0][:job_height]
                    if len(free_nodes) >= job_height:
                        node_allocation.append(free_nodes[:job_height])
                    else:
                        can_allocate = False
                        break
                
                if can_allocate:
                    return True, time + a - when_submitted, (0, a, node_allocation)
    
        # 有効な配置位置が見つからなかった
        return False, np.inf, None

    def compute_cost(self, action, allocated_job, is_valid):
        """コストを計算"""
        if is_valid:  # actionが有効だった場合
            if action[1] == 0:  # オンプレミスに割り当てる場合
                cost = 0
            elif action[1] == 1:  # クラウドに割り当てる場合
                cost = 1 * (allocated_job.width * allocated_job.height)  # (処理時間)*(クラウドで使うノード数)をコストとする
            else:  # 割り当てない場合
                cost = 0
        else:  # actionが無効だった場合
            cost = 0

        return cost

    def check_is_done(self):
        """エピソード終了条件を判定"""
        # 最大ステップ数に達した場合
        if self.step_count >= self.max_step:
            return True
        
        # 手動で終了フラグが設定された場合
        if self.done_flag:
            return True
        
        # ジョブキューが空で、実行中ジョブもなく、イベントキューも空の場合
        if (len(self.job_queue) == 0 and 
            len(self.running_jobs) == 0 and 
            len(self.event_queue) == 0):
            return True
        
        return False

    def calc_objective_values(self):
        """目的関数値の計算"""
        # クラウドに割り当てられたジョブのコスト計算
        total_cost = 0
        
        # クラウドウィンドウに記録されているジョブIDを取得
        unique_job_ids = np.unique(self.cloud_window_history_full)
        unique_job_ids = unique_job_ids[unique_job_ids >= 0]  # -1を除外
        
        # 各クラウドジョブに対してコストを計算
        for job_id in unique_job_ids:
            # 対応するジョブを検索
            matching_jobs = [job for job in self.jobs if int(job[5]) == job_id]
            
            if matching_jobs:
                job = matching_jobs[0]
                job_width = int(job[1])  # 処理時間
                job_height = int(job[2])  # ノード数
                job_cost = job_width * job_height
                total_cost += job_cost
            else:
                # クラウドウィンドウでの実際のセル数を代わりに使用
                job_cells = np.count_nonzero(self.cloud_window_history_full == job_id)
                total_cost += job_cells
        
        self.cost = total_cost
        
        # makespanは二次元配列で要素が入っているものの中で一番右側（インデックス）の値をとる
        def calculate_makespan(matrix):
            matrix = np.array(matrix)
            latest_index = -1
            target_value = -1
            
            for row in matrix:
                unique_values = np.unique(row[row != -1])  # -1を除くユニークな値を取得
                for value in unique_values:
                    last_occurrence = np.max(np.where(row == value))  # 値の最後の出現位置
                    if last_occurrence > latest_index:
                        latest_index = last_occurrence
                        target_value = value
            
            return latest_index
        
        mkspan_onpre = calculate_makespan(self.on_premise_window_history_full)
        mkspan_cloud = calculate_makespan(self.cloud_window_history_full)
        
        # 平均待ち時間を計算
        avg_waiting_time = np.mean(self.waiting_times) if self.waiting_times else 0.0

        return self.cost, max(mkspan_onpre, mkspan_cloud), avg_waiting_time

    def get_episode_metrics(self):
        """エピソードごとの待ち時間とコストを計算して返す"""
        average_waiting_time = self.total_waiting_time / self.completed_jobs_count if self.completed_jobs_count > 0 else 0
        return average_waiting_time, self.total_cost
    
    def get_windows(self):
        """ウィンドウ履歴を取得"""
        return self.on_premise_window_history_full, self.cloud_window_history_full

    def get_cost(self):
        """コストを取得"""
        return self.cost

    def show_final_window_history(self):
        """最終ウィンドウ履歴を表示"""
        print("self.on_premise_window_history_full:\n", self.on_premise_window_history_full)
        print("self.cloud_window_history_full:\n", self.cloud_window_history_full)

    def render_map(self, name):
        """スケジューリング結果をマップとして表示"""
        print("オンプレミスのスケジューリング結果:")
        print(self.on_premise_window_history_full)
        print("\nクラウドのスケジューリング結果:")
        print(self.cloud_window_history_full)
        
        # ジョブリストを作成
        job_list = [{'size': (job.width, job.height)} for job in self.job_queue]
        
        # マップを可視化
        visualize_map(self.on_premise_window_history_full, self.cloud_window_history_full, job_list, name) 