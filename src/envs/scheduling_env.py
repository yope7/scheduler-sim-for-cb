import gym
import numpy as np
from collections import deque
import sys
import os
from sklearn.preprocessing import MinMaxScaler
import csv
from numba import jit, njit
from numpy.lib.stride_tricks import sliding_window_view

# Numbaデバッグキャッシュを有効化（キャッシュのHIT/MISSをログに出力）
# より詳細なログを出力する場合は'2'を設定
# 再コンパイルの原因を調査する場合は以下の行のコメントを外す
# os.environ.setdefault('NUMBA_DEBUG_CACHE', '1')
# キャッシュディレクトリのパスも出力（デバッグ用）
# os.environ.setdefault('NUMBA_CACHE_DIR', os.path.expanduser('~/.cache/numba'))

# SciPyの利用可否を確認
try:
    from scipy.ndimage import uniform_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from gym.envs.registration import EnvSpec
from gym import spaces
from src.utils.map_visualizer import visualize_map
# import cupy as cp

# numbaで高速化するための関数
@njit(cache=True, fastmath=True)
def get_unique_job_ids_njit(history_matrix, max_job_id):
    """
    ヒストリーマトリックスからユニークなjob_idを取得（-1を除く）
    np.uniqueの代替としてNumbaで高速化
    
    Args:
        history_matrix: (H, W) の配列
        max_job_id: 最大job_id（メモリ最適化のため）
    
    Returns:
        unique_job_ids: ユニークなjob_idの配列
    """
    # -1以外の値を一時的に格納（最大要素数）
    temp_ids = np.zeros(max_job_id, dtype=np.int32)
    # Numbaではnp.bool_がサポートされていない場合があるため、int8を使用
    seen = np.zeros(max_job_id, dtype=np.int8)
    count = 0
    
    for i in range(history_matrix.shape[0]):
        for j in range(history_matrix.shape[1]):
            job_id = history_matrix[i, j]
            if job_id >= 0 and job_id < max_job_id and seen[job_id] == 0:
                seen[job_id] = 1
                temp_ids[count] = job_id
                count += 1
                if count >= max_job_id:
                    break
        if count >= max_job_id:
            break
    
    # 結果を配列に格納
    unique_job_ids = np.zeros(count, dtype=np.int32)
    for i in range(count):
        unique_job_ids[i] = temp_ids[i]
    
    return unique_job_ids

@njit(cache=True, fastmath=True)
def calculate_makespan_batch_njit(onpre_matrix, cloud_matrix):
    """
    オンプレとクラウドの両方のmakespanを一括計算（最適化版）
    
    Args:
        onpre_matrix: (H, W) の配列、-1は未使用
        cloud_matrix: (H, W) の配列、-1は未使用
    
    Returns:
        (onpre_makespan, cloud_makespan): それぞれのmakespan（最大列インデックス）
    """
    onpre_makespan = -1
    cloud_makespan = -1
    
    # オンプレミスのmakespan計算: 各行で右端の有効な列を探索
    if onpre_matrix.shape[0] > 0 and onpre_matrix.shape[1] > 0:
        n_rows, n_cols = onpre_matrix.shape[0], onpre_matrix.shape[1]
        for i in range(n_rows):
            # 右から左へ検索し、最初の有効値（-1でない）を見つけたらその列インデックスを記録
            for j in range(n_cols - 1, -1, -1):
                if onpre_matrix[i, j] != -1:
                    if j > onpre_makespan:
                        onpre_makespan = j
                    break  # この行で最大インデックスが見つかったので次の行へ
    
    # クラウドのmakespan計算: 同様の処理
    if cloud_matrix.shape[0] > 0 and cloud_matrix.shape[1] > 0:
        n_rows, n_cols = cloud_matrix.shape[0], cloud_matrix.shape[1]
        for i in range(n_rows):
            for j in range(n_cols - 1, -1, -1):
                if cloud_matrix[i, j] != -1:
                    if j > cloud_makespan:
                        cloud_makespan = j
                    break  # この行で最大インデックスが見つかったので次の行へ
    
    return onpre_makespan, cloud_makespan

@njit(cache=True, fastmath=True)
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

@njit(cache=True, fastmath=True)
def first_fit_position_njit(window, max_h, max_w, job_height, job_width):
    # 連続領域の最初の適合法を探索（上詰め・左詰め）
    limit_w = max_w - job_width + 1
    limit_h = max_h - job_height + 1
    for a in range(limit_w):
        for i in range(limit_h):
            ok = True
            for r in range(job_height):
                for c in range(job_width):
                    if window[i + r, a + c] != 0:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True, i, a
    return False, -1, -1



# 学習環境
class SchedulingEnv(gym.core.Env): 
    def __init__(self, max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck, weight_wt,
                 weight_cost, penalty_not_allocate, penalty_invalid_action, jobs_set=None,
                 job_type=0, flag=0):
        self.step_count = 0  # 現在のステップ数(今何ステップ目かを示す)
        # self.n_job_per_time = self.config['param_job']['n_job_per_time']
        self.episode = 0  # 現在のエピソード(今何エピソード目かを示す); agentに教えてもらう
        self.time = 0  # 時刻(ジョブ到着の判定に使う)
        self.max_step = max_step  # ステップの最大数(1エピソードの終了時刻)
        self.index_next_job = 0  # 次に待っている新しいジョブのインデックス 新しいジョブをジョブキューに追加するときに使う
        # self.index_next_job_ideal = 0 # 理想的な状態(処理時間を迎えたのにジョブキューがいっぱいでジョブキューに格納されていないジョブがない)であれば次に待っている新しいジョブのインデックス
        self.n_window = n_window  # スライドウィンドウの横幅
        self.n_on_premise_node = n_on_premise_node  # オンプレミス計算資源のノード数
        self.on_premise_window_history = np.zeros((6,1)) # オンプレミスのスライドウィンドウの履歴
        self.on_premise_window_user_history = np.zeros((6,1)) # オンプレミスのスライドウィンドウの履歴
        self.cloud_window_user_history_show = np.zeros((6,1)) # クラウドのスライドウィンドウの履歴
        self.on_premise_window_user_history_show = np.zeros((6,1)) # オンプレミスのスライドウィンドウの履歴
        self.cloud_window_history = np.zeros((6,1)) # クラウドのスライドウィンドウの履歴
        self.cloud_window_user_history = np.zeros((6,1)) # クラウドのスライドウィンドウの履歴
        self.n_cloud_node = n_cloud_node  # クラウド計算資源のノード数
        self.n_job_queue_obs = n_job_queue_obs  # ジョブキューの観測部分の長さ
        self.n_job_queue_bck = n_job_queue_bck  # ジョブキューのバックログ部分の長さ
        self.rear_job_queue = 0  # ジョブキューの末尾 (== 0: ジョブキューが空)
        self.weight_wt = weight_wt  # 報酬における待ち時間の重み
        self.weight_cost = weight_cost  # 報酬におけるコストの重み
        self.penalty_not_allocate = penalty_not_allocate  # 割り当てない(一時キューに格納する)という行動を選択した際のペナルティー
        self.penalty_invalid_action = penalty_invalid_action  # actionが無効だった場合のペナルティー
        self.flag = flag
        self.cost = 0
        # 構造化配列の定義
        self.dtype = [('status', 'i4'), ('job_id', 'i4')]
        self.done_flag = False

        # 観測用の固定長パラメータ（現在時刻から見るスロット数）
        self.obs_window_size = 10  # 現在時刻から10スロット先までを観測

        # オンプレミスとクラウドのスライドウィンドウを構造化配列で管理
        self.on_premise_window = np.zeros((self.n_on_premise_node, self.n_window), dtype=self.dtype)
        self.cloud_window = np.zeros((self.n_cloud_node, self.n_window), dtype=self.dtype)
        # 初期化

        # 後回し(defer)行動: SCHEDULER_ALLOW_DEFER=1 で action=2(現ジョブを今は置かず後回し)を追加。
        # 既定OFF時は n_action=2 で完全に従来通り(ビット一致)。巨大ジョブの配置タイミングを選べる。
        self._allow_defer = os.environ.get("SCHEDULER_ALLOW_DEFER", "0") == "1"
        self._defer_max = int(os.environ.get("SCHEDULER_DEFER_MAX", "3"))       # 1ジョブの後回し上限(終了保証)
        self._defer_offset = int(os.environ.get("SCHEDULER_DEFER_OFFSET", "1"))  # 何ジョブ後ろへ回すか(既定1=1つ後ろ。ユーザー設計: deferは1つ後ろのみ認める)
        # 待ち時間の目的指標: "wait"(生待ち時間, 既定=従来通り) / "slowdown"(待ち÷処理時間=stretch)。
        # slowdown は巨大ジョブの待ちを pt で割って平準化し、「巨大ジョブの待ちが報酬を支配」を緩和する
        # (谷口案: でかいジョブは待たせてもslowdownは大して変わらん・小ジョブを待たせない)。
        # env が報酬・目的値の両方で slowdown を返すので、学習/NSGA/eval が自動で同じ座標系になる。
        self._wait_metric = os.environ.get("SCHEDULER_WAIT_METRIC", "wait")
        self.n_action = 3 if self._allow_defer else 2  # 行動数
        self.action_space = gym.spaces.Discrete(self.n_action)  # 行動空間
        self.tmp_queue = deque()  # 割り当てられなかったジョブを一時的に格納するキュー
        obs_space_size = (self.n_on_premise_node * self.obs_window_size +
                         self.n_cloud_node * self.obs_window_size +
                         8 * 5)  # 全ジョブ属性(8)×5ジョブ
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(obs_space_size,), 
            dtype=np.float32
        )
        self.reward_space = spaces.Box(low=-100, high=100, shape=(2,), dtype=np.float32)
        self.spec = EnvSpec(id='SimpleScheduling-v0', entry_point='schedulingEnv:SchedulingEnv')


        self.mm = MinMaxScaler()  # 観測データを標準化するやつ
        self.multi_algorithm = True  # 複数アルゴリズムで一気に実行しており，各アルゴリズムでジョブが同じかどうか; ジョブ生成に関わる
        if self.multi_algorithm:  # クラス宣言前に既にジョブセットを定義している場合
            self.job_type = job_type  # ジョブタイプは既に決まっている

        # アルゴリズムごとに同じジョブにする場合(複数アルゴリズムで一気に実行している場合)は環境定義の前にジョブをすでに生成してあるのでそれをエピソード最初に読み取るだけ

        if self.multi_algorithm:  # アルゴリズムごとに同じジョブにする場合
            self.jobs_set = jobs_set  # 事前に生成したジョブセットを受け取る
            #jobをcsvで保存．savetxt
            #配列をファイルにjsonで保存，delimiterで改行文字を指定
            # print("jobs_set_head:\n",self.jobs_set[0])
            

            
        else:  # アルゴリズムごとに同じジョブではない場合
            # ジョブ設定
            # ジョブをエピソードごとに変えるか，固定にするかを指定
            self.job_is_constant = 1
            # ジョブタイプを指定
            if self.job_is_constant:
                self.job_type = 1
            else:
                self.job_type = int(input(
                    'select job-type \n 1:default 2:input file 3:random 4:random(エピソードごとにジョブ数やジョブサイズの範囲を変える) \n >'))

            # ランダムの場合，必要な変数を指定
            if self.job_type == 3:
                # 最大時間を入力
                self.max_t = int(input('input max time \n >'))
                # 単位時間あたりのジョブ数を入力
                while True:
                    self.n_job_per_time = int(input('input number of jobs per time (-1:random) \n >'))
                    self.n_job_per_time_is_random = False
                    if 1 <= self.n_job_per_time <= 10:
                        break
                    elif self.n_job_per_time == -1:
                        self.n_job_per_time_is_random = True
                        break
                    else:
                        print('input again')

            # 途中で性質が切り替わる場合の設定
            if self.job_type == 4:
                self.random_job_type = int(input(
                    'select random job type \n 1:a few small jobs -> many large jobs 2:a few small jobs -> a few large jobs 3:a few small jobs -> many small jobs \n >'))
                self.point_switch_random_job_type = int(input('ジョブの性質が変わるエピソードを入力 \n >'))
                # 最大時間を入力
                self.max_t = int(input('input max time \n >'))

            # ジョブが固定の場合，ここでジョブを設定してしまう
            if self.job_is_constant:
                exit("job get none")

        # デバッグ用
        self.job_queue = np.zeros((len(self.jobs_set),8))
        self.log_job = False

        self.total_waiting_time = 0
        self.total_cost = 0
        self.completed_jobs = 0

        # 処理予定のジョブ総数
        self.total_jobs_count = 0
        # 処理完了したジョブ数
        self.jobs_processed_count = 0
        
        # 待ち時間を記録するリスト
        self.waiting_times = []



    def __del__(self):
        """デストラクタ: メモリを確実に解放"""
        try:
            # 大きな配列を明示的に解放
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
            pass  # デストラクタでのエラーは無視

    def cleanup_memory(self):
        """明示的にメモリをクリーンアップ"""
        import gc
        
        # 大きな配列を明示的に解放
        if hasattr(self, 'on_premise_window_history_full'):
            del self.on_premise_window_history_full
        if hasattr(self, 'cloud_window_history_full'):
            del self.cloud_window_history_full
        
        # ガベージコレクションを強制実行
        gc.collect()

    def optimize_window_history(self):
        """ウィンドウ履歴のサイズを最適化"""
        # 履歴が大きすぎる場合（1000列以上）は古い部分を削除
        if hasattr(self, 'on_premise_window_history_full') and self.on_premise_window_history_full.shape[1] > 1000:
            # 最新の500列のみを保持
            self.on_premise_window_history_full = self.on_premise_window_history_full[:, -500:]
            print(f"オンプレミス履歴を最適化: 最新500列のみ保持")
        
        if hasattr(self, 'cloud_window_history_full') and self.cloud_window_history_full.shape[1] > 1000:
            # 最新の500列のみを保持
            self.cloud_window_history_full = self.cloud_window_history_full[:, -500:]
            print(f"クラウド履歴を最適化: 最新500列のみ保持")

    # 各ステップで実行される操作
    def step(self, action_raw):
        # ループ外で一度だけキャッシュを取得（ループ内での重複取得を避ける）
        cache_onpre = None
        cache_cloud = None
        cache_needs_refresh = True  # 初回は必ず取得
        
        while True:
            scheduled = False
            valid_action_cache = {}
            time_reward_new = 0
            time = self.time
            allocated_job = self.job_queue[0]
            action = self.get_converted_action(action_raw)   
            wt_step = 0
            is_valid = False
            
            # キャッシュが必要な場合のみ再取得（差分更新済みの場合は再構築不要）
            if cache_needs_refresh or cache_onpre is None or cache_cloud is None:
                cache_onpre = self._rebuild_cache_if_needed(use_cloud=False)
                cache_cloud = self._rebuild_cache_if_needed(use_cloud=True)
                cache_needs_refresh = False
            
            # print("find_allocation_position start")
            position, wt_real = self.find_allocation_position(action, cache_onpre=cache_onpre, cache_cloud=cache_cloud)
            # print("find_allocation_position end")
            if position is None:
                if np.all(allocated_job == 0):
                    job_none = True
                    self.time_transition(True, True)
                    # time_transition後は差分更新が試行されるため、キャッシュを再取得
                    cache_needs_refresh = True
                else:
                    job_none = False
                    if action[1] == 0:
                        self.time_transition(True, False)
                    else:
                        self.time_transition(False, True)
                    # time_transition後は差分更新が試行されるため、キャッシュを再取得
                    cache_needs_refresh = True

                var_reward = 0
                var_after = 0
                wt_step = 0
                std_mean_before = 0
                std_mean_after = 0
                std_reward = 0
                continue
            else:
                job_none = False
                job = self.job_queue[0]
                is_valid = True

                if action[1] == 0:
                    if (0,1) in valid_action_cache:
                        position_parallel, wt_parallel = valid_action_cache[(0,1)]
                    else:
                        # 既に取得したキャッシュを再利用
                        position_parallel, wt_parallel = self.find_allocation_position([0,1], cache_onpre=cache_onpre, cache_cloud=cache_cloud)
                        valid_action_cache[(0,1)] = (position_parallel, wt_parallel)
                if action[1] == 1:
                    if (0,0) in valid_action_cache:
                        position_parallel, wt_parallel = valid_action_cache[(0,0)]
                    else:
                        # 既に取得したキャッシュを再利用
                        position_parallel, wt_parallel = self.find_allocation_position([0,0], cache_onpre=cache_onpre, cache_cloud=cache_cloud)
                        valid_action_cache[(0,0)] = (position_parallel, wt_parallel)

                # time_reward_new = (
                #     1 if wt_real < wt_parallel else
                #     1 if wt_real == wt_parallel else
                #     -1
                # )
                time_reward_new = wt_real

                if action[0] == 0:
                    # print("do_schedule start")
                    wt_step = self.do_schedule(action,job,position)
                    # print("do_schedule end")
                    scheduled = True
                    self.job_queue = np.roll(self.job_queue, -1, axis=0)
                    self.job_queue[-1] = 0
                    self.rear_job_queue -= 1

                    observation = self.get_observation()
                    cost = self.compute_cost(action, allocated_job, is_valid)
                    done = self.check_is_done()

                    # メモリ最適化: 100ステップごとにウィンドウ履歴を最適化
                    # if self.step_count % 100 == 0:
                        # self.optimize_window_history()


                    rewards = np.array([-time_reward_new,-cost], dtype=np.float64)
                    # print("rewards: ",rewards)
                    self.step_count += 1
                    return observation, rewards, scheduled, wt_step, done

    def get_next_init_windows(self):
        return self.next_init_windows
    
    def safe_std(self, data):
        """データの標準偏差を安全に計算"""
        if len(data) > 1:
            return np.std(data)
        else:
            return 0.0

    def safe_mean(self, data):
        """データの平均を安全に計算"""
        if len(data) > 0:
            return np.mean(data)
        else:
            return 0.0
    # スカラーのactionをリストに変換
    def get_converted_action(self, a):
        if a == 0:
            method = 0
            use_cloud = 0
        elif a == 1:
            method = 0
            use_cloud = 1

        else:
            print('a is invalid')
            exit()
        action = [method, use_cloud]

        return action


    def init_window(self):
        self.on_premise_window = np.zeros((self.n_on_premise_node, self.n_window), dtype=self.dtype)
        self.cloud_window = np.zeros((self.n_cloud_node, self.n_window), dtype=self.dtype)
        self.on_premise_window['status'] = 0
        self.on_premise_window['job_id'] = -1
        self.cloud_window['status'] = 0
        self.cloud_window['job_id'] = -1

    # 初期化
    # 各エピソードの最初に呼出される
    def reset(self):
        # 変数を初期化
        self.time = 0
        self.sums_user =[]
        self.job_allocated = []
        self.step_count = 0
        if self.multi_algorithm:  # アルゴリズムごとに同じジョブである場合
            self.jobs = self.jobs_set[self.episode]
        else:  # アルゴリズムごとに同じジョブではない場合
            # ジョブが固定でない(ジョブをエピソードごとに変える)場合，ジョブを再設定
            if not self.job_is_constant:
               exit("job get none")
        self.max_t = self.jobs[-1][0]  # 最大時間
        self.index_next_job = 0  # 新しいジョブをジョブキューに追加するときに使う
        self.total_jobs_count = len(self.jobs)

        # self.on_premise_window = np.zeros((self.n_on_premise_node, self.n_window), dtype=self.dtype)
        # self.on_premise_window['status'] = 0  # 0: 配置されていない
        # self.on_premise_window['job_id'] = -1  # -1: 配置されていない
        # self.cloud_window = np.zeros((self.n_cloud_node, self.n_window), dtype=self.dtype)
        # self.cloud_window['status'] = 0
        # self.cloud_window['job_id'] = -1
        self.init_window()
            # print("windows: ",self.on_premise_window, self.cloud_window)

        # キャッシュを初期化（エピソード開始時にリセット）
        self._ensure_cache_initialized()
        self._version_onpre = 0
        self._version_cloud = 0
        self._cache_onpre = {'version': -1}
        self._cache_cloud = {'version': -1}

        # メモリ最適化: 古い履歴配列を明示的に解放
        if hasattr(self, 'on_premise_window_history_full'):
            del self.on_premise_window_history_full
        if hasattr(self, 'cloud_window_history_full'):
            del self.cloud_window_history_full
        
        # 新しい履歴配列を初期化（動的伸長バッファ方式）
        init_cap = max(1024, self.n_window)  # 初期容量（必要に応じて拡張）
        self._hist_cap_onpre = init_cap
        self._hist_cap_cloud = init_cap
        self._hist_len_onpre = 1  # 先頭にダミー列（-1）
        self._hist_len_cloud = 1
        self._hist_onpre_buf = np.full((self.n_on_premise_node, init_cap), -1, dtype=int)
        self._hist_cloud_buf = np.full((self.n_cloud_node, init_cap), -1, dtype=int)
        # 互換のため公開配列も1列の-1で初期化（最終化時に置き換え）
        self.on_premise_window_history_full = self._hist_onpre_buf[:, :1]
        self.cloud_window_history_full = self._hist_cloud_buf[:, :1]

        # self.cloud_window_user = np.full((self.n_cloud_node, self.n_window),-1)  # クラウドのスライドウィンドウ
        self.job_queue = np.zeros((len(self.jobs),8)) # ジョブキュー
        self.rear_job_queue = 0  # ジョブキューの末尾 (== 0: ジョブキューが空)
        self.tmp_queue = deque()  # 割り当てられなかったジョブを一時的に格納するキュー
        self.user_wt_log = []


        self.total_waiting_time = 0
        self.total_cost = 0
        self.completed_jobs = 0
        
        # 待ち時間リストを初期化
        self.waiting_times = []

        # ジョブキューに新しいジョブを追加
        # print("self.job_queue: ",self.job_queue)
        self.append_new_job2job_queue()
        # print("add job to job queue")
        # print("self.job_queue: ",self.job_queue)
        # 観測データ(状態)を取得
        observation = self.get_observation()


        return observation

    def time_transition(self, slide_on_premise=True, slide_cloud=True):
        # 時間を1進める 
        self.time += 1
        self.update_window_history()

        # 構造化配列からndarrayを取得（型とメモリレイアウトを固定してキャッシュを有効化）
        on_premise_status = np.ascontiguousarray(self.on_premise_window['status'], dtype=np.int32)
        on_premise_job_id = np.ascontiguousarray(self.on_premise_window['job_id'], dtype=np.int32)
        cloud_status = np.ascontiguousarray(self.cloud_window['status'], dtype=np.int32)
        cloud_job_id = np.ascontiguousarray(self.cloud_window['job_id'], dtype=np.int32)
    
        # Numbaで高速化された関数を呼び出し
        on_premise_status, on_premise_job_id, cloud_status, cloud_job_id = time_transition_njit(
            on_premise_status, on_premise_job_id, cloud_status, cloud_job_id,
            slide_on_premise, slide_cloud
        )
        
        # 結果を元の配列に書き戻し
        self.on_premise_window['status'] = on_premise_status
        self.on_premise_window['job_id'] = on_premise_job_id
        self.cloud_window['status'] = cloud_status
        self.cloud_window['job_id'] = cloud_job_id

        # 新しいジョブをジョブキューに追加
        self.append_new_job2job_queue()
        
        # キャッシュを無効化（次回アクション検証時に再構築される）
        # 差分更新は毎回cumsumを2回実行するため遅いため、無効化のみにする
        self._invalidate_window_cache(on_premise=slide_on_premise, cloud=slide_cloud)

    def update_window_history(self):
        # オンプレミスの履歴を更新（動的バッファへ追記）
        left_column_on_premise = self.on_premise_window['job_id'][:, 0]
        if self._hist_len_onpre >= self._hist_cap_onpre:
            new_cap = self._hist_cap_onpre * 2
            new_buf = np.empty((self.n_on_premise_node, new_cap), dtype=int)
            new_buf[:, :self._hist_cap_onpre] = self._hist_onpre_buf
            new_buf[:, self._hist_cap_onpre:new_cap] = -1
            self._hist_onpre_buf = new_buf
            self._hist_cap_onpre = new_cap
        self._hist_onpre_buf[:, self._hist_len_onpre] = left_column_on_premise
        self._hist_len_onpre += 1

        # クラウドの履歴を更新（動的バッファへ追記）
        left_column_cloud = self.cloud_window['job_id'][:, 0]
        if self._hist_len_cloud >= self._hist_cap_cloud:
            new_cap = self._hist_cap_cloud * 2
            new_buf = np.empty((self.n_cloud_node, new_cap), dtype=int)
            new_buf[:, :self._hist_cap_cloud] = self._hist_cloud_buf
            new_buf[:, self._hist_cap_cloud:new_cap] = -1
            self._hist_cloud_buf = new_buf
            self._hist_cap_cloud = new_cap
        self._hist_cloud_buf[:, self._hist_len_cloud] = left_column_cloud
        self._hist_len_cloud += 1

        # print("self.on_premise_window_history_full:\n",self.on_premise_window_history_full)
        # print("self.cloud_window_history_full:\n",self.cloud_window_history_full)

    def finalize_window_history(self):
        """ウィンドウ全体を履歴に追加"""
        # バッファに蓄積済みの履歴と、現在のウィンドウ全体を最後に1回だけ連結
        hist_onpre = self._hist_onpre_buf[:, :self._hist_len_onpre]
        hist_cloud = self._hist_cloud_buf[:, :self._hist_len_cloud]
        self.on_premise_window_history_full = np.hstack((hist_onpre, self.on_premise_window['job_id'].copy()))
        self.cloud_window_history_full = np.hstack((hist_cloud, self.cloud_window['job_id'].copy()))
        # 一番左の列（初期の-1ダミー列）を削除
        self.on_premise_window_history_full = np.delete(self.on_premise_window_history_full, 0, axis=1)
        self.cloud_window_history_full = np.delete(self.cloud_window_history_full, 0, axis=1)

        # 待ち時間とコストを計算（makespanは不要なので計算をスキップ）
        cost, _, _ = self.calc_objective_values(calc_makespan=False, calc_avg_waiting_time=False)
        self.cost = cost

    # 内部キャッシュ: ウィンドウごとの事前計算を保持
    def _ensure_cache_initialized(self):
        # hasattrチェックを1回にまとめて最適化（属性が存在しない場合のみ初期化）
        if not hasattr(self, '_cache_onpre'):
            self._cache_onpre = {'version': -1}
            self._cache_cloud = {'version': -1}
            self._version_onpre = 0
            self._version_cloud = 0
        elif not hasattr(self, '_version_onpre'):
            # 後方互換性のため（_version_cloudのみがない場合）
            self._version_onpre = 0
            self._version_cloud = 0

    def _invalidate_window_cache(self, on_premise=True, cloud=True):
        self._ensure_cache_initialized()
        if on_premise:
            self._version_onpre += 1
        if cloud:
            self._version_cloud += 1

    def _update_cache_incremental(self, use_cloud, i_start, i_end, a_start, a_end):
        """
        ジョブ割り当て時の差分更新（incremental update）
        指定領域のみキャッシュを更新し、全再構築を避ける
        
        Args:
            use_cloud: クラウドかどうか
            i_start, i_end: 行の範囲（i_start <= row < i_end）
            a_start, a_end: 列の範囲（a_start <= col < a_end）
        """
        self._ensure_cache_initialized()
        cache = self._cache_onpre if not use_cloud else self._cache_cloud
        version = self._version_onpre if not use_cloud else self._version_cloud
        
        # キャッシュが存在しない、または無効な場合は全再構築
        if cache.get('version', -1) != version:
            return False  # 差分更新不可、全再構築が必要
        
        window = self.on_premise_window if not use_cloud else self.cloud_window
        status = window['status']
        occ = cache['occ']
        ps = cache['prefix_sum']
        free_per_col = cache['free_per_col']
        
        # 指定領域のoccを更新
        for row in range(i_start, i_end):
            for col in range(a_start, a_end):
                old_occ = occ[row, col]
                new_occ = 1 if status[row, col] != 0 else 0
                occ[row, col] = new_occ
                
                # free_per_colを更新
                if old_occ != new_occ:
                    free_per_col[col] += (old_occ - new_occ)
        
        # prefix_sumを全再計算（部分更新は複雑で遅いため、全再計算の方が速い）
        # occは既に更新済みなので、occから直接再計算
        H, W = occ.shape
        ps.fill(0)  # リセット
        # 2次元累積和を再計算（効率的な方法）
        ps[1:, 1:] = np.cumsum(np.cumsum(occ.astype(np.int32), axis=0, dtype=np.int32), axis=1, dtype=np.int32)
        
        # free_nodes_listを更新（影響を受けた列のみ）
        for col in range(a_start, a_end):
            status_col = status[:, col]
            cache['free_nodes_list'][col] = np.flatnonzero(status_col == 0)
        
        # バージョンを更新（差分更新成功時はバージョンを進める）
        if not use_cloud:
            self._version_onpre += 1
            cache['version'] = self._version_onpre
        else:
            self._version_cloud += 1
            cache['version'] = self._version_cloud
        return True  # 差分更新成功

    def _update_cache_after_slide(self, use_cloud):
        """
        time_transition後の差分更新（スライド後のキャッシュ更新）
        prefix_sumを左シフトし、最後の列をクリア
        
        Args:
            use_cloud: クラウドかどうか
        
        Returns:
            True: 差分更新成功、False: 差分更新不可（全再構築が必要）
        """
        self._ensure_cache_initialized()
        cache = self._cache_onpre if not use_cloud else self._cache_cloud
        # バージョンを更新する前にチェック（現在のバージョンと一致する必要がある）
        current_version = self._version_onpre if not use_cloud else self._version_cloud
        
        # キャッシュが存在しない、または無効な場合は全再構築
        if cache.get('version', -1) != current_version:
            return False  # 差分更新不可、全再構築が必要
        
        window = self.on_premise_window if not use_cloud else self.cloud_window
        status = window['status']
        occ = cache['occ']
        ps = cache['prefix_sum']
        free_per_col = cache['free_per_col']
        H, W = occ.shape
        
        # occを左シフト（最後の列をクリア）
        occ[:, :-1] = occ[:, 1:]
        occ[:, -1] = 0
        
        # statusから最後の列のoccを更新
        last_col_status = status[:, -1]
        for row in range(H):
            old_occ = occ[row, -1]
            new_occ = 1 if last_col_status[row] != 0 else 0
            occ[row, -1] = new_occ
            if old_occ != new_occ:
                free_per_col[-1] += (old_occ - new_occ)
        
        # free_per_colを左シフト
        free_per_col[:-1] = free_per_col[1:]
        # 最後の列の空きノード数を再計算
        free_per_col[-1] = H - np.sum(last_col_status != 0)
        
        # prefix_sumを全再計算（左シフト後のoccから直接再計算）
        # 部分更新は複雑で遅いため、全再計算の方が速い
        ps.fill(0)  # リセット
        ps[1:, 1:] = np.cumsum(np.cumsum(occ.astype(np.int32), axis=0, dtype=np.int32), axis=1, dtype=np.int32)
        
        # free_nodes_listを更新
        for col in range(W - 1):
            cache['free_nodes_list'][col] = cache['free_nodes_list'][col + 1]
        # 最後の列を再計算
        cache['free_nodes_list'][-1] = np.flatnonzero(status[:, -1] == 0)
        
        # バージョンを更新（差分更新成功時はバージョンを進める）
        # これにより、次回の呼び出し時にキャッシュが有効として認識される
        if not use_cloud:
            self._version_onpre += 1
            cache['version'] = self._version_onpre
        else:
            self._version_cloud += 1
            cache['version'] = self._version_cloud
        return True  # 差分更新成功

    def _rebuild_cache_if_needed(self, use_cloud):
        # キャッシュが初期化されていない場合のみ初期化（初回呼び出し時のみ）
        self._ensure_cache_initialized()
        if not use_cloud:
            # バージョンチェックを最適化（辞書アクセスを1回に）
            cache_version = self._cache_onpre.get('version', -1)
            if cache_version != self._version_onpre:
                status = self.on_premise_window['status']
                occ = (status != 0).astype(np.int32)
                free_per_col = status.shape[0] - occ.sum(axis=0)
                
                # 2D占有判定の最適化: cumsum(cumsum())の計算を効率化
                # 注意: 矩形サイズが可変のため、prefix_sum方式を維持
                # ただし、計算を最適化（型を縮小、メモリ効率を改善）
                H, W = occ.shape
                ps = np.zeros((H+1, W+1), dtype=np.int32)
                # cumsumを2回実行（行方向→列方向）: O(HW)で効率的
                # この計算は避けられないが、型をint32に固定してメモリ帯域を削減
                ps[1:,1:] = np.cumsum(np.cumsum(occ.astype(np.int32), axis=0, dtype=np.int32), axis=1, dtype=np.int32)
                
                free_nodes_list = [np.flatnonzero(status[:, c] == 0) for c in range(status.shape[1])]
                self._cache_onpre = {
                    'version': self._version_onpre,
                    'free_per_col': free_per_col,
                    'prefix_sum': ps,
                    'free_nodes_list': free_nodes_list,
                    'shape': status.shape,
                    'occ': occ  # 矩形判定用にoccも保存
                }
            return self._cache_onpre
        else:
            # バージョンチェックを最適化（辞書アクセスを1回に）
            cache_version = self._cache_cloud.get('version', -1)
            if cache_version != self._version_cloud:
                status = self.cloud_window['status']
                occ = (status != 0).astype(np.int32)
                free_per_col = status.shape[0] - occ.sum(axis=0)
                
                # 2D占有判定の最適化: cumsum(cumsum())の計算を効率化
                # 注意: 矩形サイズが可変のため、prefix_sum方式を維持
                # ただし、計算を最適化（型を縮小、メモリ効率を改善）
                H, W = occ.shape
                ps = np.zeros((H+1, W+1), dtype=np.int32)
                # cumsumを2回実行（行方向→列方向）: O(HW)で効率的
                # この計算は避けられないが、型をint32に固定してメモリ帯域を削減
                ps[1:,1:] = np.cumsum(np.cumsum(occ.astype(np.int32), axis=0, dtype=np.int32), axis=1, dtype=np.int32)
                
                free_nodes_list = [np.flatnonzero(status[:, c] == 0) for c in range(status.shape[1])]
                self._cache_cloud = {
                    'version': self._version_cloud,
                    'free_per_col': free_per_col,
                    'prefix_sum': ps,
                    'free_nodes_list': free_nodes_list,
                    'shape': status.shape,
                    'occ': occ  # 矩形判定用にoccも保存
                }
            return self._cache_cloud

    def get_cost(self):
        return self.cost

    def calc_objective_values(self, calc_makespan=True, calc_avg_waiting_time=True):
        # 待ち時間とコストを計算　

        """return:cost,makespan,avg_waiting_time 待ち時間の定義は，ジョブが到着してから，ジョブが開始するまでの時間．
        つまり，各ステップにおける遅延時間の総和をとればよい．
        
        Args:
            calc_makespan: makespanを計算するかどうか（デフォルト: True）
            calc_avg_waiting_time: 平均待ち時間を計算するかどうか（デフォルト: True）
        
        Returns:
            (cost, makespan, avg_waiting_time): 
            - calc_makespan=Falseの場合、makespanは-1を返す
            - calc_avg_waiting_time=Falseの場合、avg_waiting_timeは0.0を返す
        """
        
        # クラウドに割り当てられたジョブのコスト計算
        total_cost = 0
        
        # デバッグ出力
        # print("=== デバッグ情報 ===")
        # print(f"クラウドウィンドウ履歴: {self.cloud_window_history_full}")
        
        # クラウドウィンドウに記録されているジョブIDを取得（np.uniqueの代わりにNumba関数を使用）
        # 型とサイズを固定してキャッシュを有効化（contiguous配列として確保）
        history_matrix = np.ascontiguousarray(self.cloud_window_history_full, dtype=np.int32)
        # 最大job_idを固定値にする（毎回変わることでシグネチャが変わり再コンパイルされるのを防ぐ）
        # 十分大きな固定値を使用することで、キャッシュが有効に機能する
        max_job_id_fixed = 50000  # 固定値（実行ごとに変わらない）
        unique_job_ids = get_unique_job_ids_njit(history_matrix, max_job_id_fixed)
        # print(f"検出されたジョブID: {unique_job_ids}")
        
        # job_idからjob情報への辞書を作成（検索を高速化）
        job_dict = {}
        for job in self.jobs:
            job_id = int(job[5])
            if job_id not in job_dict:
                job_dict[job_id] = job
        
        # 各クラウドジョブに対してコストを計算
        for job_id in unique_job_ids:
            if job_id in job_dict:
                job = job_dict[job_id]
                # ジョブのサイズからコストを計算
                job_width = int(job[1])  # 処理時間
                job_height = int(job[2])  # ノード数
                job_cost = job_width * job_height
                total_cost += job_cost
                # print(f"ジョブID {job_id}: 処理時間={job_width}, ノード数={job_height}, コスト={job_cost}")
            else:
                # print(f"警告: ジョブID {job_id} に対応するジョブが見つかりません")
                # クラウドウィンドウでの実際のセル数を代わりに使用
                job_cells = np.count_nonzero(self.cloud_window_history_full == job_id)
                total_cost += job_cells
                # print(f"  代替計算: セル数={job_cells} をコストとして使用")
        
        self.cost = total_cost
        # print(f"総コスト: {total_cost}")
        
        # makespanは二次元配列で要素が入っているものの中で一番右側（インデックス）の値をとる
        # オンプレとクラウドを一括計算（Numba化で高速化）
        if calc_makespan:
            # 型とメモリレイアウトを固定してキャッシュを有効化（contiguous配列として確保）
            onpre_matrix = np.ascontiguousarray(self.on_premise_window_history_full, dtype=np.int32)
            cloud_matrix = np.ascontiguousarray(self.cloud_window_history_full, dtype=np.int32)
            mkspan_onpre, mkspan_cloud = calculate_makespan_batch_njit(onpre_matrix, cloud_matrix)
            makespan = max(mkspan_onpre, mkspan_cloud)
        else:
            makespan = -1  # 計算をスキップした場合は-1を返す
        
        # 平均待ち時間を計算
        if calc_avg_waiting_time:
            avg_waiting_time = np.mean(self.waiting_times) if self.waiting_times else 0.0
        else:
            avg_waiting_time = 0.0

        return self.cost, makespan, avg_waiting_time

    def show_final_window_history(self):
        #show simply
        print("self.on_premise_window_history_full:\n",self.on_premise_window_history_full)
        print("self.cloud_window_history_full:\n",self.cloud_window_history_full)

    # ウィンドウの履歴を取得
    def get_window_history_onpre(self):
        # mapの右端に-1を追加
        new_column = np.zeros((self.n_on_premise_node, 1), dtype=self.dtype)
        new_column['status'] = 0
        new_column['job_id'] = -1
        self.on_premise_window = np.hstack((self.on_premise_window, new_column))
        # print("self.on_premise_window:\n", self.on_premise_window)

    # ウィンドウの履歴を取得
    def get_window_history_cloud(self):
        # mapの右端に-1を追加
        new_column = np.zeros((self.n_cloud_node, 1), dtype=self.dtype)
        new_column['status'] = 0
        new_column['job_id'] = -1
        self.cloud_window = np.hstack((self.cloud_window, new_column))
        # print("self.cloud_window:\n", self.cloud_window)


    def append_new_job2job_queue(self):
        for i in range(len(self.jobs)):
            # print("self.jobs:\n",self.jobs)
            # print('index_next_job: ' + str(self.index_next_job))
            # print("len(self.jobs): "+ str(len(self.jobs)))
            if self.index_next_job == len(self.jobs):  # 最後のジョブまでジョブキューに格納した場合、脱出
                # print('job_queue'  + str(self.job_queue))
                # exit()
                break
            head_job = self.jobs[self.index_next_job]  # 先頭ジョブ


            # print('time',self.time)

            if head_job[0] <= self.time:  # 先頭のジョブが到着時刻を迎えていればジョブキューに追加
                # print('in')
                # ジョブキューに格納する前に提出時間が末尾に，処理時間が先頭になるようにインデックスをずらす

                # print(self.job_queue[i][3])

                if int(self.job_queue[i][2]) == 0:
                    # print('in2')
                    head_job = np.roll(head_job, -1)

                    #self.job_queue[i] = head_job[1:]

                    self.job_queue[i] = head_job

                    self.rear_job_queue += 1
                    self.index_next_job += 1
                # print('job_queue',self.job_queue)
        # 理想的な状態であれば次に待っている新しいジョブのインデックスを更新


    def do_schedule(self, action, job, position):
        self.jobs_processed_count += 1
        job_width = int(job[0])
        job_height = int(job[1])
        job_id = int(job[4])
        when_submitted = int(job[-1])
        use_cloud = action[1]
        
        # 位置情報を解析
        if len(position) == 2:
            # 従来の連続した割り当て（Numbaで高速化）
            i, a = position
            i_end = i + job_height
            a_end = a + job_width
            if not use_cloud:  # オンプレミスに割り当てる場合
                self.on_premise_window['status'][i:i_end, a:a_end] = 1
                self.on_premise_window['job_id'][i:i_end, a:a_end] = job_id
                # キャッシュを無効化（次回アクション検証時に再構築される）
                self._invalidate_window_cache(on_premise=True, cloud=False)
            else:  # クラウドに割り当てる場合
                self.cloud_window['status'][i:i_end, a:a_end] = 1
                self.cloud_window['job_id'][i:i_end, a:a_end] = job_id
                # キャッシュを無効化（次回アクション検証時に再構築される）
                self._invalidate_window_cache(on_premise=False, cloud=True)
        else:
            # 分散した割り当て（Python側で処理、Numba非対応）
            i, a, node_allocation = position
            window = self.on_premise_window if not use_cloud else self.cloud_window
            
            # 分散割り当ての場合
            for col_offset, nodes in enumerate(node_allocation):
                col = a + col_offset
                for node in nodes:
                    window['status'][node, col] = 1
                    window['job_id'][node, col] = job_id
            
            # キャッシュを無効化（次回アクション検証時に再構築される）
            if action[1] == 0:
                self._invalidate_window_cache(on_premise=True, cloud=False)
            else:
                self._invalidate_window_cache(on_premise=False, cloud=True)
        
        waiting_time = self.time - when_submitted
        
        # 待ち時間を記録
        self.waiting_times.append(waiting_time)
        
        return waiting_time

    # 観測データ(状態)を取得
    def get_observation(self):
        # マップの右端から固定長Nの部分を取得
        # オンプレミスウィンドウの右端から固定長分を取得
        obs_on_premise_window_status = self.on_premise_window['status'][:, -self.obs_window_size:].flatten()
        
        # クラウドウィンドウの右端から固定長分を取得
        obs_cloud_window_status = self.cloud_window['status'][:, -self.obs_window_size:].flatten()
        
        # ジョブキューを全属性含めてフラット化
        obs_job_queue_obs = self.job_queue[:5].flatten()
        
        # 1次元配列として連結
        observation = np.concatenate([
            obs_on_premise_window_status,
            obs_cloud_window_status,
            obs_job_queue_obs
        ]).astype(np.float32)
        
        return observation

    # コストを計算
    def compute_cost(self, action, allocated_job, is_valid):
        if is_valid:  # actionが有効だった場合
            if action[1] == 0:  # オンプレミスに割り当てる場合
                cost = 0
            elif action[1] == 1:  # クラウドに割り当てる場合
                cost = 1*((allocated_job[0] * allocated_job[1]))  # (処理時間)*(クラウドで使うノード数)をコストとする
                #todo マジックナンバーの解消
            else:  # 割り当てない場合
                cost = 0
        else:  # actionが無効だった場合
            cost = 0  # 平均コストの計算で母数に入れないように

        return cost

    # 割り当て位置を探索（有効性チェックは削除、位置が見つからない場合はNoneを返す）
    def find_allocation_position(self, action, cache_onpre=None, cache_cloud=None):
        """
        割り当て位置を探索
        
        Args:
            action: [method, use_cloud]
            cache_onpre: オンプレミスのキャッシュ（既に取得済みの場合に渡す）
            cache_cloud: クラウドのキャッシュ（既に取得済みの場合に渡す）
        
        Returns:
            (position, wt_real): 位置が見つかった場合は(position, 待ち時間)、見つからない場合は(None, np.inf)
        """
        method = action[0]
        use_cloud = action[1]
        job = self.job_queue[0]
        if method == 0:
            job = self.job_queue[0]

        job_width = int(job[0])
        job_height = int(job[1])
        when_submitted = int(job[-1])
        time = self.time

        # job が 0 なら早期リターン（キャッシュ取得前に早期リターン）
        if job[0] == 0 and job[1] == 0:
            return None, np.inf

        # 使用するウィンドウの選択とキャッシュ取得
        # キャッシュが既に渡されている場合はそれを使用、そうでなければ取得
        if not use_cloud:
            window = self.on_premise_window['status']
            max_h, max_w = self.n_on_premise_node, self.n_window
            if cache_onpre is None:
                cache = self._rebuild_cache_if_needed(use_cloud=False)
            else:
                cache = cache_onpre
        else:
            window = self.cloud_window['status']
            max_h, max_w = self.n_cloud_node, self.n_window
            if cache_cloud is None:
                cache = self._rebuild_cache_if_needed(use_cloud=True)
            else:
                cache = cache_cloud

        # ジョブサイズが大きすぎる場合は早期リターン
        if job_width > max_w or job_height > max_h:
            print("ジョブサイズが大きすぎる")
            return None, np.inf
    
        # 前計算を使った高速探索
        free_per_col = cache['free_per_col']
        ps = cache['prefix_sum']
        free_nodes_list = cache['free_nodes_list']
        occ = cache['occ']

        W = max_w
        need = job_height
        k = job_width
        if k <= 0:
            return None, np.inf

        # スライド最小値（各開始位置での列空き数の最小）: dequeループをNumPyに置換
        if k <= free_per_col.shape[0]:
            # sliding_window_viewでスライディングウィンドウを作成し、各ウィンドウの最小値を計算
            mins = sliding_window_view(free_per_col, k).min(axis=1)  # shape (W-k+1,)
            # minsのインデックスをlimit_aと揃える（mins[0]はa=0に対応）
        else:
            # kが大きすぎる場合は早期リターン
            return None, np.inf

        # 現在列 (a=0) のみ。未来列への先取り予約はしない
        a = 0
        if mins[a] < need:
            return None, np.inf
        a2 = a + k
        max_i = max_h - job_height + 1
        for i in range(max_i):
            i2 = i + job_height
            occ_sum = ps[i2, a2] - ps[i, a2] - ps[i2, a] + ps[i, a]
            if occ_sum == 0:
                return (i, a), time + a - when_submitted

        # クラウドは連続矩形のみ（載せた後はノード変更不可）
        if not use_cloud:
            # 分散割り当て: 開始列でノード集合を固定し、全期間同一ノードを使用
            nodes_at_a = free_nodes_list[a]
            if nodes_at_a.size >= need:
                fixed_nodes = nodes_at_a[:need]
                ok = True
                for col_offset in range(k):
                    col = a + col_offset
                    for node in fixed_nodes:
                        if occ[node, col] != 0:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    fixed_list = fixed_nodes.tolist()
                    node_allocation = [fixed_list] * k
                    return (0, a, node_allocation), time + a - when_submitted

        return None, np.inf

    # エピソード終了条件を判定
    def check_is_done(self):
        # 1エピソードの最大ステップ数に達するか、# 最後のジョブまでジョブキューに格納していた場合、終了する
        # print("index_next_job: ",self.index_next_job)
        # print("len(self.jobs): ",len(self.jobs))
        return self.step_count == self.max_step or (
                self.index_next_job == len(self.jobs) and np.all(self.job_queue == 0)) or self.done_flag

    def get_episode_metrics(self):
        """エピソードごとの待ち時間とコストを計算して返す"""
        average_waiting_time = self.total_waiting_time / self.completed_jobs if self.completed_jobs > 0 else 0
        return average_waiting_time, self.total_cost
    
    def get_windows(self):
        return self.on_premise_window_history_full, self.cloud_window_history_full

    def render_map(self, name):
        """スケジューリング結果をマップとして表示"""
        print("オンプレミスのスケジューリング結果:")
        print(self.on_premise_window_history_full)
        print("\nクラウドのスケジューリング結果:")
        print(self.cloud_window_history_full)
        
        # ジョブリストを作成
        job_list = [{'size': (job[0], job[1])} for job in self.job_allocated]
        
        # マップを可視化
        visualize_map(self.on_premise_window_history_full, self.cloud_window_history_full, job_list, name)

    def save_map(self, filename):
        """現在のマップをファイルに保存"""
        np.savetxt(filename + "_on_premise.csv", self.on_premise_window_user, fmt='%d')
        np.savetxt(filename + "_cloud.csv", self.cloud_window_user, fmt='%d')

    def show_all_maps(self):
        """保存されたすべてのマップを表示"""
        for episode in range(1, 5001, 500):
            on_premise_map = np.loadtxt(f"map_episode_{episode}_on_premise.csv", dtype=int)
            cloud_map = np.loadtxt(f"map_episode_{episode}_cloud.csv", dtype=int)
            job_list = [{'size': (job[0], job[1])} for job in self.job_allocated]
            visualize_map(on_premise_map, cloud_map, job_list)

    def rearrange_resource_map(self, resource_map):
        # 各ノードの右端の位置を計算
        right_edges = [np.max(np.where(row != 0)[0]) if np.any(row != 0) else -1 for row in self.on_premise_window['job_id']]
        
        # 右端の位置に基づいてノードを並び替え
        sorted_indices = np.argsort(right_edges)
        self.on_premise_window['job_id'] = self.on_premise_window['job_id'][sorted_indices]

        right_edges = [np.max(np.where(row != 0)[0]) if np.any(row != 0) else -1 for row in self.cloud_window['job_id']]
        sorted_indices = np.argsort(right_edges)
        self.cloud_window['job_id'] = self.cloud_window['job_id'][sorted_indices]

    def visualize_evaluation_history(self, save_dir="evaluation_history"):
        """評価履歴を可視化し、実際の値（実行時間とコスト）を表示"""
        if not self.evaluation_history:
            print("評価履歴がありません")
            return
        
        # ディレクトリ作成
        import os
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import datetime
        import random
        os.makedirs(save_dir, exist_ok=True)
        
        # 一意のIDを生成
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = f"{timestamp}_{random.randint(1000, 9999)}"
        
        # 実際の値を抽出
        all_actual_values = []
        for history in self.evaluation_history:
            # valuesには[value_cost, value_wt]の配列が保存されている
            actual_values = []
            for val in history['values']:
                actual_values.append(val)  # [コスト, 実行時間]のリスト
            all_actual_values.append(actual_values)
        
        # 表示範囲の計算
        all_x_values = []  # 実行時間
        all_y_values = []  # コスト値
        
        for values_list in all_actual_values:
            for val in values_list:
                # values[1]が実行時間、values[0]がコスト
                all_x_values.append(val[1])  # 実行時間
                all_y_values.append(val[0])  # コスト
        
        # 表示範囲の計算（少しマージンを追加）
        if all_x_values and all_y_values:
            x_min, x_max = min(all_x_values), max(all_x_values)
            y_min, y_max = min(all_y_values), max(all_y_values)
            x_margin = (x_max - x_min) * 0.1 if x_max != x_min else 1.0
            y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
            x_range = [x_min - x_margin, x_max + x_margin]
            y_range = [y_min - y_margin, y_max + y_margin]
        else:
            # デフォルト範囲（データがない場合）
            x_range = [0, 10]
            y_range = [0, 10]
        
        # パレートフロントの進化を可視化（実際の値を使用）
        plt.figure(figsize=(15, 10))
        
        # 各評価時点のパレートフロントをプロット
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.evaluation_history)))
        
        for i, (history, step) in enumerate(zip(self.evaluation_history, self.global_steps_at_evaluation)):
            # 非支配解のインデックスを抽出
            non_dominated_inds = get_non_dominated_inds(np.array(history['all_returns']))
            
            # 非支配解に対応する実際の値を取得
            actual_values = np.array(history['values'])
            pareto_actual_values = actual_values[non_dominated_inds]
            
            # 実際の値をプロット（x軸：実行時間、y軸：コスト）
            plt.scatter(
                [val[1] for val in pareto_actual_values],  # 実行時間
                [val[0] for val in pareto_actual_values],  # コスト
                color=colors[i], 
                label=f"Step {step}",
                alpha=0.7
            )
        
        plt.title("実際の値によるパレートフロントの進化")
        plt.xlabel("実行時間")
        plt.ylabel("コスト")
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()
        
        # 一意のIDを含むファイル名で保存
        pareto_png_filename = f"{save_dir}/pareto_evolution_actual_{unique_id}.png"
        plt.savefig(pareto_png_filename)
        plt.close()
        
        # 最終評価のスケジューリングマップを可視化
        final_maps = self.evaluation_history[-1]['maps']
        final_schedule_filename = f"{save_dir}/final_schedule_{unique_id}.png"
        visualize_map(final_maps[0], final_maps[1], [], final_schedule_filename)
        
        # アニメーション作成（実際の値を使用）
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def update(frame):
            ax.clear()
            history = self.evaluation_history[frame]
            
            # 全ての解の実際の値をプロット
            all_actual_values = np.array(history['values'])
            
            # 非支配解のインデックスを取得
            non_dominated_inds = get_non_dominated_inds(np.array(history['all_returns']))
            pareto_actual_values = all_actual_values[non_dominated_inds]
            
            # 全ての解をプロット
            ax.scatter(
                [val[1] for val in all_actual_values],  # 実行時間
                [val[0] for val in all_actual_values],  # コスト
                alpha=0.3, color='blue', label="全ての解"
            )
            
            # パレートフロントをプロット
            ax.scatter(
                [val[1] for val in pareto_actual_values],  # 実行時間
                [val[0] for val in pareto_actual_values],  # コスト
                color='red', s=80, label="パレートフロント"
            )
            
            ax.set_title(f"Step {self.global_steps_at_evaluation[frame]}での実際の値によるパレートフロント")
            ax.set_xlabel("実行時間")
            ax.set_ylabel("コスト")
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.grid(True)
            ax.legend()
        
        ani = FuncAnimation(fig, update, frames=len(self.evaluation_history), repeat=True)
        
        # 一意のIDを含むファイル名でGIFを保存
        pareto_gif_filename = f"{save_dir}/pareto_animation_actual_{unique_id}.gif"
        ani.save(pareto_gif_filename, writer='pillow', fps=2)
        plt.close()
        
        print(f"実際の値による評価履歴の可視化を保存しました:")
        print(f" - パレートフロント画像: {pareto_png_filename}")
        print(f" - スケジュール画像: {final_schedule_filename}")
        print(f" - アニメーションGIF: {pareto_gif_filename}") 