import gym
import numpy as np
from collections import deque
import sys
from sklearn.preprocessing import MinMaxScaler
import csv
from numba import jit, njit

from gym.envs.registration import EnvSpec
from gym import spaces
from src.utils.map_visualizer import visualize_map
# import cupy as cp

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

        # オンプレミスとクラウドのスライドウィンドウを構造化配列で管理
        self.on_premise_window = np.zeros((self.n_on_premise_node, self.n_window), dtype=self.dtype)
        self.cloud_window = np.zeros((self.n_cloud_node, self.n_window), dtype=self.dtype)
        # 初期化

        self.n_action = 2  # 行動数
        self.action_space = gym.spaces.Discrete(self.n_action)  # 行動空間
        self.tmp_queue = deque()  # 割り当てられなかったジョブを一時的に格納するキュー
        obs_space_size = (self.n_on_premise_node * self.n_window +
                          self.n_cloud_node * self.n_window +
                          4 * self.n_job_queue_obs + 1)
        self.observation_space = spaces.Discrete(obs_space_size)
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



    # 各ステップで実行される操作
    def step(self, action_raw):
        while True:
            # print(self.job_queue)
            # print(f"Current exe_mode: {exe_mode}")  # デバッグ用
            scheduled = False

            valid_action_cache = {}


            time_reward_new = 0
            # self.user_wt = [[i,-100] for i in range(100)]
            # 時刻
            time = self.time
            allocated_job = self.job_queue[0]
            action = self.get_converted_action(action_raw)   
            wt_step = 0
            is_valid = False  # actionが有効かどうか

            # print(self.on_premise_window)

            # self.rearrange_window()
            # print(self.on_premise_window)

            is_valid, wt_real, position = self.check_is_valid_action(action)
            # print("job_queue head: ",self.job_queue[0])
            # print("is_valid: ",is_valid)

            if is_valid == False:  # すきまがない場合or job_queueが空の場合
                # print("allocated_job: ",allocated_job)
                if np.all(allocated_job == 0):  # ジョブキューが空の場合
                    job_none = True
                    # 全体的に時間を進める必要がある場合は両方スライド
                    self.time_transition(True, True)
                else:  # ジョブキューが空でない場合
                    job_none = False
                    # どちらのリソースへの割当てが失敗したかを判断
                    if action[1] == 0:  # オンプレミスへの割当てが失敗した場合
                        self.time_transition(True, False)  # オンプレミスのみスライド
                    else:  # クラウドへの割当てが失敗した場合
                        self.time_transition(False, True)  # クラウドのみスライド
                # print('is_valid: ' + str(is_valid))
                # print("time_transition")

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
                # print('valid action')


                if action[1] == 0:
                    if (0,1) in valid_action_cache:
                        is_valid_parallel,wt_parallel,piyo = valid_action_cache[(0,1)]
                    else:
                        is_valid_parallel,wt_parallel,piyo = self.check_is_valid_action([0,1])
                        valid_action_cache[(0,1)] = (is_valid_parallel,wt_parallel,piyo)
                if action[1] == 1:
                    if (0,0) in valid_action_cache:
                        is_valid_parallel,wt_parallel,piyo = valid_action_cache[(0,0)]
                    else:
                        is_valid_parallel,wt_parallel,piyo = self.check_is_valid_action([0,0])
                        valid_action_cache[(0,0)] = (is_valid_parallel,wt_parallel,piyo)

                # 並列実行時の待ち時間と実際の待ち時間を比較して報酬を決定
                time_reward_new = (
                    1 if wt_real < wt_parallel else  # 実際の方が短い
                    0 if wt_real == wt_parallel else # 同じ待ち時間
                    -1                               # 実際の方が長い
                )
                # print("time_reward_new: ",time_reward_new)

                if action[0] == 0:  # FIFO
                    # print('job_queue before fio\n',self.job_queue)
                    wt_step = self.do_schedule(action,job,position)
                    scheduled = True
                    # print("wt_step: ",wt_step)
                    # print('schedule fifo')
                    #ジョブキューをスライド
                    self.job_queue = np.roll(self.job_queue, -1, axis=0)
                    self.job_queue[-1] = 0
                    self.rear_job_queue -= 1

                    # 観測データ(状態)を取得
                    observation = self.get_observation()
                    # 報酬を取得
                    # reward = self.get_reward(action, allocated_job, time, is_valid, job_none)
                    # コストを取得
                    cost = self.compute_cost(action, allocated_job, is_valid)

                    # エピソードの終了時刻を満たしているかの判定
                    done = self.check_is_done()
                    # done = False
                    # print("done: ",done)

                    # print("self.job_queue: \n",self.job_queue)
                    
                    # self.show_final_window_history()

                    # input()

                    # print("time: ",time)
                    # print('================================================')
                    # print('self.user_wt: ',self.user_wt)

                    rewards = np.array([time_reward_new,cost])

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

        self.on_premise_window_history_full = np.full((self.n_on_premise_node,1),-1)
        self.cloud_window_history_full = np.full((self.n_cloud_node,1),-1)

        # self.cloud_window_user = np.full((self.n_cloud_node, self.n_window),-1)  # クラウドのスライドウィンドウ
        self.job_queue = np.zeros((len(self.jobs),8)) # ジョブキュー
        self.rear_job_queue = 0  # ジョブキューの末尾 (== 0: ジョブキューが空)
        self.tmp_queue = deque()  # 割り当てられなかったジョブを一時的に格納するキュー
        self.user_wt_log = []


        self.total_waiting_time = 0
        self.total_cost = 0
        self.completed_jobs = 0

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

        # 構造化配列からndarrayを取得
        on_premise_status = self.on_premise_window['status']
        on_premise_job_id = self.on_premise_window['job_id']
        cloud_status = self.cloud_window['status']
        cloud_job_id = self.cloud_window['job_id']
    
        # Numbaで高速化された関数を呼び出し
        on_premise_status, on_premise_job_id, cloud_status, cloud_job_id = time_transition_njit(
            on_premise_status, on_premise_job_id, cloud_status, cloud_job_id,
            slide_on_premise, slide_cloud
        )

        # 新しいジョブをジョブキューに追加
        self.append_new_job2job_queue()

    def update_window_history(self):
        # オンプレミスの履歴を更新
        left_column_on_premise = self.on_premise_window['job_id'][:, 0:1]  # 左端の1列を取得
        self.on_premise_window_history_full = np.hstack((self.on_premise_window_history_full, left_column_on_premise.copy()))  # 左端の1列を履歴に追加
        
        # クラウドの履歴を更新
        left_column_cloud = self.cloud_window['job_id'][:, 0:1]  # 左端の1列を取得
        self.cloud_window_history_full = np.hstack((self.cloud_window_history_full, left_column_cloud.copy()))  # 左端の1列を履歴に追加

        # print("self.on_premise_window_history_full:\n",self.on_premise_window_history_full)
        # print("self.cloud_window_history_full:\n",self.cloud_window_history_full)

    def finalize_window_history(self):
        """ウィンドウ全体を履歴に追加"""
        # print("finalize_window_history")
        self.on_premise_window_history_full = np.hstack((self.on_premise_window_history_full, self.on_premise_window['job_id'].copy()))
        self.cloud_window_history_full = np.hstack((self.cloud_window_history_full, self.cloud_window['job_id'].copy()))
        #一番左の列を削除
        self.on_premise_window_history_full = np.delete(self.on_premise_window_history_full, 0, axis=1)
        self.cloud_window_history_full = np.delete(self.cloud_window_history_full, 0, axis=1)

        # 待ち時間とコストを計算
        self.cost = self.calc_objective_values()

    def get_cost(self):
        return self.cost

    def calc_objective_values(self):
        # 待ち時間とコストを計算　

        """return:cost,makespan 待ち時間の定義は，ジョブが到着してから，ジョブが開始するまでの時間．
        つまり，各ステップにおける遅延時間の総和をとればよい．
        """
        # self.waiting_time = 
        
        #costはcloud_window_history_fullで-1じゃないマスを総計すればいい．ただし，next_init_windowsがある場合は，その分を引く．
        self.cost = np.count_nonzero(self.cloud_window_history_full[self.cloud_window_history_full != -1])
        #makespanは二次元配列で要素が入っているものの中で一番右側（インデックス）の値をとる
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

        return self.cost, max(mkspan_onpre, mkspan_cloud)

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
        
        
        return self.time - when_submitted

    # 観測データ(状態)を取得
    def get_observation(self):
        # オンプレミスとクラウドのウィンドウの最後の10行を取得
        obs_on_premise_window_status = self.on_premise_window['status'].flatten()

        # obs_on_premise_window_job_id = self.on_premise_window['job_id'].flatten()
        obs_cloud_window_status = self.cloud_window['status'].flatten()
        # obs_cloud_window_job_id = self.cloud_window['job_id'].flatten()

        # ジョブキューの観測部分の観測データ
        obs_job_queue_obs = self.job_queue[:10, :4].flatten()

        observation = np.concatenate([
            obs_on_premise_window_status,
            obs_cloud_window_status,
            obs_job_queue_obs
        ]).astype(np.float32)

        return observation

    # 報酬を取得
    def get_reward(self, action, allocated_job, time, is_valid, job_none):
        reward_liner = 0
        reward_wt = 0
        reward_cost = 0
        reward = [0,0]
        use_cloud = action[1] 
        if job_none == False:  # ジョブキューが空でない場合            
            if is_valid:  # actionが有効だった場合
                submitted_time = allocated_job[-1]

                # 割り当てたジョブの待ち時間
                # print(self.time)
                waiting_time = self.time
                reward_wt = 100 - waiting_time

                # penalty = min(waiting_time/self.n_window, 2) + use_cloud
                # reward_liner = weight_cost * (1 - use_cloud * 2) + weight_wt * (1 - waiting_time / self.n_window)
                # reward_wt = 1 - waiting_time / self.n_window
                # reward_wt = - waiting_time
                # reward_cost = 1 - use_cloud



        # #cost or waitingtimeがゼロになるときはrewardを0にする
        # if reward_cost == 0 or reward_wt == 0:
        #     reward = [0,0]
        # else:
        #     reward = [1/(reward_cost), 1/(reward_wt)]
            
        #cost or waitingtimeがゼロになるときはrewardを0にする
        reward =[0, reward_wt] 
        # reward = [reward_liner, reward_liner]

        return reward

    # コストを計算
    def compute_cost(self, action, allocated_job, is_valid):
        if is_valid:  # actionが有効だった場合
            if action[1] == 0:  # オンプレミスに割り当てる場合
                cost = 0
            elif action[1] == 1:  # クラウドに割り当てる場合
                cost = -1*((allocated_job[0] * allocated_job[1]))  # (処理時間)*(クラウドで使うノード数)をコストとする
                #todo マジックナンバーの解消
            else:  # 割り当てない場合
                cost = 0
        else:  # actionが無効だった場合
            cost = 0  # 平均コストの計算で母数に入れないように

        return cost

    # stateをもとにactionが有効かどうかを判定
    def check_is_valid_action(self, action):
        method = action[0]
        use_cloud = action[1]
        job = self.job_queue[0]
        if method == 0:
            job = self.job_queue[0]

        job_width = int(job[0])
        job_height = int(job[1])
        when_submitted = int(job[-1])
        time = self.time

        # job が 0 なら早期リターン
        if job[0] == 0 and job[1] == 0:
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
                
                # 連続した領域がない場合、分散して割り当てる（より複雑な実装が必要）
                # ここでは簡易的に、各列で上から順に利用可能なノードを見つける
                node_allocation = []
                for col_offset in range(job_width):
                    col = a + col_offset
                    free_nodes = [i for i in range(max_h) if window[i, col] == 0][:job_height]
                    if len(free_nodes) >= job_height:
                        node_allocation.append(free_nodes[:job_height])
                    else:
                        # 十分なノードが見つからない場合はスキップ
                        can_allocate = False
                        break
                
                if can_allocate:
                    # 最初のノードの位置を返す（簡易実装）
                    # 実際の割り当ては do_schedule で行う必要がある
                    return True, time + a - when_submitted, (0, a, node_allocation)
    
        # 有効な配置位置が見つからなかった
        return False, np.inf, None

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
        os.makedirs(save_dir, exist_ok=True)
        
        # 一意のIDを生成
        import datetime
        import random
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