import json
import numpy as np
import pandas as pd
from src.utils.job_gen.poason_job import JobSimulator


class JobGenerator:
    jobs_set = {}
    job_id = 0
    nb_steps = None
    nb_episodes = None
    max_t = None
    job_type = None
    n_jobs = None
    lam = None
    # envのパラメータ
    n_window = None
    n_on_premise_node = None
    n_cloud_node = None

    # ファイルに出力するジョブのパラメータ(dataframe)
    df_prmj = None

    def __init__(self, seed, nb_steps, n_window, n_on_premise_node, n_cloud_node, config,n_jobs,lam,nb_episodes):
        self.config = config
        self.nb_steps = nb_steps
        self.n_jobs = n_jobs
        # 最大時間を入力
        self.lam = lam
        self.max_t = self.config['param_job']['max_t']
        self.nb_episodes = nb_episodes
        self.seed = seed


        # envのパラメータ
        self.n_window = n_window
        self.n_on_premise_node = n_on_premise_node
        self.n_cloud_node = n_cloud_node

        # ジョブ設定
        # ジョブをエピソードごとに変えるか，固定にするかを指定
        job_is_constant = self.config['param_job']['job_is_constant']

        # ジョブタイプを指定
        if job_is_constant:
            self.job_type = config['param_job']['job_type']
        else:
            self.job_type = int(input(
                'select job-type \n 1:default 2:input file 3:random 4:random(エピソードごとにジョブ数やジョブサイズの範囲を変える) 5:dqnに有利でヒューリスティックに不利な選択が大事なジョブ環境 \n >'))

    # 最終的にenvに渡すジョブのまとまりを生成
    def generate_jobs_set(self):
        if self.job_type == 1:
            jobgen = JobSimulator(self.seed, n_jobs=self.n_jobs, n_users=self.n_jobs, lam=self.lam)
            jobs = jobgen.generate_jobs()
            for episode in range(self.nb_episodes + 1):
                self.jobs_set[episode] = jobs

        # csvファイルからジョブを読み込む
        elif self.job_type == 2:
            jobs = pd.read_csv('jobs.csv')
            jobs = jobs.values.tolist()
            for episode in range(self.nb_episodes + 1):
                self.jobs_set[episode] = jobs

        # ランダムの場合，必要な変数を指定
        elif self.job_type == 3:
            # 最大時間を入力
            # max_t = int(input('input max time \n >'))

            # 単位時間あたりのジョブ数を入力
            n_job_per_time = self.config['param_job']['n_job_per_time']
            n_job_per_time_is_random = False
            if n_job_per_time == -1:
                n_job_per_time_is_random = True

            # 最大処理時間を入力
            max_processing_time = self.config['param_job']['max_processing_time']
            max_processing_time_is_random = False
            if max_processing_time == -1:
                max_processing_time_is_random = True

            # 最大要求ノード数を入力
            max_n_required_nodes = self.config['param_job']['max_n_required_nodes']
            max_n_required_nodes_is_random = False
            if max_n_required_nodes == -1:
                max_n_required_nodes_is_random = True

            nb_jobs_per_user = self.config['param_job']['nb_jobs_per_user']

            #ユーザ数を入力
            max_user_id = self.config['param_job']['max_user_id']-1

            user_increment = self.config['param_job']['user_increment']


            #waiting_timeの初期値は-1
            waiting_time = -1 

            prm_job = [self.nb_steps, self.max_t, n_job_per_time, max_processing_time, max_n_required_nodes, max_user_id]
            self.df_prmj = pd.DataFrame(
                [prm_job],
                columns=['n_steps', 'max_time', 'n_job_per_time', 'max_processing_time', 'max_n_required_nodes', 'max_user_id'],
                index=['idx']
            )

            if user_increment:

                for episode in range(self.nb_episodes): 
                    user_id = 1
                    jobs = []
                    self.job_id = 0
                    user_count = 0
                    for i in range(self.max_t + 1):
                        if n_job_per_time_is_random:
                            n_job_per_time = np.random.randint(1, 6)
                        for _ in range(n_job_per_time):
                            submit_time = i

                            # 通常サイズのジョブ(デフォルト仕様)
                            processing_time = np.random.randint(1, max_processing_time + 1)
                            n_required_nodes = np.random.randint(1, max_n_required_nodes + 1)
                            #ユーザIDをランダムに割り振る。
                            user_count += 1


                            can_use_cloud = 1
                            jobs.append(
                                self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud, user_id)
                            )

                            if user_count == n_job_per_time*self.max_t //max_user_id:
                                user_id += 1
                                user_count = 0

                    self.jobs_set[episode] = jobs
                
            else:
                for episode in range(self.nb_episodes): 
                    jobs = []
                    self.job_id = 0
                    user_count = 0
                    for i in range(self.max_t + 1):
                        if n_job_per_time_is_random:
                            n_job_per_time = np.random.randint(1, 6)
                        for _ in range(n_job_per_time):
                            submit_time = i

                            # 通常サイズのジョブ(デフォルト仕様)
                            processing_time = np.random.randint(1, max_processing_time + 1)
                            n_required_nodes = np.random.randint(1, max_n_required_nodes + 1)
                            #ユーザIDをランダムに割り振る。
                            user_id = np.random.randint(1, max_user_id+2)



                            can_use_cloud = 1
                            jobs.append(
                                self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud, user_id)
                            )


                    self.jobs_set[episode] = jobs


        # 途中で性質が切り替わる場合の設定
        elif self.job_type == 4:
            random_job_type = int(input(
                'select random job type \n 1:a few small jobs -> a few big jobs -> a few small jobs(only on-premise many '
                '\n 2:n_job gradually go up) \n 3:小 -> 中 -> 大 \n 4:n_job 2->3->2->3->4 \n 5: njob:2, '
                'maxsize:2×2->3×3->2×2->3×3->4×4 \n 6: njob:2, maxsize:2×2->3×3->4×4 \n 7: njob:2, maxsize:2×2->2×2->3×3 '
                '\n 8: njob:2->2->3 \n 9: njob:1->2->3 \n >'
            ))

            # 小少 -> 大少 -> オンプレミスのみ多
            if random_job_type == 1:
                points_switch_random_job_type = [self.nb_episodes // 3, self.nb_episodes // 3 * 2]
                for episode in range(self.nb_episodes + 1):
                    jobs = []
                    self.job_id = 0

                    # 小
                    if episode < points_switch_random_job_type[0]:
                        n_job_per_time = 3
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # 大
                    elif episode < points_switch_random_job_type[1]:
                        n_job_per_time = 2
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(3, 5)
                                n_required_nodes = np.random.randint(3, 5)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # 小(オンプレミスのみが多め)
                    else:
                        n_job_per_time = 3
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                # 20%の確率でクラウド使用可
                                if np.random.uniform() <= 0.2:
                                    can_use_cloud = 1
                                else:
                                    can_use_cloud = 0
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    self.jobs_set[episode] = jobs

            # ジョブ数：3 -> 4 -> 5
            elif random_job_type == 2:
                # nb_episodesを再設定
                nb_episodes = self.nb_steps // (self.max_t * 4)

                points_switch_random_job_type = [nb_episodes // 3, nb_episodes // 3 * 2]
                for episode in range(nb_episodes + 1):
                    jobs = []
                    self.job_id = 0

                    # ジョブ数=3
                    if episode < points_switch_random_job_type[0]:
                        n_job_per_time = 3
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # ジョブ数=4
                    elif episode < points_switch_random_job_type[1]:
                        n_job_per_time = 4
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # ジョブ数=5
                    else:
                        n_job_per_time = 5
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    self.jobs_set[episode] = jobs

            # 小 -> 中 -> 大
            elif random_job_type == 3:
                points_switch_random_job_type = [self.nb_episodes // 3, self.nb_episodes // 3 * 2]
                for episode in range(self.nb_episodes + 1):
                    jobs = []
                    self.job_id = 0

                    # 小
                    if episode < points_switch_random_job_type[0]:
                        n_job_per_time = 3
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # 中
                    elif episode < points_switch_random_job_type[1]:
                        n_job_per_time = 3
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(2, 4)
                                n_required_nodes = np.random.randint(2, 4)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # 大
                    else:
                        n_job_per_time = 2
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(3, 5)
                                n_required_nodes = np.random.randint(3, 5)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    self.jobs_set[episode] = jobs

                prm_job = [self.nb_steps, self.max_t, points_switch_random_job_type[0], points_switch_random_job_type[1]]
                self.df_prmj = pd.DataFrame(
                    [prm_job],
                    columns=['n_steps', 'max_time', 'point_switch_job_type1', 'point_switch_job_type2'],
                    index=['idx']
                )

            # ジョブ数：2 -> 3 -> 2 -> 3 -> 4
            elif random_job_type == 4:
                # nb_episodesを再設定
                # nb_episodes = nb_steps // (max_t * 3)
                nb_episodes = 50000

                # points_switch_random_job_type = [nb_episodes // 6, nb_episodes // 6 * 2, nb_episodes // 6 * 3, nb_episodes // 6 * 4]
                points_switch_random_job_type = [7500, 15000, 20000, 25000]
                for episode in range(nb_episodes + 1):
                    jobs = []
                    self.job_id = 0

                    # ジョブ数=2(事前学習)
                    if episode < points_switch_random_job_type[0]:
                        n_job_per_time = 2
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # ジョブ数=3(事前学習)
                    elif episode < points_switch_random_job_type[1]:
                        n_job_per_time = 3
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # ジョブ数=2(経験済み)
                    elif episode < points_switch_random_job_type[2]:
                        n_job_per_time = 2
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # ジョブ数=3(経験済み)
                    elif episode < points_switch_random_job_type[3]:
                        n_job_per_time = 3
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # ジョブ数=4(未経験)
                    else:
                        n_job_per_time = 4
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    self.jobs_set[episode] = jobs

                prm_job = [self.nb_steps, self.max_t, points_switch_random_job_type[0], points_switch_random_job_type[1],
                           points_switch_random_job_type[2], points_switch_random_job_type[3]]
                self.df_prmj = pd.DataFrame(
                    [prm_job],
                    columns=['n_steps', 'max_time', 'point_switch_job_type1', 'point_switch_job_type2',
                             'point_switch_job_type3', 'point_switch_job_type4'],
                    index=['idx']
                )

            # njob:2, maxsize:2×2->3×3->2×2->3×3->4×4
            elif random_job_type == 5:
                # nb_episodesを再設定
                # nb_episodes = nb_steps // (max_t * 3)
                nb_episodes = 60000

                # points_switch_random_job_type = [nb_episodes // 6, nb_episodes // 6 * 2, nb_episodes // 6 * 3, nb_episodes // 6 * 4]
                points_switch_random_job_type = [7500, 15000, 20000, 25000]
                for episode in range(nb_episodes + 1):
                    jobs = []
                    self.job_id = 0
                    n_job_per_time = 2  # njob=2で固定

                    # maxsize:2×2(事前学習)
                    if episode < points_switch_random_job_type[0]:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # maxsize:3×3(事前学習)
                    elif episode < points_switch_random_job_type[1]:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 4)
                                n_required_nodes = np.random.randint(1, 4)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # maxsize:2×2(経験済み)
                    elif episode < points_switch_random_job_type[2]:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # maxsize:3×3(経験済み)
                    elif episode < points_switch_random_job_type[3]:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 4)
                                n_required_nodes = np.random.randint(1, 4)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # maxsize:4×4(未経験)
                    else:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 5)
                                n_required_nodes = np.random.randint(1, 5)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    self.jobs_set[episode] = jobs

                prm_job = [self.nb_steps, self.max_t, points_switch_random_job_type[0], points_switch_random_job_type[1],
                           points_switch_random_job_type[2], points_switch_random_job_type[3]]
                self.df_prmj = pd.DataFrame(
                    [prm_job],
                    columns=['n_steps', 'max_time', 'point_switch_job_type1', 'point_switch_job_type2',
                             'point_switch_job_type3', 'point_switch_job_type4'],
                    index=['idx']
                )

            # njob:2, maxsize:2×2->3×3->4×4
            elif random_job_type == 6:
                # nb_episodesを再設定
                # nb_episodes = nb_steps // (max_t * 3)
                nb_episodes = 50000

                # points_switch_random_job_type = [nb_episodes // 6, nb_episodes // 6 * 2, nb_episodes // 6 * 3, nb_episodes // 6 * 4]
                points_switch_random_job_type = [7500, 15000]
                for episode in range(nb_episodes + 1):
                    jobs = []
                    self.job_id = 0
                    n_job_per_time = 2  # njob=2で固定

                    # maxsize:2×2
                    if episode < points_switch_random_job_type[0]:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # maxsize:3×3
                    elif episode < points_switch_random_job_type[1]:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 4)
                                n_required_nodes = np.random.randint(1, 4)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # maxsize:4×4
                    else:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 5)
                                n_required_nodes = np.random.randint(1, 5)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    self.jobs_set[episode] = jobs

                prm_job = [self.nb_steps, self.max_t, points_switch_random_job_type[0], points_switch_random_job_type[1]]
                self.df_prmj = pd.DataFrame(
                    [prm_job],
                    columns=['n_steps', 'max_time', 'point_switch_job_type1', 'point_switch_job_type2'],
                    index=['idx']
                )

            # njob:2, maxsize:2×2->2×2->3×3
            elif random_job_type == 7:
                n_job_per_time = 2  # njob=2で固定
                # nb_episodesを再設定
                # nb_episodes = nb_steps // (max_t * n_job_per_time)
                nb_episodes = 70000

                # points_switch_random_job_type = [nb_episodes // 6, nb_episodes // 6 * 2, nb_episodes // 6 * 3, nb_episodes // 6 * 4]
                points_switch_random_job_type = [10000, 20000]
                for episode in range(nb_episodes + 1):
                    jobs = []
                    self.job_id = 0

                    # maxsize:2×2
                    if episode < points_switch_random_job_type[0]:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # maxsize:2×2
                    elif episode < points_switch_random_job_type[1]:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # maxsize:3×3
                    else:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 4)
                                n_required_nodes = np.random.randint(1, 4)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    self.jobs_set[episode] = jobs

                prm_job = [self.nb_steps, self.max_t, points_switch_random_job_type[0], points_switch_random_job_type[1]]
                self.df_prmj = pd.DataFrame(
                    [prm_job],
                    columns=['n_steps', 'max_time', 'point_switch_job_type1', 'point_switch_job_type2'],
                    index=['idx']
                )

            # njob:2->2->3
            elif random_job_type == 8:
                # nb_episodesを再設定
                # nb_episodes = nb_steps // (max_t * n_job_per_time)
                nb_episodes = 70000

                # points_switch_random_job_type = [nb_episodes // 6, nb_episodes // 6 * 2, nb_episodes // 6 * 3, nb_episodes // 6 * 4]
                points_switch_random_job_type = [10000, 20000]
                for episode in range(nb_episodes + 1):
                    jobs = []
                    self.job_id = 0

                    # njob=2
                    if episode < points_switch_random_job_type[0]:
                        n_job_per_time = 2
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # njob=2
                    elif episode < points_switch_random_job_type[1]:
                        n_job_per_time = 2
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # njob=3
                    else:
                        n_job_per_time = 3
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    self.jobs_set[episode] = jobs

                prm_job = [self.nb_steps, self.max_t, points_switch_random_job_type[0], points_switch_random_job_type[1]]
                self.df_prmj = pd.DataFrame(
                    [prm_job],
                    columns=['n_steps', 'max_time', 'point_switch_job_type1', 'point_switch_job_type2'],
                    index=['idx']
                )

            # njob:1->2->3
            elif random_job_type == 9:
                # nb_episodesを再設定
                # nb_episodes = nb_steps // (max_t * n_job_per_time)
                nb_episodes = 70000

                # points_switch_random_job_type = [nb_episodes // 6, nb_episodes // 6 * 2, nb_episodes // 6 * 3, nb_episodes // 6 * 4]
                points_switch_random_job_type = [10000, 20000]
                for episode in range(nb_episodes + 1):
                    jobs = []
                    self.job_id = 0

                    # njob=1
                    if episode < points_switch_random_job_type[0]:
                        n_job_per_time = 1
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # njob=2
                    elif episode < points_switch_random_job_type[1]:
                        n_job_per_time = 2
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # njob=3
                    else:
                        n_job_per_time = 3
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    self.jobs_set[episode] = jobs

                prm_job = [self.nb_steps, self.max_t, points_switch_random_job_type[0], points_switch_random_job_type[1]]
                self.df_prmj = pd.DataFrame(
                    [prm_job],
                    columns=['n_steps', 'max_time', 'point_switch_job_type1', 'point_switch_job_type2'],
                    index=['idx']
                )


        # 処理するジョブを選択できるdqnに有利でヒューリスティックに不利なジョブを生成
        elif self.job_type == 5:
            # 最大時間を入力
            #         max_t = int(input('input max time \n >'))
            # 単位時間あたりのジョブ数を入力
            while True:
                n_job_per_time = int(input('input number of jobs per time (-1:random) \n >'))
                n_job_per_time_is_random = False
                if 1 <= n_job_per_time <= 10:
                    break
                elif n_job_per_time == -1:
                    n_job_per_time_is_random = True
                    break
                else:
                    print('input again')

            for episode in range(self.nb_episodes + 1):
                jobs = []
                self.job_id = 0
                for i in range(self.max_t + 1):
                    if n_job_per_time_is_random:
                        n_job_per_time = np.random.randint(1, 6)
                    if np.random.uniform() < 0.3:  # たまに選択が大事なジョブ群がくる
                        for j in range(n_job_per_time):
                            submit_time = i
                            if j == 0:  # 一つ目は
                                # オンプレミスをほとんど埋めるジョブ
                                processing_time = self.n_window - 1
                                n_required_nodes = self.n_on_premise_node - 1
                                can_use_cloud = np.random.randint(0, 2)
                            else:
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = 0
                            jobs.append(
                                self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                            )

                    else:
                        for j in range(n_job_per_time):
                            submit_time = i
                            processing_time = np.random.randint(1, 3)
                            n_required_nodes = np.random.randint(1, 3)
                            can_use_cloud = np.random.randint(0, 2)
                            jobs.append(
                                self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                            )

                self.jobs_set[episode] = jobs

        return self.jobs_set

    # ジョブ単体を生成
    def generate_job(self, submit_time, processing_time, n_required_nodes, can_use_cloud, user_id,  waiting_time=-1):
        job = [submit_time, processing_time, n_required_nodes, can_use_cloud, user_id, self.job_id, waiting_time,0]
        self.job_id += 1

        return job

    # def save_jobs_set(self, path):
    #     with open(path, 'w') as f:
    #         json.dump(self.jobs_set, f, indent=5)

    # if __name__ == '__main__':
    #     #jobを生成
    #     job_generator = JobGenerator()
    #     job_generator.generate_jobs_set()