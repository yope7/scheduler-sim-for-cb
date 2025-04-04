# スケジューリング環境の1エピソード実行と結果の可視化
import numpy as np
import yaml
import json
import os
from datetime import datetime
from src.utils.job_gen.job_generator import JobGenerator
from multiEnv2 import SchedulingEnv
from PCNagent2 import PCN

# 出力を改行なしで表示
np.set_printoptions(linewidth=np.inf)

class SchedulingVisualizer:
    def __init__(self):
        """初期化"""
        self.output_dir = "visualization_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run_single_episode(self, nb_jobs=50, algorithm="pcn"):
        """
        1エピソードを実行し、結果をJSONで保存
        
        引数:
            nb_jobs (int): ジョブ数
            algorithm (str): 使用するアルゴリズム ("pcn" または "random")
        """
        # 設定ファイルの読み込み
        with open("config.yml", "r") as yml:
            config = yaml.safe_load(yml)
            
        # 環境パラメータの取得
        max_step = np.inf
        n_window = config['param_env']['n_window']
        n_on_premise_node = config['param_env']['n_on_premise_node']
        n_cloud_node = config['param_env']['n_cloud_node']
        n_job_queue_obs = config['param_env']['n_job_queue_obs']
        n_job_queue_bck = config['param_env']['n_job_queue_bck']
        weight_wt = config['param_agent']['weight_wt']
        weight_cost = config['param_agent']['weight_cost']
        penalty_not_allocate = config['param_env']['penalty_not_allocate']
        penalty_invalid_action = config['param_env']['penalty_invalid_action']
        nb_steps = config['param_simulation']['nb_steps']
        
        # ジョブセットの生成
        print(f"ジョブセット生成中...")
        job_generator = JobGenerator(0, nb_steps, n_window, n_on_premise_node, n_cloud_node, config, nb_jobs, 0.3, 10)
        jobs_set = job_generator.generate_jobs_set()
        print(f"ジョブセット生成完了: {len(jobs_set)}個のジョブ")
        
        # 環境の初期化
        env = SchedulingEnv(
            max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck,
            weight_wt, weight_cost, penalty_not_allocate, penalty_invalid_action, jobs_set,
            next_init_windows=None, flag=0, exe_mode=0
        )
        
        # アルゴリズムに基づいてエピソードを実行
        if algorithm == "pcn":
            results = self._run_pcn_episode(env, nb_jobs)
        else:
            results = self._run_random_episode(env, nb_jobs)
            
        # 結果を保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scheduling_results_{algorithm}_{nb_jobs}jobs_{timestamp}.json"
        self._save_results(results, filename)
        
        # HTMLファイルのパスを返す
        html_path = os.path.join("visualization", "scheduler_visualizer.html")
        print(f"結果を保存しました。以下のHTMLファイルで可視化できます: {html_path}")
        print(f"データファイル: {os.path.join(self.output_dir, filename)}")
        
        return os.path.join(self.output_dir, filename)
    
    def _run_pcn_episode(self, env, nb_jobs):
        """PCNエージェントを使用して1エピソードを実行"""
        print("PCNエージェントを初期化中...")
        agent = PCN(
            env,
            device="auto",
            scaling_factor=np.array([1, 1, 1]),
            learning_rate=1e-3,
            batch_size=256,
            hidden_dim=256,
            project_name="temp",
            experiment_name="PCN_1episode",
            log=True,
        )
        
        # 1エピソードだけ学習
        print("1エピソード学習中...")
        agent.train(
            eval_env=env,
            total_timesteps=1,  # 1エピソードのみ
            ref_point=np.array([-1000, -1000]),
            num_er_episodes=1,
            num_step_episodes=1,
            num_model_updates=1,
            max_buffer_size=100,
            known_pareto_front=[1, 1],
        )
        
        # 結果の収集
        on_premise_map, cloud_map = env.get_windows()
        average_waiting_time, total_cost = env.get_episode_metrics()
        
        # 結果をまとめる
        results = {
            'algorithm': 'PCN',
            'nb_jobs': nb_jobs,
            'metrics': {
                'average_waiting_time': float(average_waiting_time),
                'total_cost': float(total_cost)
            },
            'maps': {
                'on_premise': on_premise_map.tolist(),
                'cloud': cloud_map.tolist()
            },
            'jobs': [job if isinstance(job, list) else job.tolist() if hasattr(job, 'tolist') else list(job) for job in env.jobs_set]
        }
        
        return results
    
    def _run_random_episode(self, env, nb_jobs):
        """ランダム方策で1エピソードを実行"""
        print("ランダム方策でエピソード実行中...")
        obs = env.reset()
        done = False
        step = 0
        
        # マップの履歴を記録
        map_history = {
            'on_premise': [],
            'cloud': []
        }
        
        while not done and step < nb_jobs:
            # ランダムな行動を選択
            action = env.action_space.sample()
            
            # 環境ステップの実行
            obs, rewards, scheduled, wt_step, _, done, _ = env.step(action)
            
            if scheduled:
                step += 1
                # 現在のマップ状態を記録
                on_premise, cloud = env.get_windows()
                map_history['on_premise'].append(on_premise.tolist())
                map_history['cloud'].append(cloud.tolist())
        
        # 最終状態を記録
        on_premise_map, cloud_map = env.get_windows()
        average_waiting_time, total_cost = env.get_episode_metrics()
        
        # 結果をまとめる
        results = {
            'algorithm': 'Random',
            'nb_jobs': nb_jobs,
            'metrics': {
                'average_waiting_time': float(average_waiting_time),
                'total_cost': float(total_cost)
            },
            'maps': {
                'on_premise': on_premise_map.tolist(),
                'cloud': cloud_map.tolist()
            },
            'map_history': map_history,
            'jobs': [job if isinstance(job, list) else job.tolist() if hasattr(job, 'tolist') else list(job) for job in env.jobs_set]
        }
        
        return results
    
    def _save_results(self, results, filename):
        """結果をJSONファイルに保存"""
        json_path = os.path.join(self.output_dir, filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"結果を保存しました: {json_path}")

if __name__ == "__main__":
    visualizer = SchedulingVisualizer()
    
    # コマンドライン引数の代わりに直接指定
    nb_jobs = 20  # ジョブ数
    algorithm = "random"  # "pcn" または "random"
    
    # 1エピソード実行して結果を保存
    visualizer.run_single_episode(nb_jobs, algorithm)
