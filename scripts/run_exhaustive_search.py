import itertools
import yaml
import numpy as np
import json
import os
from datetime import datetime
from src.utils.job_gen.job_generator import JobGenerator
from multiEnv2 import SchedulingEnv
import matplotlib.pyplot as plt

# printで改行しない
np.set_printoptions(linewidth=np.inf)

class SchedulingExplorer:
    """ジョブスケジューリングの全探索を行う実装"""
    def __init__(self):
        """初期化"""
        self.output_dir = "exploration_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run_exhaustive_search(self, nb_jobs, visualize=True, debug=False):
        """
        指定したジョブ数に対して全組み合わせのスケジューリングを実行
        
        引数:
            nb_jobs (int): ジョブ数
            visualize (bool): 可視化するかどうか
            debug (bool): デバッグ情報を出力するかどうか
            
        戻り値:
            str: 結果ファイルのパス
        """
        # 設定ファイルの読み込み
        with open("config.yml", "r") as yml:
            config = yaml.safe_load(yml)
        
        # 環境パラメータの取得
        env_params = {
            'n_window': config["param_env"]["n_window"],
            'n_on_premise_node': config["param_env"]["n_on_premise_node"],
            'n_cloud_node': config["param_env"]["n_cloud_node"],
            'n_job_queue_obs': config["param_env"]["n_job_queue_obs"],
            'n_job_queue_bck': config["param_env"]["n_job_queue_bck"],
            'weight_wt': config["param_agent"]["weight_wt"],
            'weight_cost': config["param_agent"]["weight_cost"],
            'penalty_not_allocate': config["param_env"]["penalty_not_allocate"],
            'penalty_invalid_action': config["param_env"]["penalty_invalid_action"],
            'nb_steps': config["param_simulation"]["nb_steps"]
        }
        
        # ジョブセットの生成
        print("ジョブセット生成中...")
        lam = 0.3  # ジョブ生成のパラメータ
        job_generator = JobGenerator(
            1, env_params['nb_steps'], env_params['n_window'], 
            env_params['n_on_premise_node'], env_params['n_cloud_node'], 
            config, nb_jobs, lam, 0
        )
        jobs_set = job_generator.generate_jobs_set()
        print(f"ジョブセット生成完了: {len(jobs_set[0])}個のジョブ")
        
        if debug:
            print("生成されたジョブセット:")
            for i, job in enumerate(jobs_set[0]):
                print(f"ジョブ {i}: {job} (幅:{job[0]}, 高さ:{job[1]}, クラウド使用可:{job[3]}, ID:{job[4]})")
        
        # アクションの全組み合わせを生成
        # 各ジョブに対して0（オンプレミス）または1（クラウド）のアクションを設定
        all_action_sets = list(itertools.product([0, 1], repeat=nb_jobs))
        total_combinations = len(all_action_sets)
        print(f"アクションセット数: {total_combinations}")
        
        # 結果を格納するリスト
        results = []
        successful_allocations = []
        
        # 各アクションセットに対してシミュレーション実行
        for i, action_set in enumerate(all_action_sets):
            # 進捗表示
            progress = (i + 1) / total_combinations * 100
            print(f"\r進捗: {progress:.1f}% ({i+1}/{total_combinations})", end="")
            
            # 環境の実行
            env_result, success = self._execute_environment(
                action_set, env_params, jobs_set, nb_jobs, visualize, debug
            )
            results.append(env_result)
            
            if success:
                successful_allocations.append(i)
            
            # 結果表示
            allocation_status = "✓" if success else "✗"
            print(f" -> アクションセット {action_set} -> コスト: {env_result['total_cost']:.2f}, 待ち時間: {env_result['total_waiting_time']:.2f} {allocation_status}")
        
        print("\n全探索完了!")
        
        # 結果の分析
        if len(successful_allocations) == 0:
            print("すべてのジョブを割り当てることができる組み合わせが見つかりませんでした。")
            if debug:
                print("問題の可能性がある点:")
                print("1. ジョブのサイズが大きすぎる")
                print("2. クラウド使用制約がある")
                print("3. ジョブのタイミングが重なりすぎている")
        else:
            print(f"全組み合わせ中、{len(successful_allocations)}/{total_combinations}の組み合わせですべてのジョブを割り当てることができました。")
            
            # 成功した割り当てのみでパレート最適解を計算
            successful_results = [results[i] for i in successful_allocations]
            pareto_indices = self._identify_pareto_solutions(successful_results)
            
            # 元のインデックスに変換
            original_pareto_indices = [successful_allocations[i] for i in pareto_indices]
            
            # 結果にパレート最適フラグを追加
            for i, result in enumerate(results):
                result['is_pareto_optimal'] = (i in original_pareto_indices)
                result['all_jobs_allocated'] = (i in successful_allocations)
            
            # パレート最適解の表示
            print("\nパレート最適解 (すべてのジョブを割り当て可能な中から):")
            for i in pareto_indices:
                result = successful_results[i]
                print(f"アクションセット: {result['action_set']}, コスト: {result['total_cost']:.2f}, 待ち時間: {result['total_waiting_time']:.2f}")
        
        # 結果を保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scheduling_results_{nb_jobs}jobs_{timestamp}.json"
        
        # すべての結果を保存
        self._save_results(results, filename)
        
        # 可視化グラフの作成
        if visualize:
            self._create_visualization(results, f"scheduling_viz_{nb_jobs}jobs_{timestamp}")
            
            # 成功した割り当てのみの可視化
            if successful_allocations:
                successful_results = [results[i] for i in successful_allocations]
                self._create_visualization(successful_results, f"successful_scheduling_{nb_jobs}jobs_{timestamp}")
        
        return os.path.join(self.output_dir, filename)
    
    def _execute_environment(self, action_set, env_params, jobs_set, nb_jobs, visualize=True, debug=False):
        """
        環境を実行してスケジューリング結果を取得
        
        引数:
            action_set (tuple): アクションの組み合わせ
            env_params (dict): 環境パラメータ
            jobs_set (list): ジョブセット
            nb_jobs (int): ジョブ数
            visualize (bool): マップ情報を記録するかどうか
            debug (bool): デバッグ情報を出力するかどうか
            
        戻り値:
            dict: スケジューリング結果
            bool: すべてのジョブを割り当てることができたかどうか
        """
        # 環境の初期化
        env = SchedulingEnv(
            nb_jobs, env_params['n_window'], env_params['n_on_premise_node'],
            env_params['n_cloud_node'], env_params['n_job_queue_obs'],
            env_params['n_job_queue_bck'], env_params['weight_wt'],
            env_params['weight_cost'], env_params['penalty_not_allocate'],
            env_params['penalty_invalid_action'], jobs_set=jobs_set,
            job_type=1, flag=1, next_init_windows=None, exe_mode=1
        )
        
        # 環境のリセット
        obs = env.reset()
        
        # 実行に必要な変数の初期化
        step = 0  # 現在のステップ数
        total_waiting_time = 0  # 総待ち時間
        total_cost = 0  # 総コスト
        processed_jobs = 0  # 処理済みジョブ数
        max_iterations = nb_jobs * 10  # 最大イテレーション数
        iteration_count = 0  # ループカウンタ
        
        allocation_success_log = []  # 各ジョブの割り当て成功ログ
        job_allocation_details = []  # 各ジョブの割り当て詳細
        
        # 可視化用のマップ情報
        map_snapshots = []
        if visualize:
            # 初期状態を記録
            map_snapshots.append({
                'step': 0,
                'on_premise': env.on_premise_window['status'].tolist(),
                'cloud': env.cloud_window['status'].tolist(),
                'job_queue': env.job_queue.tolist(),
                'description': '初期状態'
            })
        
        # メインループ
        while True:
            iteration_count += 1
            if iteration_count > max_iterations:
                if debug:
                    print(f"\n最大イテレーション数 {max_iterations} に達しました。")
                break
            
            # 現在のジョブの情報を取得
            if len(env.job_queue) > 0 and not np.all(env.job_queue[0] == 0):
                current_job = env.job_queue[0].copy()
                job_id = int(current_job[4]) if len(current_job) > 4 else -1
                
                if debug:
                    print(f"\n現在処理中のジョブ: ID={job_id}, サイズ=({current_job[0]}, {current_job[1]})")
                    print(f"現在のステップ: {step}/{len(action_set)}")
                    print(f"現在のジョブキュー: {env.job_queue}")
            else:
                # ジョブキューが空なら終了
                if debug:
                    print("\nジョブキューが空になりました。")
                break
            
            # アクションを設定
            if step < len(action_set):
                action_value = action_set[step]
                action = [0, action_value]  # メソッド=0、クラウド使用=action_value
                
                if debug:
                    print(f"実行するアクション: {action} (クラウド使用={action_value})")
            else:
                # アクションセットを使い果たした場合は終了
                if debug:
                    print("\nアクションセットを使い果たしました。")
                break
            
            # ステップ実行
            obs, rewards, info = env.step(action)
            
            # 割り当て結果の確認
            job_allocated = info.get('job_allocated', False)
            waiting_time = info.get('waiting_time', 0)
            cost = info.get('cost', 0)
            
            if job_allocated:
                step += 1
                processed_jobs += 1
                total_waiting_time += waiting_time
                total_cost += cost
                
                allocation_success_log.append(True)
                job_allocation_details.append({
                    'job_id': job_id,
                    'action': action_value,
                    'waiting_time': waiting_time,
                    'cost': cost,
                    'success': True
                })
                
                if debug:
                    print(f"ジョブ割り当て成功: ID={job_id}, 待ち時間={waiting_time}, コスト={cost}")
            else:
                allocation_success_log.append(False)
                job_allocation_details.append({
                    'job_id': job_id,
                    'action': action_value,
                    'waiting_time': 0,
                    'cost': 0,
                    'success': False,
                    'reason': info.get('fail_reason', '不明')
                })
                
                if debug:
                    print(f"ジョブ割り当て失敗: ID={job_id}, 理由={info.get('fail_reason', '不明')}")
            
            # 可視化用の状態を記録
            if visualize:
                map_snapshots.append({
                    'step': iteration_count,
                    'on_premise': env.on_premise_window['status'].tolist(),
                    'cloud': env.cloud_window['status'].tolist(),
                    'job_queue': env.job_queue.tolist(),
                    'action': action,
                    'job_allocated': job_allocated,
                    'description': f'ステップ {iteration_count}: ジョブID={job_id}, アクション={action}, 割り当て{"成功" if job_allocated else "失敗"}'
                })
            
            # すべてのジョブが処理されたか確認
            if env.check_is_done_for_exhaustive():
                if debug:
                    print("\nすべてのジョブが処理されました。")
                break
        
        # すべてのジョブが割り当てられたかどうか
        all_jobs_allocated = (processed_jobs == nb_jobs)
        
        # スケジューリング結果を作成
        result = {
            'action_set': action_set,
            'total_cost': float(total_cost),
            'total_waiting_time': float(total_waiting_time),
            'processed_jobs': int(processed_jobs),
            'total_jobs': int(nb_jobs),
            'all_jobs_allocated': all_jobs_allocated,
            'allocation_success_log': allocation_success_log,
            'job_allocation_details': job_allocation_details
        }
        
        # マップ情報がある場合は追加
        if visualize:
            result['map_snapshots'] = map_snapshots
        
        return result, all_jobs_allocated
    
    def _identify_pareto_solutions(self, results):
        """パレート最適解を特定"""
        pareto_indices = []
        
        for i, result1 in enumerate(results):
            is_dominated = False
            for result2 in results:
                # result2がresult1を支配しているかチェック
                if (result2['total_waiting_time'] <= result1['total_waiting_time'] and 
                    result2['total_cost'] <= result1['total_cost'] and 
                    (result2['total_waiting_time'] < result1['total_waiting_time'] or 
                     result2['total_cost'] < result1['total_cost'])):
                    is_dominated = True
                    break
            
            # 支配されていなければパレート最適
            if not is_dominated:
                pareto_indices.append(i)
                
        return pareto_indices
    
    def _save_results(self, results, filename):
        """結果をJSONファイルに保存"""
        json_path = os.path.join(self.output_dir, filename)
        
        data_object = {
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data_object, f, indent=2)
            
        print(f"結果を保存: {json_path}")
    
    def _create_visualization(self, results, filename):
        """スケジューリング結果の可視化グラフを作成"""
        # パレート最適解の特定
        pareto_indices = self._identify_pareto_solutions(results)
        
        # データの準備
        all_costs = [r['total_cost'] for r in results]
        all_waiting_times = [r['total_waiting_time'] for r in results]
        pareto_costs = [results[i]['total_cost'] for i in pareto_indices]
        pareto_waiting_times = [results[i]['total_waiting_time'] for i in pareto_indices]
        
        # プロット作成
        plt.figure(figsize=(10, 7))
        
        # すべての点をプロット
        plt.scatter(all_costs, all_waiting_times, 
                   color='lightgray', alpha=0.7, label='すべての組み合わせ')
        
        # パレート最適解を強調表示
        plt.scatter(pareto_costs, pareto_waiting_times, 
                   color='red', s=80, label='パレート最適解')
        
        # パレート最適解を線で接続
        pareto_points = sorted(zip(pareto_costs, pareto_waiting_times))
        if pareto_points:
            plt.plot([p[0] for p in pareto_points], 
                    [p[1] for p in pareto_points], 
                    'r--', alpha=0.7)
        
        # グラフの設定
        plt.title('スケジューリング結果とパレート最適解')
        plt.xlabel('総コスト')
        plt.ylabel('総待ち時間')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 保存
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}.png"), dpi=300)
        plt.close()

# メイン実行部分
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ジョブスケジューリング全探索ツール')
    parser.add_argument('--jobs', type=int, default=3, 
                        help='ジョブ数（デフォルト: 3、大きな値は計算時間が指数関数的に増加するため注意）')
    parser.add_argument('--no-vis', action='store_true', 
                        help='可視化を無効にする（デフォルト: False）')
    parser.add_argument('--debug', action='store_true', 
                        help='デバッグ情報を出力する（デフォルト: False）')
    args = parser.parse_args()
    
    explorer = SchedulingExplorer()
    result_file = explorer.run_exhaustive_search(args.jobs, not args.no_vis, args.debug)
    
    print(f"""
データ生成完了

1. ビューアでデータを確認するには、以下のファイルを参照してください:
   {result_file}

2. パレート最適解のグラフは以下のディレクトリにあります:
   {explorer.output_dir}
""") 