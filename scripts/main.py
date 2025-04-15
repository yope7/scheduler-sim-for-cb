import numpy as np
import yaml
import sys
import argparse
import itertools
import matplotlib.pyplot as plt
from morl_baselines.common.pareto import get_non_dominated_inds
from src.agents.pcn_agent import PCN  
from src.envs.scheduling_env import SchedulingEnv
from src.utils.job_gen.job_generator import JobGenerator
from src.utils.map_visualizer import visualize_map
from src.agents.dqn_agent import DQNAgent
from numba import jit
from src.agents.all_agent import ExhaustiveSearchAgent
np.set_printoptions(linewidth=np.inf) 

with open('config/config.yml', 'r') as yml:
    config = yaml.safe_load(yml)
# 環境のパラメータをUUU
max_step = np.inf
n_window = config['param_env']['n_window']
n_on_premise_node = config['param_env']['n_on_premise_node']
n_cloud_node = config['param_env']['n_cloud_node']
n_job_queue_obs = config['param_env']['n_job_queue_obs']
n_job_queue_bck = config['param_env']['n_job_queue_bck']
penalty_not_allocate = config['param_env']['penalty_not_allocate']  # 割り当てない(一時キューに格納する)という行動を選択した際のペナルティー
penalty_invalid_action = config['param_env']['penalty_invalid_action']  # actionが無効だった場合のペナルティー
n_action = 2
weight_wt = config['param_agent']['weight_wt']
weight_cost = config['param_agent']['weight_cost']
nb_steps = config['param_simulation']['nb_steps']
nb_episodes = config['param_simulation']['nb_episodes']
nb_max_episode_steps = config['param_simulation']['nb_max_episode_steps'] # 1エピソードあたりの最大ステップ数(-1:最大ステップ無し)
if nb_max_episode_steps == -1:
    nb_max_episode_steps = np.inf
multi_algorithm = config['param_simulation']['multi_algorithm'] # 0:single algorithm(後でジョブを決める) 1:multi algorithm

# デバッグフラグ
DEBUG_MODE = False
# DEBUG_MODE = True

def run_debug_episode(env):
    """デバッグ用に1エピソードを実行し、スケジューリング結果を表示"""
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # ランダムな行動を選択
        obs, rewards, is_fifo, wt_step, std_mean_after, done, _ = env.step(action)

    env.render_map()
    # env.show_final_window_history()
    sys.exit()  # コメントアウト

def evaluate_and_render(agent, env, objective_index):
    """指定されたob番号で評価とレンダリングを行う"""
    print(f"training finished, start evaluation {objective_index}...")

    _, values = agent.evaluate_and_execute_selected_policy(
        env, 
        max_return=np.full(2, 100.0, dtype=np.float32), 
        objective_index=objective_index, 
        n=10
    )
    # print(next_init_windows)
    print(f"evaluation finished, start rendering {objective_index}...")

    agent.env.render_map(f"ob{objective_index}")

    return values


    # Define environment

def set_and_train(nb_steps, lams, loops, how_many_episodes,ob_number,nb_jobs):
    next_init_windows = None
    values_all = []
    job_generator = JobGenerator(0,nb_steps, n_window, n_on_premise_node, n_cloud_node, config,nb_jobs,0.3,how_many_episodes)
    jobs_set = job_generator.generate_jobs_set()

    env = SchedulingEnv(
        max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck,
        weight_wt, weight_cost, penalty_not_allocate, penalty_invalid_action, jobs_set,
        next_init_windows,flag=0
    )

    # デバッグモードで1エピソードを実行
    # if DEBUG_MODE:
    #     run_debug_episode(env)
    # else:

    # print("train_and_execute")

    """NNの初期化→学習→割り当て"""
    agent = PCN(
        env,
        device="auto",
        state_dim=1,
        scaling_factor=np.array([1, 1, 1]),
        learning_rate=1e-2,
        batch_size=4096,
        hidden_dim=256,
        project_name="temp",
        experiment_name="PCN",
        log=True,
    )
    agent.train(
        eval_env=env,
        total_timesteps=int(how_many_episodes),
        ref_point=np.array([-1000,-1000]),
        num_er_episodes=10,
        num_step_episodes=50,  
        num_model_updates=2,
        max_buffer_size=5000,
        known_pareto_front=[1, 1],
    )

    # e_returns = agent.get_e_returns()
    # transitions = agent.get_transitions()
    mapmap = agent.get_mapmap()
    # print("mapmap: ",mapmap)

    import matplotlib.pyplot as plt

    def visualize_map(mapmap):
        # データの取得（最初の配列を使用）
        map_data = mapmap[0]
        
        # プロットの設定
        plt.figure(figsize=(12, 8))
        
        # ヒートマップの描画
        plt.imshow(map_data, cmap='tab20', interpolation='nearest')
        
        # カラーバーの追加
        plt.colorbar(label='Region ID')
        
        # タイトルと軸ラベルの設定
        plt.title('マップの可視化')
        plt.xlabel('X座標')
        plt.ylabel('Y座標')
        
        # グリッドの表示
        plt.grid(True, which='minor', color='black', linestyle='-', alpha=0.2)
        
        # 表示
        plt.show()

    return 0

    # print("values: ",values)
    #0 = waiting time , 1 = cost
    for i in range(loops):
        job_generator = JobGenerator(i,nb_steps, n_window, n_on_premise_node, n_cloud_node, config, nb_jobs,lams[i])
        jobs_set = job_generator.generate_jobs_set()

        next_init_windows = env.get_next_init_windows()
        env = SchedulingEnv(
            max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck,
            weight_wt, weight_cost, penalty_not_allocate, penalty_invalid_action, jobs_set,
            next_init_windows,flag=1,exe_mode=0
        )
        values = train_and_execute(env, how_many_episodes, ob_number)
        print("values: ",i,values)
        values_all.append(values)
        print("values_all: ",values_all)
    return values_all

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='single', 
                      choices=['single', 'pareto','pcn','all'], 
                      help='実行モード（single: 単一重み, pareto: パレートフロント探索, pcn: PCN, all: 全探索）')
    # 既存の引数
    parser.add_argument('--how_many_episodes', type=int, default=1000)
    parser.add_argument('--nb_jobs', type=int, default=11)
    return parser.parse_args()

def run_single_rl_mode(nb_steps: int, lams: list, loops: int, how_many_episodes: int, 
                       ob_number: int, nb_jobs: int) -> list:
    next_init_windows = None
    values_all = []
    np.random.seed(0)
    job_generator = JobGenerator(0, nb_steps, n_window, n_on_premise_node, 
                               n_cloud_node, config, nb_jobs, 0.2, how_many_episodes)
    jobs_set = job_generator.generate_jobs_set()

    env = SchedulingEnv(
        max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck,
        weight_wt, weight_cost, penalty_not_allocate, penalty_invalid_action, jobs_set,
        next_init_windows, flag=0
    )

    # 観測空間のサイズを取得
    obs_space_size = (12790)

    agent = DQNAgent(
        env,
        device="auto",
        state_dim=obs_space_size,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        hidden_dim=256,
        target_update=10,
        weight_wt=1.0,    # 待ち時間を重視
        weight_cost=0.0   # コストの重みを小さく
    )

    losses = agent.train(
        how_many_episodes,
        early_stop_threshold=0.01,
        patience=5,
        min_episodes=50
    )
    
    # 損失履歴をプロット
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss History')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.close()
    
    return 0

def run_PCN_mode(nb_steps: int, lams: list, loops: int, how_many_episodes: int, 
                ob_number: int, nb_jobs: int) -> list:
    """PCN強化学習モードの実行"""
    next_init_windows = None
    values_all = []
    np.random.seed(0)
    job_generator = JobGenerator(0, nb_steps, n_window, n_on_premise_node, 
                               n_cloud_node, config, nb_jobs, 0.2, how_many_episodes)
    jobs_set = job_generator.generate_jobs_set()

    # print("jobs_set: ",jobs_set)

    env = SchedulingEnv(
        max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck,
        weight_wt, weight_cost, penalty_not_allocate, penalty_invalid_action, jobs_set,
        next_init_windows, flag=0
    )

    agent = PCN(
        env,
        device="auto",
        state_dim=1,
        scaling_factor=np.array([1, 1, 1]),
        learning_rate=1e-2,
        batch_size=4096,
        hidden_dim=256,
        project_name="temp",
        experiment_name="PCN",
        log=True,
    )
    
    agent.train(
        eval_env=env,
        total_timesteps=int(how_many_episodes),
        ref_point=np.array([-1000,-1000]),
        num_er_episodes=30000,
        num_step_episodes=50,  
        num_model_updates=10,
        max_buffer_size=5000,
        known_pareto_front=[1, 1],
    )

    on_premise_map, cloud_map = env.get_windows()
    print(env.calc_objective_values())

    return agent.get_mapmap()


def run_exhaustive_mode(nb_jobs: int):
    """全探索モードの実行"""
    # 環境パラメータの設定
    next_init_windows = None
    np.random.seed(0)
    job_generator = JobGenerator(0, nb_steps, n_window, n_on_premise_node, 
                               n_cloud_node, config, nb_jobs, 0.2, 0)

    jobs_set = job_generator.generate_jobs_set()

    # 環境の初期化
    env = SchedulingEnv(
        max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck,
        weight_wt, weight_cost, penalty_not_allocate, penalty_invalid_action, jobs_set,
        next_init_windows, flag=1
    )
    
    # エージェントの初期化と実行
    agent = ExhaustiveSearchAgent()
    search_results = agent.run_exhaustive_search(env, nb_jobs)
    
    print("results: ", search_results['results'])
    print("pareto_front: ", search_results['pareto_front'])
    
    return search_results['results']

def run_pareto_search(nb_steps: int, how_many_episodes: int, nb_jobs: int, weight_steps: int = 10):
    pareto_points = []
    weights = np.linspace(0, 1, weight_steps)
    
    for w_wt in weights:
        w_cost = 1 - w_wt
        print(f"\nTraining with weights - Waiting Time: {w_wt:.2f}, Cost: {w_cost:.2f}")
        
        # 環境の初期化
        np.random.seed(0)
        job_generator = JobGenerator(0, nb_steps, n_window, n_on_premise_node, 
                                   n_cloud_node, config, nb_jobs, 0.2, how_many_episodes)
        jobs_set = job_generator.generate_jobs_set()


        
        env = SchedulingEnv(
            max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck,
            w_wt, w_cost, penalty_not_allocate, penalty_invalid_action, jobs_set,
            None, flag=0
        )
        
        # エージェントの初期化と学習
        agent = DQNAgent(
            env,
            device="auto",
            state_dim=(12790),
            learning_rate=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=64,
            hidden_dim=256,
            target_update=10,
            weight_wt=w_wt,
            weight_cost=w_cost
        )
        
        agent.train(
            how_many_episodes,
            early_stop_threshold=0.01,
            patience=5,
            min_episodes=50
        )
        
        # 最終的な報酬を記録
        waiting_time, cost = agent.get_rewards()
        pareto_points.append((waiting_time, cost))
        
    # パレートフロントのプロット
    wt_rewards, cost_rewards = zip(*pareto_points)
    plt.figure(figsize=(10, 8))
    plt.scatter(wt_rewards, cost_rewards, c='blue', label='Solutions')
    plt.xlabel('Waiting Time')
    plt.ylabel('Cost')
    plt.title('Pareto Front Approximation')
    plt.grid(True)
    plt.savefig('pareto_front.png')
    plt.close()
    
    return pareto_points

if __name__ == "__main__":

    # コマンドライン引数の解析
    args = parse_args()

    if args.mode == 'pcn':
        # 強化学習モードのパラメータ設定と実行
        loops = 0
        lams = [0.2] * loops
        how_many_episodes = 10000
        ob_number = 1
        nb_jobs = 11
        mapmap = run_PCN_mode(nb_steps, lams, loops, how_many_episodes, ob_number, nb_jobs)
        print("PCN強化学習による実行が完了しました")

    if args.mode == 'single':
        loops = 0
        lams = [0.2] * loops
        how_many_episodes = 50000
        ob_number = 1
        nb_jobs = 11
        mapmap = run_single_rl_mode(nb_steps, lams, loops, how_many_episodes, ob_number, nb_jobs)
        print("単目的強化学習による実行が完了しました")

    elif args.mode == 'pareto':     

        how_many_episodes = 10000
        pareto_points = run_pareto_search(
            nb_steps,
            how_many_episodes,
            nb_jobs=args.nb_jobs,
            weight_steps=1  # 重みの分割数
        )
        print("\nPareto Front Points:")
        for wt, cost in pareto_points:
            print(f"Waiting Time: {wt:.2f}, Cost: {cost:.2f}")

    elif args.mode == 'all':
        # 全探索モードの実行
        results = run_exhaustive_mode(args.nb_jobs)
        print("全探索による実行が完了しました")




    # env2 = SimpleEnv()
    # agent2 = PCN(
    #     env2,
    #     device = "cpu",
    #     scaling_factor=np.array([1, 1,1]),
    #     learning_rate=1e-3,
    #     batch_size=256,
    #     project_name="MORL-Baselines",
    #     experiment_name="PCN",
    #     log=True,
    # )
    # agent2.train(
    # eval_env=env2,
    # total_timesteps=int(1e6),
    # ref_point=np.array([0, 0]),
    # num_er_episodes=20,
    # max_buffer_size=50,
    # num_model_updates=50,
    # # max_return=np.array([1.5, 1.5, -0.0]),
    # known_pareto_front=[1,1],
    # )



