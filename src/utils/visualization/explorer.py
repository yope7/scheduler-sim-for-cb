import itertools
import wandb
import yaml
from src.utils.job_gen.job_generator import JobGenerator
from multiEnv2 import SchedulingEnv
import numpy as np

#printでnpのはいれつを出力した際に改行しない
np.set_printoptions(linewidth=np.inf)

def exhaustive_assignment_evaluation(nb_jobs):
    """
    固定ジョブセットに対して、各ジョブごとの割り当て（0 または 1）の全通りを生成し、
    各シーケンスで環境を実行、エピソード終了時の総 waiting time と総 cost を評価して wandb にログ出力する関数。

    PCNAgent の evaluate_and_execute_selected_policy の遷移再現部分を参考にしています。
    """
    wandb.init(project="ExhaustiveAssignmentEvaluation", name="full_exploration")
    
    with open("config.yml", "r") as yml:
        config = yaml.safe_load(yml)
        
    n_window = config["param_env"]["n_window"]
    n_on_premise_node = config["param_env"]["n_on_premise_node"]
    n_cloud_node = config["param_env"]["n_cloud_node"]
    n_job_queue_obs = config["param_env"]["n_job_queue_obs"]
    n_job_queue_bck = config["param_env"]["n_job_queue_bck"]
    weight_wt = config["param_agent"]["weight_wt"]
    weight_cost = config["param_agent"]["weight_cost"]
    penalty_not_allocate = config["param_env"]["penalty_not_allocate"]
    penalty_invalid_action = config["param_env"]["penalty_invalid_action"]
    nb_steps = config["param_simulation"]["nb_steps"]
    nb_episodes = 0

    # 固定ジョブセットの生成
    lam = 0.3
    job_generator = JobGenerator(1, nb_steps, n_window, n_on_premise_node, n_cloud_node, config, nb_jobs, lam, nb_episodes)
    jobs_set = job_generator.generate_jobs_set()
    print("jobs_set:", jobs_set)

    max_step = nb_jobs

    # 各ジョブに対するアクション（0 または 1）の全組み合わせを生成
    all_action_sets = list(itertools.product([0, 1], repeat=nb_jobs))
    print("Total action sets:", len(all_action_sets))
    results = []

    for action_set in all_action_sets:
        # 環境リセット
        env = SchedulingEnv(
            max_step,
            n_window,
            n_on_premise_node,
            n_cloud_node,
            n_job_queue_obs,
            n_job_queue_bck,
            weight_wt,
            weight_cost,
            penalty_not_allocate,
            penalty_invalid_action,
            jobs_set=jobs_set,
            job_type=1,
            flag=1,
            next_init_windows=None,
            exe_mode=1
        )
        obs = env.reset()
        done = False
        step = 0
        wt_step_sum = 0

        # 無限ループ防止用：最大イテレーション回数（適宜調整）
        iteration_count = 0
        max_iterations = nb_jobs * 10

        while not done and step < nb_jobs and iteration_count < max_iterations:
            iteration_count += 1
            action = action_set[step]
            print(f"Step: {step}, Action: {action}")

            # 例: デバッグ用のログを追加する
            print(f"【Debug】step: {step}, time: {env.time}, job_queue head: {env.job_queue[0]}")
            obs, rewards, scheduled, wt_step, _, done, _ = env.step(action, exe_mode=0)
            print(f"【Debug】scheduled: {scheduled}, wt_step: {wt_step}, done: {done}")
            
            if scheduled:
                step += 1
                wt_step_sum += wt_step
            
            # done フラグを確認
            if done:
                print("done フラグが True になりました")
                break

        # 万が一、イテレーション上限に達してしまった場合の注意
        if iteration_count >= max_iterations:
            print("警告: max_iterations に到達しました。ループ内で何か問題がないか確認してください")

        total_cost, makespan = env.calc_objective_values()
        results.append([total_cost, wt_step_sum])
        wandb.log({
            "total_cost": total_cost,
            "total_wt": wt_step_sum
        })
        print(f"アクションセット {action_set} -> コスト: {total_cost}, 待ち時間: {wt_step_sum}")
        input()

    return results

def new_func(seq, env):
    env.render_map(str(seq))

if __name__ == "__main__":
    nb_jobs = 15
     # 例としてジョブ数4の場合（注意：ジョブ数が大きいと組み合わせ数が急増します）
    obj = exhaustive_assignment_evaluation(nb_jobs)
    print(obj)

    #objは[total_cost, total_wt]のリストのリスト
    #二次元スキャッタープロットを表示
    import matplotlib.pyplot as plt

    total_costs = [result[0] for result in obj]
    total_wts = [result[1] for result in obj]

    plt.scatter(total_costs, total_wts)
    plt.xlabel("Total Cost")
    plt.ylabel("Total Waiting Time")
    plt.show()

