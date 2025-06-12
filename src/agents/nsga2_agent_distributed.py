import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import os
import datetime
from morl_baselines.common.pareto import get_non_dominated_inds
import multiprocessing as mp
from joblib import Parallel, delayed
import ray

class Individual:
    """NSGA-IIで扱う個体クラス"""
    
    def __init__(self, chromosome: List[int]):
        """
        Args:
            chromosome: 0または1の行動リスト (各ジョブに対するクラウド利用の有無)
        """
        self.chromosome = chromosome  # 染色体（アクションの配列）
        self.objectives = [0.0, 0.0]  # 目的関数値 [コスト, 待ち時間]
        self.rank = 0                # 非支配ランク
        self.crowding_distance = 0.0  # 混雑度
        
    def dominates(self, other: 'Individual') -> bool:
        """自分が他の個体を支配するかどうか判定"""
        better_in_any = False
        worse_in_any = False
        
        # 両方の目的関数を小さくする問題と仮定
        for i in range(len(self.objectives)):
            if self.objectives[i] < other.objectives[i]:
                better_in_any = True
            elif self.objectives[i] > other.objectives[i]:
                worse_in_any = True
                
        # 少なくとも1つの目的関数で優れており、他のどの目的関数でも劣っていない場合、支配する
        return better_in_any and not worse_in_any

class NSGA2Worker:
    """Rayワーカーとして動作するNSGA-IIのサブ集団"""
    
    def __init__(self, pop_size: int, num_generations: int, crossover_prob: float, mutation_prob: float):
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.population = []
        
    def initialize_population(self, nb_jobs: int):
        """初期集団の生成"""
        self.population = []
        for _ in range(self.pop_size):
            chromosome = [random.randint(0, 1) for _ in range(nb_jobs)]
            self.population.append(Individual(chromosome))
            
    def evaluate_population(self, env_params: Dict):
        """個体評価を実行"""
        for ind in self.population:
            if not hasattr(ind, 'objectives') or any(obj == 0 for obj in ind.objectives):
                # 環境のコピーを作成
                env_copy = type(env)(**env_params)
                
                obs = env_copy.reset()
                done = False
                step = 0
                total_reward = [0, 0]
                
                while not done:
                    if step < len(ind.chromosome):
                        action = ind.chromosome[step]
                    else:
                        action = 0
                        
                    obs, reward, scheduled, wt_step, done = env_copy.step(action)
                    if scheduled:
                        step += 1
                    if done:
                        env_copy.finalize_window_history()
                    total_reward[0] += reward[0]
                    total_reward[1] += reward[1]
                    
                cost, makespan = env_copy.calc_objective_values()
                ind.objectives = [cost, makespan]
                
    def run_generation(self):
        """1世代の進化を実行"""
        # 非支配ソートと混雑度計算
        self.non_dominated_sort()
        self.calculate_crowding_distance()
        
        # 子孫集団の生成
        offspring = self.create_offspring()
        
        # 親と子の集団を結合
        self.population.extend(offspring)
        
        # 非支配ソートと混雑度計算
        self.non_dominated_sort()
        self.calculate_crowding_distance()
        
        # ランクと混雑度でソート
        self.population.sort(key=lambda x: (x.rank, -x.crowding_distance))
        
        # 上位pop_size個体を選択
        self.population = self.population[:self.pop_size]
        
        return self.population

class DistributedNSGA2Agent:
    """Rayを使用した分散化NSGA-IIエージェント"""
    
    def __init__(self, 
                 num_workers: int = 4,
                 pop_size_per_worker: int = 50,
                 num_generations: int = 100,
                 migration_interval: int = 10,
                 crossover_prob: float = 0.9,
                 mutation_prob: float = 0.6
                 ):
        """
        Args:
            num_workers: ワーカー数
            pop_size_per_worker: ワーカーあたりの集団サイズ
            num_generations: 世代数
            migration_interval: 解の交換間隔
            crossover_prob: 交叉確率
            mutation_prob: 突然変異確率
        """
        self.num_workers = num_workers
        self.pop_size_per_worker = pop_size_per_worker
        self.num_generations = num_generations
        self.migration_interval = migration_interval
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        # Rayの初期化
        if not ray.is_initialized():
            ray.init()
            
        # ワーカーの作成
        self.workers = [
            NSGA2Worker.remote(
                pop_size=pop_size_per_worker,
                num_generations=num_generations,
                crossover_prob=crossover_prob,
                mutation_prob=mutation_prob
            )
            for _ in range(num_workers)
        ]
        
        self.history = {
            'pareto_fronts': [],
            'all_solutions': [],
            'hypervolume': []
        }
        
    def run(self, env, nb_jobs: int, verbose: bool = True):
        """分散化NSGA-IIによる最適化を実行"""
        # 環境パラメータの準備
        env_params = {
            'max_step': env.max_step,
            'n_window': env.n_window,
            'n_on_premise_node': env.n_on_premise_node,
            'n_cloud_node': env.n_cloud_node,
            'n_job_queue_obs': env.n_job_queue_obs,
            'n_job_queue_bck': env.n_job_queue_bck,
            'weight_wt': env.weight_wt,
            'weight_cost': env.weight_cost,
            'penalty_not_allocate': env.penalty_not_allocate,
            'penalty_invalid_action': env.penalty_invalid_action,
            'jobs_set': env.jobs_set,
            'flag': 0
        }
        
        # 各ワーカーの初期集団を生成
        ray.get([
            worker.initialize_population.remote(nb_jobs)
            for worker in self.workers
        ])
        
        # 世代を進める
        for generation in range(self.num_generations):
            # 各ワーカーで評価と進化を実行
            populations = ray.get([
                worker.run_generation.remote()
                for worker in self.workers
            ])
            
            # 解の交換（マイグレーション）
            if generation % self.migration_interval == 0:
                self._migrate_solutions(populations)
            
            # パレートフロントの保存
            self._save_pareto_front(populations, generation)
            
            if verbose and (generation % 10 == 0 or generation == self.num_generations - 1):
                pareto_front = self._get_global_pareto_front(populations)
                print(f"世代 {generation} - パレートフロントサイズ: {len(pareto_front)}")
                
        # 最終パレートフロントの可視化
        self.visualize_progress()
        
        return self._get_final_pareto_front()
        
    def _migrate_solutions(self, populations: List[List[Individual]]):
        """解の交換（マイグレーション）を実行"""
        # 各ワーカーから最良の解を選択
        best_solutions = []
        for pop in populations:
            pareto_front = [ind for ind in pop if ind.rank == 1]
            if pareto_front:
                best_solutions.extend(pareto_front[:self.pop_size_per_worker // 10])
                
        # 解をシャッフルして再分配
        random.shuffle(best_solutions)
        solutions_per_worker = len(best_solutions) // self.num_workers
        
        for i, worker in enumerate(self.workers):
            start_idx = i * solutions_per_worker
            end_idx = start_idx + solutions_per_worker
            worker_solutions = best_solutions[start_idx:end_idx]
            
            # ワーカーの集団に解を追加
            ray.get(worker.add_solutions.remote(worker_solutions))
            
    def _save_pareto_front(self, populations: List[List[Individual]], generation: int):
        """現在のパレートフロントを保存"""
        # 全ワーカーの解を結合
        all_solutions = []
        for pop in populations:
            all_solutions.extend(pop)
            
        # 非支配ソートを実行
        pareto_front = get_non_dominated_inds(all_solutions)
        
        if not pareto_front:
            print(f"警告: 世代 {generation} でパレートフロントが空です")
            if self.history['pareto_fronts']:
                self.history['pareto_fronts'].append(self.history['pareto_fronts'][-1])
            else:
                self.history['pareto_fronts'].append(np.array([[0.0, 0.0]]))
            return
            
        objectives = np.array([ind.objectives for ind in pareto_front])
        
        # 目的関数値が無効な場合はフィルタリング
        valid_indices = ~np.isnan(objectives).any(axis=1) & ~np.isinf(objectives).any(axis=1)
        objectives = objectives[valid_indices]
        
        if len(objectives) == 0:
            print(f"警告: 世代 {generation} でパレートフロントの目的関数値が全て無効です")
            if self.history['pareto_fronts']:
                self.history['pareto_fronts'].append(self.history['pareto_fronts'][-1])
            else:
                self.history['pareto_fronts'].append(np.array([[0.0, 0.0]]))
            return
            
        # 全解も記録
        all_solutions = np.array([ind.objectives for ind in all_solutions])
        
        self.history['pareto_fronts'].append(objectives)
        self.history['all_solutions'].append(all_solutions)
        
    def _get_global_pareto_front(self, populations: List[List[Individual]]) -> List[Individual]:
        """全ワーカーの解からグローバルなパレートフロントを取得"""
        all_solutions = []
        for pop in populations:
            all_solutions.extend(pop)
            
        return get_non_dominated_inds(all_solutions)
        
    def _get_final_pareto_front(self) -> Dict:
        """最終的なパレートフロントの取得"""
        # 全ワーカーの解を取得
        populations = ray.get([
            worker.get_population.remote()
            for worker in self.workers
        ])
        
        # グローバルなパレートフロントを取得
        pareto_front = self._get_global_pareto_front(populations)
        
        objectives = np.array([ind.objectives for ind in pareto_front])
        chromosomes = [ind.chromosome for ind in pareto_front]
        
        return {
            'objectives': objectives,
            'chromosomes': chromosomes
        }

def visualize_nsga2_results(result):
    """NSGA-IIの結果を可視化する関数"""
    plt.figure(figsize=(12, 10))
    
    # パレートフロントのプロット
    objectives = result['objectives']
    plt.scatter(objectives[:, 1], objectives[:, 0], c='blue', s=100, label='NSGA-II Pareto Front', alpha=0.7)
    
    # 各解をラベル付け
    for i, obj in enumerate(objectives):
        plt.annotate(
            f"{i+1}", 
            (obj[1], obj[0]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    plt.title('NSGA-IIによる最適化結果のパレートフロント', fontsize=16)
    plt.xlabel('Waiting Time (Makespan)', fontsize=14)
    plt.ylabel('Cost', fontsize=14)
    plt.grid(True)
    plt.legend()
    
    # 保存
    plt.tight_layout()
    plt.savefig('nsga2_pareto_front.png')
    plt.close()
    
    # 詳細結果の表示と保存
    print("\nNSGA-II パレートフロント:")
    print("----------------------------------")
    print("No. | コスト  | 待ち時間")
    print("----------------------------------")
    
    for i, obj in enumerate(objectives):
        print(f"{i+1:2d} | {obj[0]:7.2f} | {obj[1]:7.2f}")
