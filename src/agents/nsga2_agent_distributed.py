import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import os
import datetime
from morl_baselines.common.pareto import get_non_dominated_inds, get_non_dominated_inds_minimize
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
        self.objectives = [0.0, 0.0]  # 目的関数値 [コスト, 平均待ち時間]
        self.cumulative_reward = [0.0, 0.0]  # 累積報酬 [報酬1, 報酬2]
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

@ray.remote
class NSGA2Worker:
    """Rayワーカーとして動作するNSGA-IIのサブ集団"""
    
    def __init__(self, pop_size: int, num_generations: int, crossover_prob: float, mutation_prob: float):
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.population = []
        
    def initialize_population(self, nb_jobs: int, worker_id: int = 0):
        """初期集団の生成（改善された染色体表現）"""
        # ワーカー固有のランダムシードを設定
        random.seed(hash(f"worker_{worker_id}_{random.random()}") % (2**32))
        
        self.population = []
        for i in range(self.pop_size):
            # 改善された染色体表現: [job_priority, cloud_usage]
            # job_priority: ジョブの処理優先度 (0-99の整数)
            # cloud_usage: クラウド使用確率 (0-100の整数、パーセンテージ)
            
            # ワーカーごとに異なる戦略で初期化
            if worker_id % 4 == 0:
                # 低コスト戦略：クラウド使用を控えめに
                cloud_bias = random.randint(10, 40)
            elif worker_id % 4 == 1:
                # バランス戦略：適度にクラウドを使用
                cloud_bias = random.randint(30, 70)
            elif worker_id % 4 == 2:
                # 高速戦略：積極的にクラウドを使用
                cloud_bias = random.randint(60, 90)
            else:
                # ランダム戦略：完全にランダム
                cloud_bias = random.randint(0, 100)
            
            # 染色体の生成：各ジョブに対して [優先度, クラウド使用確率]
            chromosome = []
            for job_idx in range(nb_jobs):
                priority = random.randint(0, 99)  # ジョブ優先度
                cloud_usage = min(100, max(0, cloud_bias + random.randint(-20, 20)))  # クラウド使用確率
                chromosome.extend([priority, cloud_usage])
            
            self.population.append(Individual(chromosome))
            
    def evaluate_population(self, env_params: Dict):
        """個体評価を実行（改善された評価方法）"""
        # 環境クラスを動的にインポート
        import sys
        sys.path.append('src')
        from envs.scheduling_env import SchedulingEnv
        
        for ind in self.population:
            if not hasattr(ind, 'objectives') or any(obj == 0 for obj in ind.objectives):
                try:
                    # 環境のコピーを作成
                    env_copy = SchedulingEnv(**env_params)
                    
                    # 改善された評価方法：染色体からスケジューリング戦略を構築
                    nb_jobs = len(ind.chromosome) // 2  # 各ジョブに2つの遺伝子
                    
                    # ジョブの優先度順序を決定
                    job_priorities = []
                    for i in range(nb_jobs):
                        priority = ind.chromosome[i * 2]  # 優先度
                        cloud_prob = ind.chromosome[i * 2 + 1]  # クラウド使用確率
                        job_priorities.append((i, priority, cloud_prob))
                    
                    # 優先度でソート（高い値ほど優先）
                    job_priorities.sort(key=lambda x: x[1], reverse=True)
                    
                    obs = env_copy.reset()
                    done = False
                    processed_jobs = 0
                    total_reward = [0, 0]
                    
                    while not done and processed_jobs < nb_jobs:
                        # 現在の優先度順でジョブを処理
                        if processed_jobs < len(job_priorities):
                            job_idx, priority, cloud_prob = job_priorities[processed_jobs]
                            
                            # クラウド使用確率に基づいてアクションを決定
                            # より柔軟な決定：環境の状態も考慮
                            if hasattr(env_copy, 'job_queue') and len(env_copy.job_queue) > 0:
                                # 現在のキューの状況を考慮
                                queue_length = np.count_nonzero(env_copy.job_queue[0])
                                if queue_length > 3:  # キューが混雑している場合
                                    cloud_prob += 20  # クラウド使用確率を上げる
                                
                            # 確率的にアクションを決定
                            action = 1 if random.randint(0, 100) < cloud_prob else 0
                        else:
                            action = 0
                            
                        obs, reward, scheduled, wt_step, done = env_copy.step(action)
                        if scheduled:
                            processed_jobs += 1
                        if done:
                            env_copy.finalize_window_history()
                        
                        # 累積報酬を正しく計算
                        total_reward[0] += reward[0]
                        total_reward[1] += reward[1]
                    
                    # 目的関数値の計算（0番目と2番目を最小化）
                    cost, _, avg_waiting_time = env_copy.calc_objective_values()
                    # 0番目と2番目を最小化するため、NSGAには負の値を渡す
                    ind.objectives = [-cost, -avg_waiting_time]
                    # 累積報酬も保存
                    ind.cumulative_reward = total_reward
                    
                except Exception as e:
                    print(f"個体評価でエラーが発生: {e}")
                    # エラーが発生した場合はデフォルト値を設定
                    ind.objectives = [float('inf'), float('inf')]
                    ind.cumulative_reward = [0, 0]
    
    def non_dominated_sort(self):
        """非支配ソーティング"""
        domination_counts = [0] * len(self.population)
        dominated_sets = [[] for _ in range(len(self.population))]
        
        for individual in self.population:
            individual.rank = 0
            
        current_rank = 1
        for i in range(len(self.population)):
            for j in range(len(self.population)):
                if i != j:
                    if self.population[i].dominates(self.population[j]):
                        dominated_sets[i].append(j)
                    elif self.population[j].dominates(self.population[i]):
                        domination_counts[i] += 1
            
            if domination_counts[i] == 0:
                self.population[i].rank = current_rank
                
        # 残りのランクを処理
        while True:
            next_rank = []
            for i in range(len(self.population)):
                if self.population[i].rank == current_rank:
                    for j in dominated_sets[i]:
                        domination_counts[j] -= 1
                        if domination_counts[j] == 0:
                            next_rank.append(j)
            
            if not next_rank:
                break
                
            current_rank += 1
            for j in next_rank:
                self.population[j].rank = current_rank
    
    def calculate_crowding_distance(self):
        """混雑度の計算"""
        for individual in self.population:
            individual.crowding_distance = 0.0
            
        num_objectives = len(self.population[0].objectives)
        
        for obj_idx in range(num_objectives):
            # 目的関数でソート
            self.population.sort(key=lambda x: x.objectives[obj_idx])
            
            # 境界の個体は無限大の混雑度
            self.population[0].crowding_distance = float('inf')
            self.population[-1].crowding_distance = float('inf')
            
            # 中間の個体の混雑度を計算
            obj_range = self.population[-1].objectives[obj_idx] - self.population[0].objectives[obj_idx]
            if obj_range > 0:
                for i in range(1, len(self.population) - 1):
                    self.population[i].crowding_distance += (
                        self.population[i + 1].objectives[obj_idx] - 
                        self.population[i - 1].objectives[obj_idx]
                    ) / obj_range
    
    def tournament_selection(self):
        """トーナメント選択"""
        def crowded_comparison(a, b):
            if a.rank < b.rank:
                return a
            elif a.rank > b.rank:
                return b
            else:
                return a if a.crowding_distance > b.crowding_distance else b
        
        parents = []
        for _ in range(self.pop_size):
            # 2個体をランダムに選択
            idx1, idx2 = random.sample(range(len(self.population)), 2)
            winner = crowded_comparison(self.population[idx1], self.population[idx2])
            parents.append(winner)
        return parents
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交叉操作（改善された染色体表現対応）"""
        if random.random() < self.crossover_prob:
            nb_jobs = len(parent1.chromosome) // 2
            
            # 複数点交叉：ジョブ単位で交叉
            child1_chrom = []
            child2_chrom = []
            
            for i in range(nb_jobs):
                if random.random() < 0.5:  # 50%の確率で親1から継承
                    child1_chrom.extend(parent1.chromosome[i*2:(i+1)*2])
                    child2_chrom.extend(parent2.chromosome[i*2:(i+1)*2])
                else:  # 50%の確率で親2から継承
                    child1_chrom.extend(parent2.chromosome[i*2:(i+1)*2])
                    child2_chrom.extend(parent1.chromosome[i*2:(i+1)*2])
            
            return Individual(child1_chrom), Individual(child2_chrom)
        else:
            return Individual(parent1.chromosome.copy()), Individual(parent2.chromosome.copy())
    
    def mutation(self, individual: Individual):
        """突然変異操作（改善された染色体表現対応）"""
        nb_jobs = len(individual.chromosome) // 2
        
        for i in range(nb_jobs):
            if random.random() < self.mutation_prob:
                # 優先度の突然変異（±10の範囲でランダム変更）
                priority_idx = i * 2
                current_priority = individual.chromosome[priority_idx]
                new_priority = max(0, min(99, current_priority + random.randint(-10, 10)))
                individual.chromosome[priority_idx] = new_priority
                
            if random.random() < self.mutation_prob:
                # クラウド使用確率の突然変異（±15%の範囲でランダム変更）
                cloud_idx = i * 2 + 1
                current_cloud = individual.chromosome[cloud_idx]
                new_cloud = max(0, min(100, current_cloud + random.randint(-15, 15)))
                individual.chromosome[cloud_idx] = new_cloud
    
    def create_offspring(self):
        """子孫集団の生成"""
        offspring = []
        parents = self.tournament_selection()
        
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child1, child2 = self.crossover(parents[i], parents[i+1])
                self.mutation(child1)
                self.mutation(child2)
                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(parents[i])
        
        return offspring
    
    def run_generation(self, env_params: Dict):
        """1世代の進化を実行"""
        # 個体評価
        self.evaluate_population(env_params)
        
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
    
    def replace_worst_solution(self, new_solution: Individual):
        """最悪の解を新しい解で置き換え（改善された多様性考慮版）"""
        if len(self.population) > 0:
            # 新しい染色体表現に対応した類似度計算
            nb_jobs = len(new_solution.chromosome) // 2
            min_distance = float('inf')
            most_similar_idx = -1
            
            for i, ind in enumerate(self.population):
                # 戦略の類似度を計算（優先度パターンとクラウド使用パターン）
                priority_distance = 0
                cloud_distance = 0
                
                for j in range(nb_jobs):
                    # 優先度の差
                    priority_diff = abs(new_solution.chromosome[j*2] - ind.chromosome[j*2])
                    priority_distance += priority_diff / 99  # 正規化
                    
                    # クラウド使用確率の差
                    cloud_diff = abs(new_solution.chromosome[j*2+1] - ind.chromosome[j*2+1])
                    cloud_distance += cloud_diff / 100  # 正規化
                
                # 総合距離（戦略の類似度）
                total_distance = (priority_distance + cloud_distance) / (2 * nb_jobs)
                
                if total_distance < min_distance:
                    min_distance = total_distance
                    most_similar_idx = i
            
            # 類似解がある場合は置き換え、ない場合は最悪解を置き換え
            if min_distance < 0.2:  # 戦略が20%以上異なる場合のみ追加
                # 最悪の解（ランクが高く、混雑度が低い）を削除
                self.population.sort(key=lambda x: (x.rank, -x.crowding_distance))
                self.population.pop()  # 最悪の解を削除
                self.population.append(new_solution)  # 新しい解を追加

class DistributedNSGA2Agent:
    """Rayを使用した分散化NSGA-IIエージェント"""
    
    def __init__(self, 
                 num_workers: int = 32,
                 pop_size_per_worker: int = 50,
                 num_generations: int = 100,
                 migration_interval: int = 15,
                 crossover_prob: float = 0.8,
                 mutation_prob: float = 0.15
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
        
        # 各ワーカーの初期集団を生成（ワーカーIDを渡して多様性を確保）
        ray.get([
            worker.initialize_population.remote(nb_jobs, i)
            for i, worker in enumerate(self.workers)
        ])
        
        # 世代を進める
        for generation in range(self.num_generations):
            # 各ワーカーで評価と進化を実行
            populations = ray.get([
                worker.run_generation.remote(env_params)
                for worker in self.workers
            ])
            
            # 解の交換（マイグレーション）- 早期は頻度を下げる
            migration_freq = self.migration_interval
            if generation < self.num_generations // 3:
                migration_freq = self.migration_interval * 2  # 初期は頻度を下げる
            
            if generation % migration_freq == 0 and generation > 0:
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
        """解の交換（マイグレーション）を実行（多様性保持改善版）"""
        # 各ワーカーから多様な解を選択
        diverse_solutions = []
        for pop_idx, pop in enumerate(populations):
            # ランク1の解を選択
            rank1_solutions = [ind for ind in pop if ind.rank == 1]
            if rank1_solutions:
                # 混雑度でソートして多様な解を選択
                rank1_solutions.sort(key=lambda x: -x.crowding_distance)
                # 上位1個のみを選択（多様性を保つため）
                diverse_solutions.append(rank1_solutions[0])
        
        # 解をランダムに再配布（ring topology）
        if diverse_solutions and len(diverse_solutions) >= 2:
            random.shuffle(diverse_solutions)  # ランダム化
            for i, worker in enumerate(self.workers):
                if i < len(diverse_solutions):
                    # 隣接ワーカーからの解のみを受け取る（ring topology）
                    source_idx = (i + 1) % len(diverse_solutions)
                    if source_idx != i:  # 自分自身からは受け取らない
                        ray.get(worker.replace_worst_solution.remote(diverse_solutions[source_idx]))
    
    def _save_pareto_front(self, populations: List[List[Individual]], generation: int):
        """現在のパレートフロントを保存"""
        # 全ワーカーの解を統合
        all_individuals = []
        for pop in populations:
            all_individuals.extend(pop)
        
        # ランク1の個体を抽出
        pareto_front = [ind for ind in all_individuals if ind.rank == 1]
        
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
        all_solutions = np.array([ind.objectives for ind in all_individuals])
        
        self.history['pareto_fronts'].append(objectives)
        self.history['all_solutions'].append(all_solutions)
        
        # デバッグ出力
        if generation % 10 == 0 or generation == 0:
            print(f"世代 {generation} のパレートフロント: {objectives.shape}")
    
    def _get_global_pareto_front(self, populations: List[List[Individual]]) -> List[Individual]:
        """全ワーカーの解からグローバルパレートフロントを取得"""
        all_individuals = []
        for pop in populations:
            all_individuals.extend(pop)
        
        # 非支配ソートを実行
        domination_counts = [0] * len(all_individuals)
        for i in range(len(all_individuals)):
            for j in range(len(all_individuals)):
                if i != j and all_individuals[i].dominates(all_individuals[j]):
                    domination_counts[i] += 1
        
        # ランク1の個体を返す
        return [ind for i, ind in enumerate(all_individuals) if domination_counts[i] == 0]
    
    def _get_final_pareto_front(self) -> Dict:
        """最終パレートフロントを取得"""
        # 最後の世代のパレートフロントを取得
        if self.history['pareto_fronts']:
            objectives = self.history['pareto_fronts'][-1]
        else:
            objectives = np.array([[0.0, 0.0]])
        
        return {
            'objectives': objectives,
            'chromosomes': []  # 分散処理版では染色体は保持していない
        }
    
    def visualize_progress(self, save_dir: str = "nsga2_distributed_results"):
        """最適化の進捗を可視化する（DistributedPCNと完全に同じ形式）"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if len(self.history['pareto_fronts']) == 0:
            print("警告: パレートフロントの履歴が空です")
            return
        
        # 1. 報酬空間でのパレートフロント（最大化目的）
        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.history['pareto_fronts'])))
        
        # 全解をプロット（累積報酬を使用）
        all_solutions = self.history['all_solutions'][-1] if self.history['all_solutions'] else np.array([[0.0, 0.0]])
        plt.scatter(all_solutions[:, 0], all_solutions[:, 1], c='lightblue', alpha=0.6, label='All Solutions', s=50)
        
        # 世代ごとのパレートフロントをプロット
        for i, front in enumerate(self.history['pareto_fronts']):
            if i % max(1, len(self.history['pareto_fronts']) // 10) == 0:
                plt.scatter(front[:, 0], front[:, 1], c=[colors[i]], label=f"Gen {i}", alpha=0.7, s=80)
        
        # 最終世代のパレートフロントを強調表示
        final_front = self.history['pareto_fronts'][-1]
        plt.scatter(
            final_front[:, 0], final_front[:, 1],
            c=[colors[-1]], label=f"Gen {len(self.history['pareto_fronts'])-1}",
            alpha=0.8, s=100, edgecolor='black'
        )
        
        # パレートフロントの線を描画
        if len(final_front) > 1:
            sorted_indices = np.lexsort((final_front[:, 1], final_front[:, 0]))
            sorted_pareto = final_front[sorted_indices]
            plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'r-', linewidth=2, alpha=0.8)
        
        plt.title(f"Evolution of Pareto Front Using Distributed NSGA-II (Objective Space)\nNon-dominated: {len(final_front)}", fontsize=12)
        plt.xlabel("Cost (Minimize)", fontsize=10)
        plt.ylabel("Avg Waiting Time (Minimize)", fontsize=10)
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/pareto_evolution_objectives_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 実数値空間でのパレートフロント（最小化目的）
        plt.figure(figsize=(10, 8))
        
        # 全解をプロット
        plt.scatter(all_solutions[:, 0], all_solutions[:, 1], c='lightgreen', alpha=0.6, label='All Solutions', s=50)
        
        # 世代ごとのパレートフロントをプロット
        for i, front in enumerate(self.history['pareto_fronts']):
            if i % max(1, len(self.history['pareto_fronts']) // 10) == 0:
                plt.scatter(front[:, 0], front[:, 1], c=[colors[i]], label=f"Gen {i}", alpha=0.7, s=80)
        
        # 最終世代のパレートフロントを強調表示
        plt.scatter(
            final_front[:, 0], final_front[:, 1],
            c=[colors[-1]], label=f"Gen {len(self.history['pareto_fronts'])-1}",
            alpha=0.8, s=100, edgecolor='black'
        )
        
        # パレートフロントの線を描画
        if len(final_front) > 1:
            sorted_indices = np.lexsort((final_front[:, 1], final_front[:, 0]))
            sorted_pareto = final_front[sorted_indices]
            plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'r-', linewidth=2, alpha=0.8)
        
        plt.title(f"Evolution of Pareto Front Using Distributed NSGA-II (Objective Space)\nNon-dominated: {len(final_front)}", fontsize=12)
        plt.xlabel("Cost (Minimize)", fontsize=10)
        plt.ylabel("Avg Waiting Time (Minimize)", fontsize=10)
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/pareto_evolution_objectives_detailed_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 詳細データの保存（DistributedPCNと同じ形式）
        details_path = f"{save_dir}/pareto_front_details_{timestamp}.txt"
        with open(details_path, 'w', encoding='utf-8') as f:
            f.write(f"=== 分散処理版NSGA-II パレートフロント詳細 ===\n")
            f.write(f"生成日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"総世代数: {len(self.history['pareto_fronts'])}\n")
            f.write(f"最終世代のパレートフロントサイズ: {len(final_front)}\n")
            f.write(f"最終世代の全解数: {len(all_solutions)}\n")
            
            # 軸範囲の計算（DistributedPCNと同じ方法）
            if len(all_solutions) > 0:
                x_min, x_max = all_solutions[:, 0].min(), all_solutions[:, 0].max()
                y_min, y_max = all_solutions[:, 1].min(), all_solutions[:, 1].max()
                x_margin = (x_max - x_min) * 0.1
                y_margin = (y_max - y_min) * 0.1
                f.write(f"実数値空間軸範囲: X[{x_min-x_margin:.4f}, {x_max+x_margin:.4f}], Y[{y_min-y_margin:.4f}, {y_max+y_margin:.4f}]\n")
            
            # 最終世代の非支配解を詳細に記録
            f.write(f"\n=== 最終世代の非支配解 ===\n")
            for i, obj in enumerate(final_front):
                f.write(f"解{i+1}: コスト={obj[0]:.4f}, 平均待ち時間={obj[1]:.4f}\n")
        
        print(f"可視化結果を保存しました: {save_dir}")

def visualize_nsga2_results(result):
    """NSGA-IIの結果を可視化する関数（DistributedPCNと完全に同じ形式）"""
    plt.figure(figsize=(12, 10))
    
    # パレートフロントのプロット
    objectives = result['objectives']
    
    # 全解をプロット
    plt.scatter(objectives[:, 0], objectives[:, 1], c='lightgreen', alpha=0.6, label='All Solutions', s=50)
    
    # 非支配解を強調表示
    plt.scatter(objectives[:, 0], objectives[:, 1], c='red', s=100, label='Pareto Front', alpha=0.7, edgecolor='black')
    
    # パレートフロントの線を描画
    if len(objectives) > 1:
        sorted_indices = np.lexsort((objectives[:, 1], objectives[:, 0]))
        sorted_pareto = objectives[sorted_indices]
        plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'r-', linewidth=2, alpha=0.8)
    
    # 各解をラベル付け
    for i, obj in enumerate(objectives):
        plt.annotate(
            f"{i+1}", 
            (obj[0], obj[1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    plt.title(f'分散処理版NSGA-IIによる最適化結果のパレートフロント\nNon-dominated: {len(objectives)}', fontsize=16)
    plt.xlabel('コスト（最小化）', fontsize=14)
    plt.ylabel('平均待ち時間（最小化）', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存
    plt.tight_layout()
    plt.savefig('nsga2_distributed_pareto_front.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 詳細結果の表示と保存
    print("\n分散処理版NSGA-II パレートフロント:")
    print("----------------------------------")
    print("No. | コスト  | 平均待ち時間")
    print("----------------------------------")
    
    for i, obj in enumerate(objectives):
        print(f"{i+1:2d} | {-obj[0]:7.2f} | {-obj[1]:7.2f}")
