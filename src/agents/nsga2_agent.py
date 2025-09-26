import random
import numpy as np
import multiprocessing as mp
import os
import datetime
from typing import List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import traceback
from morl_baselines.common.performance_indicators import hypervolume
import time
import ray

def evaluate_individual_global(args):
    """並列処理用のグローバル評価関数"""
    chromosome, env_class, env_params = args
    
    env = env_class(**env_params)
    obs = env.reset()
    done = False
    step = 0
    total_reward = [0, 0]
    
    start_time = time.time()
    while not done:
        if step < len(chromosome):
            action = chromosome[step]
        else:
            action = 0
            
        obs, reward, scheduled, wt_step, done = env.step(action)
        if scheduled:
            step += 1
        if done:
            env.finalize_window_history()
        total_reward[0] += reward[0]
        total_reward[1] += reward[1]
        
    cost, _, makespan = env.calc_objective_values()
    evaluation_time = time.time() - start_time
    print(f"評価時間: {evaluation_time:.4f}秒")
    # 修正: 負の値を返さない（最小化が目的）
    return [cost, makespan]

def evaluate_individual_pure_step_time(env, chromosome):
    """環境インスタンスの作成・初期化を除外し、純粋にenv.step()の処理時間のみを計測"""
    import time
    
    # 環境をリセット（初期化時間は除外）
    obs = env.reset()
    done = False
    step = 0
    total_reward = [0, 0]
    nb_jobs = len(chromosome)
    processed_jobs = 0
    
    # 純粋なstep処理時間のみを計測
    step_start_time = time.time()
    
    action_index = 0
    while not done and processed_jobs < nb_jobs and action_index < len(chromosome):
        # アクションセットから次のアクションを取得
        action = chromosome[action_index]
        
        obs, reward, scheduled, wt_step, done = env.step(action)
        
        # スケジュールされた場合のみジョブカウントを増加
        if scheduled:
            processed_jobs += 1
        
        # 累積報酬を計算
        if isinstance(reward, (list, np.ndarray)) and len(reward) >= 2:
            total_reward[0] += reward[0]
            total_reward[1] += reward[1]
        else:
            # スカラー報酬の場合は適切に分配
            total_reward[0] += float(reward)
            total_reward[1] += 0
        
        # 次のアクションインデックスを増加
        action_index += 1
        
        if done:
            break
    
    # 環境の最終化
    if hasattr(env, 'finalize_window_history'):
        env.finalize_window_history()
    
    # NSGA-IIと同じ集計値を使用
    cost, _, avg_waiting_time = env.calc_objective_values()
    
    # 純粋な処理時間
    pure_step_time = time.time() - step_start_time
    # print(f"純粋なstep処理時間: {pure_step_time:.4f}秒")
    
    # 修正: 負の値を返さない（最小化が目的）
    return [cost, avg_waiting_time], pure_step_time

@dataclass
class Individual:
    """個体クラス"""
    chromosome: List[int]
    objectives: Optional[np.ndarray] = None
    rank: int = 0
    crowding_distance: float = 0.0
    
    def dominates(self, other: 'Individual') -> bool:
        """自分が他の個体を支配するかどうか判定"""
        # objectivesがNoneの場合は比較できない
        if self.objectives is None or other.objectives is None:
            return False
            
        better_in_any = False
        worse_in_any = False
        
        for i in range(len(self.objectives)):
            if self.objectives[i] < other.objectives[i]:
                better_in_any = True
            elif self.objectives[i] > other.objectives[i]:
                worse_in_any = True
                
        return better_in_any and not worse_in_any

class NSGA2Agent:
    """NSGA-IIを用いた多目的最適化によるスケジューリングエージェント"""
    
    def __init__(self, 
                 pop_size: int = 200, 
                 num_generations: int = 150,
                 crossover_prob: float = 0.9,
                 mutation_prob: float = 0.2,
                 tournament_size: int = 2,
                 eliminate_duplicates: bool = False
                 ):
        """
        Args:
            pop_size: 集団サイズ
            num_generations: 世代数
            crossover_prob: 交叉確率
            mutation_prob: 突然変異確率
            tournament_size: トーナメントサイズ
            eliminate_duplicates: 重複解の排除を有効にするかどうか
        """
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.eliminate_duplicates = eliminate_duplicates
        self.population = []
        self.history = {
            'pareto_fronts': [],
            'all_solutions': []
        }
        
        # 実行ディレクトリの作成
        self.execution_dir = self.create_execution_directory()
        
        # Rayの初期化確認
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
    
    def create_execution_directory(self):
        """実行ごとにディレクトリを作成"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"execution_{timestamp}"
        os.makedirs(dir_name, exist_ok=True)
        print(f"実行ディレクトリを作成しました: {dir_name}")
        return dir_name
    
    def save_solutions_to_file(self, generation: int, solutions: List[Individual]):
        """解のリストをテキストファイルに保存"""
        if not solutions:
            return
        
        # 世代を5回更新するごとに保存
        if generation % 5 == 0:
            filename = os.path.join(self.execution_dir, f"solutions_generation_{generation:03d}.txt")
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"世代 {generation} の解のリスト\n")
                f.write("=" * 50 + "\n")
                f.write("個体ID\tコスト\t待ち時間\t染色体\n")
                f.write("-" * 50 + "\n")
                
                for i, solution in enumerate(solutions):
                    if solution.objectives is not None:
                        # 修正: 負の値の変換を削除（既に正の値）
                        cost = solution.objectives[0]
                        makespan = solution.objectives[1]
                        
                        # 染色体を文字列として表示
                        chromosome_str = ''.join(map(str, solution.chromosome))
                        
                        f.write(f"{i+1:03d}\t{cost:.2f}\t{makespan:.2f}\t{chromosome_str}\n")
                
                f.write("-" * 50 + "\n")
                f.write(f"総個体数: {len(solutions)}\n")
                f.write(f"保存時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            print(f"世代 {generation} の解を保存しました: {filename}")

    def eliminate_duplicate_individuals(self, population: List[Individual]) -> List[Individual]:
        """重複する個体を排除する"""
        if not self.eliminate_duplicates:
            return population
        
        unique_individuals = []
        seen_chromosomes = set()
        
        for individual in population:
            # 染色体をタプルに変換してハッシュ可能にする
            chromosome_tuple = tuple(individual.chromosome)
            
            if chromosome_tuple not in seen_chromosomes:
                seen_chromosomes.add(chromosome_tuple)
                unique_individuals.append(individual)
            # else:
                # print(f"重複個体を排除: 染色体 ＿{chromosome_tuple[:20]}...")
        
        # removed_count = len(population) - len(unique_individuals)
        # if removed_count > 0:
            # print(f"重複排除: {removed_count}個の重複個体を排除しました")
        
        return unique_individuals

    def initialize_population(self, nb_jobs: int):
        """多様性を考慮した初期集団の生成"""
        self.population = []
        
        # 1. 基本的なランダム個体（30%）
        for _ in range(int(self.pop_size * 0.3)):
            chromosome = [random.randint(0, 1) for _ in range(nb_jobs)]
            self.population.append(Individual(chromosome))
        
        # 2. 重み付き個体（30%）
        weighted_individuals = self.create_weighted_individuals(nb_jobs, int(self.pop_size * 0.3))
        self.population.extend(weighted_individuals)
        
        # 3. 極端な個体（20%）
        for _ in range(int(self.pop_size * 0.1)):
            # すべてオンプレミス
            all_on_premise = [0] * nb_jobs
            self.population.append(Individual(all_on_premise))
            
            # すべてクラウド
            all_cloud = [1] * nb_jobs
            self.population.append(Individual(all_cloud))
        
        # 4. パターン個体（20%）
        pattern_individuals = self.create_pattern_individuals(nb_jobs, int(self.pop_size * 0.2))
        self.population.extend(pattern_individuals)
        
        # 重複排除を実行
        self.population = self.eliminate_duplicate_individuals(self.population)
        
        # 重複排除後、必要に応じて個体数を調整
        while len(self.population) < self.pop_size:
            chromosome = [random.randint(0, 1) for _ in range(nb_jobs)]
            new_individual = Individual(chromosome)
            if not any(tuple(new_individual.chromosome) == tuple(ind.chromosome) for ind in self.population):
                self.population.append(new_individual)

    def evaluate_individual(self, chromosome, env):
        """個体を評価する"""
        obs = env.reset()
        done = False
        step = 0
        total_reward = [0, 0]
        
        start_time = time.time()
        while not done:
            if step < len(chromosome):
                action = chromosome[step]
            else:
                action = 0
                
            obs, reward, scheduled, wt_step, done = env.step(action)
            if scheduled:
                step += 1
            if done:
                env.finalize_window_history()
            total_reward[0] += reward[0]
            total_reward[1] += reward[1]
            
        cost, _, makespan = env.calc_objective_values()
        evaluation_time = time.time() - start_time
        print(f"個体評価時間: {evaluation_time:.4f}秒")
        # 修正: 負の値を返さない（最小化が目的）
        return [cost, makespan]
            
    # 不要な関数をコメントアウト
    # def evaluate_population_pure_step_time(self, env, n_jobs=-1):
    #     """純粋なstep処理時間のみを計測する個体評価（環境作成・初期化時間除外）"""
    #     # この関数は使用されていないためコメントアウト

    def evaluate_population_ray(self, env, n_jobs=-1):
        """Rayを使用した個体評価（multiprocessing.Poolの代わり）"""
        individuals_to_evaluate = [
            ind for ind in self.population 
            if not hasattr(ind, 'objectives') or ind.objectives is None or any(obj == 0 for obj in ind.objectives)
        ]
        
        if not individuals_to_evaluate:
            return
        
        # 環境パラメータを準備
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
        
        # Rayを使用した並列処理
        @ray.remote
        def evaluate_individual_ray(chromosome, env_params):
            # 環境の作成と評価
            from src.envs.scheduling_env import SchedulingEnv
            env_copy = SchedulingEnv(**env_params)
            return evaluate_individual_pure_step_time(env_copy, chromosome)
        
        # 並列実行
        futures = [
            evaluate_individual_ray.remote(ind.chromosome, env_params) 
            for ind in individuals_to_evaluate
        ]
        
        # 結果の収集
        results = ray.get(futures)
        
        # 結果を個体に割り当て
        for i, result in enumerate(results):
            if isinstance(result, tuple) and len(result) == 2:
                individuals_to_evaluate[i].objectives = np.array(result[0])
            else:
                # エラーが発生した場合
                individuals_to_evaluate[i].objectives = np.array([1e6, 1e6])  # 大きな有限値
        
        # print(f"Ray並列処理で {len(individuals_to_evaluate)} 個体を評価完了")

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
                
        while True:
            current_members = [i for i, ind in enumerate(self.population) if ind.rank == current_rank]
            next_front = []
            
            for i in current_members:
                for j in dominated_sets[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        self.population[j].rank = current_rank + 1
                        next_front.append(j)
                        
            if not next_front:
                break
                
            current_rank += 1
            
    def calculate_crowding_distance(self):
        """混雑度の計算"""
        for individual in self.population:
            individual.crowding_distance = 0.0
            
        for obj_index in range(2):
            self.population.sort(key=lambda x: x.objectives[obj_index])
            
            self.population[0].crowding_distance = float('inf')
            self.population[-1].crowding_distance = float('inf')
            
            obj_range = self.population[-1].objectives[obj_index] - self.population[0].objectives[obj_index]
            if obj_range == 0:
                continue
                
            for i in range(1, len(self.population) - 1):
                distance = (self.population[i+1].objectives[obj_index] - self.population[i-1].objectives[obj_index]) / obj_range
                self.population[i].crowding_distance += distance
                
    def tournament_selection(self, selection_size):
        """トーナメント選択"""
        def crowded_comparison(a, b):
            if a.rank < b.rank:
                return a
            elif a.rank > b.rank:
                return b
            elif a.crowding_distance > b.crowding_distance:
                return a
            else:
                return b
        
        selected = []
        for _ in range(selection_size):
            candidates = random.sample(self.population, self.tournament_size)
            winner = candidates[0]
            for candidate in candidates[1:]:
                winner = crowded_comparison(winner, candidate)
            selected.append(winner)
            
        return selected
        
    def crossover(self, parent1: Individual, parent2: Individual):
        """多様な交叉戦略"""
        if random.random() > self.crossover_prob:
            return parent1, parent2
        
        crossover_type = random.random()
        
        if crossover_type < 0.4:
            # 一点交叉
            point = random.randint(1, len(parent1.chromosome) - 1)
            child1_chromosome = parent1.chromosome[:point] + parent2.chromosome[point:]
            child2_chromosome = parent2.chromosome[:point] + parent1.chromosome[point:]
        
        elif crossover_type < 0.7:
            # 二点交叉
            points = sorted(random.sample(range(1, len(parent1.chromosome)), 2))
            child1_chromosome = (parent1.chromosome[:points[0]] + 
                                parent2.chromosome[points[0]:points[1]] + 
                                parent1.chromosome[points[1]:])
            child2_chromosome = (parent2.chromosome[:points[0]] + 
                                parent1.chromosome[points[0]:points[1]] + 
                                parent2.chromosome[points[1]:])
        
        else:
            # 一様交叉
            mask = [random.randint(0, 1) for _ in range(len(parent1.chromosome))]
            child1_chromosome = [p1 if m else p2 for p1, p2, m in zip(parent1.chromosome, parent2.chromosome, mask)]
            child2_chromosome = [p2 if m else p1 for p1, p2, m in zip(parent1.chromosome, parent2.chromosome, mask)]
        
        child1 = Individual(child1_chromosome)
        child2 = Individual(child2_chromosome)
        
        return child1, child2
        
    def mutation(self, individual: Individual):
        """強化された突然変異戦略"""
        # 基本突然変異
        for i in range(len(individual.chromosome)):
            if random.random() < self.mutation_prob:
                individual.chromosome[i] = 1 - individual.chromosome[i]
        
        # 追加の突然変異戦略
        # 1. ビット反転（確率0.1）
        if random.random() < 0.1:
            flip_count = random.randint(1, 3)  # 1-3ビットを反転
            flip_indices = random.sample(range(len(individual.chromosome)), flip_count)
            for idx in flip_indices:
                individual.chromosome[idx] = 1 - individual.chromosome[idx]
        
        # 2. 部分的な再生成（確率0.05）
        if random.random() < 0.05:
            start = random.randint(0, len(individual.chromosome) - 8)
            end = min(start + 8, len(individual.chromosome))
            for i in range(start, end):
                individual.chromosome[i] = random.randint(0, 1)
                
    def create_weighted_individuals(self, nb_jobs: int, num_individuals: int = 20):
        """重み付き目的関数による多様な個体生成"""
        weighted_individuals = []
        
        # 異なる重みの組み合わせで個体を生成
        for i in range(num_individuals):
            # コストと待ち時間の重みをランダムに設定
            cost_weight = random.uniform(0.1, 0.9)
            makespan_weight = 1.0 - cost_weight
            
            # 重みに基づいて染色体を生成
            chromosome = []
            for j in range(nb_jobs):
                # 重みに基づいて確率的に0または1を選択
                if random.random() < cost_weight:
                    # コスト重視: より多くのジョブをオンプレミスに配置
                    chromosome.append(0)
                else:
                    # 待ち時間重視: より多くのジョブをクラウドに配置
                    chromosome.append(1)
            
            weighted_individuals.append(Individual(chromosome))
        
        return weighted_individuals

    def create_pattern_individuals(self, nb_jobs: int, num_individuals: int):
        """パターンに基づく個体生成"""
        patterns = []
        
        # 交互パターン
        alternating = [i % 2 for i in range(nb_jobs)]
        patterns.append(Individual(alternating))
        
        # ブロックパターン
        block_size = max(1, nb_jobs // 4)
        for i in range(num_individuals - 1):
            pattern = []
            for j in range(nb_jobs):
                block_idx = j // block_size
                pattern.append(block_idx % 2)
            patterns.append(Individual(pattern))
        
        return patterns

    def create_offspring(self):
        """子孫集団の生成（重み付き個体を含む）"""
        offspring = []
        parents = self.tournament_selection(self.pop_size)
        
        # 通常の交叉・突然変異による子孫生成
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child1, child2 = self.crossover(parents[i], parents[i+1])
                self.mutation(child1)
                self.mutation(child2)
                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(parents[i])
        
        # 重み付き個体を追加（多様性向上）
        if len(offspring) < self.pop_size:
            nb_jobs = len(self.population[0].chromosome) if self.population else 32
            additional_individuals = min(10, self.pop_size - len(offspring))
            weighted_individuals = self.create_weighted_individuals(nb_jobs, additional_individuals)
            offspring.extend(weighted_individuals)
        
        return offspring
        
    def save_pareto_front(self, generation: int):
        """現在のパレートフロントを保存"""
        pareto_front = [ind for ind in self.population if ind.rank == 1]
        
        if not pareto_front:
            if self.history['pareto_fronts']:
                self.history['pareto_fronts'].append(self.history['pareto_fronts'][-1])
            else:
                self.history['pareto_fronts'].append(np.array([[0.0, 0.0]]))
            return
        
        objectives = np.array([ind.objectives for ind in pareto_front])
        valid_indices = ~np.isnan(objectives).any(axis=1) & ~np.isinf(objectives).any(axis=1)
        objectives = objectives[valid_indices]
        
        if len(objectives) == 0:
            if self.history['pareto_fronts']:
                self.history['pareto_fronts'].append(self.history['pareto_fronts'][-1])
            else:
                self.history['pareto_fronts'].append(np.array([[0.0, 0.0]]))
            return
        
        # 修正: 評価済みの個体のみを対象とする
        evaluated_individuals = [ind for ind in self.population if ind.objectives is not None]
        if evaluated_individuals:
            all_solutions = np.array([ind.objectives for ind in evaluated_individuals])
        else:
            all_solutions = np.array([[0.0, 0.0]])
        
        self.history['pareto_fronts'].append(objectives)
        self.history['all_solutions'].append(all_solutions)
        
    def run(self, env, nb_jobs: int, verbose: bool = True, n_jobs=-1):
        """改善されたNSGA-IIによる最適化を実行"""
        import time
        
        # 初期集団の生成
        self.initialize_population(nb_jobs)
        
        # 初期集団の評価
        self.evaluate_population_ray(env, n_jobs)
        
        # 評価済み個体のみで非支配ソートと混雑度計算
        evaluated_population = [ind for ind in self.population if ind.objectives is not None]
        if not evaluated_population:
            print("警告: 評価済みの個体がありません")
            return self.get_final_pareto_front()
        
        self.population = evaluated_population
        self.non_dominated_sort()
        self.calculate_crowding_distance()
        
        # 初期パレートフロントの保存
        self.save_pareto_front(0)
        
        # 初期世代の解を保存
        self.save_solutions_to_file(0, self.population)
        
        if verbose:
            print(f"世代 0 - パレートフロントサイズ: {len([ind for ind in self.population if ind.rank == 1])}")
        
        # 世代を進める
        for generation in range(1, self.num_generations + 1):
            # 現在の最良個体を保存
            current_best = self.population[:max(1, self.pop_size // 10)]
            
            # 子孫集団の生成
            offspring = self.create_offspring()
            
            # 子孫集団の評価
            self.population = offspring
            self.evaluate_population_ray(env, n_jobs)

            # 評価済み個体のみを対象とする
            evaluated_offspring = [ind for ind in offspring if ind.objectives is not None]
            
            # 親と子の集団を結合
            self.population = evaluated_offspring + current_best
            
            # 重複排除を実行
            self.population = self.eliminate_duplicate_individuals(self.population)
            
            # 非支配ソートと混雑度計算
            self.non_dominated_sort()
            self.calculate_crowding_distance()
            
            # ランクと混雑度でソート
            self.population.sort(key=lambda x: (x.rank, -x.crowding_distance))
            
            # 上位pop_size個体を選択（エリート保存）
            self.population = self.population[:self.pop_size]
            
            # 多様性維持のためのランダム個体注入
            if generation % 20 == 0:
                self.inject_diversity(nb_jobs)
                # 注入後の重複排除
                self.population = self.eliminate_duplicate_individuals(self.population)
            
            # パレートフロントの保存
            self.save_pareto_front(generation)
            
            # 世代を5回更新するごとに解のリストを保存
            self.save_solutions_to_file(generation, self.population)
            
            if verbose and (generation % 10 == 0 or generation == self.num_generations):
                pareto_front = [ind for ind in self.population if ind.rank == 1]
                print(f"世代 {generation} - パレートフロントサイズ: {len(pareto_front)}")
                if pareto_front:
                    print(f"  目的関数値の例: {pareto_front[0].objectives}")
        
        # 最終世代の解も保存
        self.save_solutions_to_file(self.num_generations, self.population)
        
        # 実行完了のサマリーファイルを作成
        self.create_execution_summary()

        return self.get_final_pareto_front()
    
    def create_execution_summary(self):
        """実行完了時のサマリーファイルを作成"""
        summary_file = os.path.join(self.execution_dir, "execution_summary.txt")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("NSGA-II実行サマリー\n")
            f.write("=" * 50 + "\n")
            f.write(f"実行ディレクトリ: {self.execution_dir}\n")
            f.write(f"集団サイズ: {self.pop_size}\n")
            f.write(f"世代数: {self.num_generations}\n")
            f.write(f"交叉確率: {self.crossover_prob}\n")
            f.write(f"突然変異確率: {self.mutation_prob}\n")
            f.write(f"トーナメントサイズ: {self.tournament_size}\n")
            f.write(f"重複排除: {'有効' if self.eliminate_duplicates else '無効'}\n")
            f.write(f"実行開始時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"保存された解ファイル数: {len([f for f in os.listdir(self.execution_dir) if f.startswith('solutions_generation_')])}\n")
        
        print(f"実行サマリーを作成しました: {summary_file}")

    def get_final_pareto_front(self):
        """最終的なパレートフロントの取得"""
        pareto_front = [ind for ind in self.population if ind.rank == 1]
        objectives = np.array([ind.objectives for ind in pareto_front])
        chromosomes = [ind.chromosome for ind in pareto_front]
        
        return {
            'objectives': objectives,
            'chromosomes': chromosomes
        }
    
    def calc_hypervolume(self):
        """ハイパーボリュームの計算"""
        objectives = np.array([ind.objectives for ind in self.population])
        return hypervolume([200000,20],objectives)

    # 不要な関数をコメントアウト
    # def visualize_nsga2_results(result):
    #     """NSGA-IIの結果を可視化"""
    #     # この関数はmain.pyで定義されているためコメントアウト

    def inject_diversity(self, nb_jobs: int):
        """多様性維持のためのランダム個体注入"""
        # 最下位の10%をランダム個体で置き換え
        replace_count = max(1, self.pop_size // 10)
        
        for i in range(replace_count):
            if len(self.population) > replace_count:
                # ランダム個体を生成
                new_chromosome = [random.randint(0, 1) for _ in range(nb_jobs)]
                new_individual = Individual(new_chromosome)
                
                # 最下位の個体を置き換え
                self.population[-(i+1)] = new_individual
