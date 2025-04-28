import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import os
import datetime
from morl_baselines.common.pareto import get_non_dominated_inds
import multiprocessing as mp
from joblib import Parallel, delayed

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

class NSGA2Agent:
    """NSGA-IIを用いた多目的最適化によるスケジューリングエージェント"""
    
    def __init__(self, 
                 pop_size: int = 50, 
                 num_generations: int = 100,
                 crossover_prob: float = 0.9,
                 mutation_prob: float = 0.3
                 ):
        """
        Args:
            pop_size: 集団サイズ
            num_generations: 世代数
            crossover_prob: 交叉確率
            mutation_prob: 突然変異確率
        """
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.population = []          # 現在の集団
        self.offspring = []           # 子孫集団
        self.history = {              # 最適化履歴
            'pareto_fronts': [],
            'all_solutions': [],
            'hypervolume': []  # ハイパーボリュームを追加
        }
        
    def initialize_population(self, nb_jobs: int):
        """初期集団の生成"""
        self.population = []
        for _ in range(self.pop_size):
            # 各ジョブにランダムに0または1を割り当てる
            chromosome = [random.randint(0, 1) for _ in range(nb_jobs)]
            self.population.append(Individual(chromosome))
            
    def evaluate_population_parallel(self, env, n_jobs=-1):
        """個体評価を並列化して実行"""
        # 未評価の個体のみを対象にする
        individuals_to_evaluate = [
            ind for ind in self.population 
            if not hasattr(ind, 'objectives') or any(obj == 0 for obj in ind.objectives)
        ]
        
        if not individuals_to_evaluate:
            return
        
        # 利用可能なCPUコア数を取得
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        
        # 環境をシリアライズ可能な形で準備
        # 注: 環境によってはこの方法が使えない場合がある
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
        
        # 1個体の評価を行う関数
        def evaluate_individual(chromosome):
            # 環境のコピーを作成
            env_copy = type(env)(**env_params)
            
            obs = env_copy.reset()
            done = False
            step = 0
            total_reward = [0, 0]
            
            while not done:
                if step < len(chromosome):
                    action = chromosome[step]
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
            return [cost, makespan]
        
        # 並列評価の実行
        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_individual)(ind.chromosome) 
            for ind in individuals_to_evaluate
        )
        
        # 結果を各個体に設定
        for ind, result in zip(individuals_to_evaluate, results):
            ind.objectives = result
    
    def non_dominated_sort(self):
        """非支配ソーティング"""
        # 各個体の支配関係と支配されている個体数をカウント
        domination_counts = [0] * len(self.population)
        dominated_sets = [[] for _ in range(len(self.population))]
        
        # 各個体のランクを初期化
        for individual in self.population:
            individual.rank = 0
            
        # 第一フロント（ランク1）を探索
        current_rank = 1
        for i in range(len(self.population)):
            for j in range(len(self.population)):
                if i != j:
                    if self.population[i].dominates(self.population[j]):
                        dominated_sets[i].append(j)
                    elif self.population[j].dominates(self.population[i]):
                        domination_counts[i] += 1
            
            # 支配されていない個体は第一フロント
            if domination_counts[i] == 0:
                self.population[i].rank = current_rank
                
        # 残りのフロントを探索
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
            
        # 各目的関数ごとに混雑度を計算
        for obj_index in range(2):
            # 目的関数値でソート
            self.population.sort(key=lambda x: x.objectives[obj_index])
            
            # 境界点は無限大に設定
            self.population[0].crowding_distance = float('inf')
            self.population[-1].crowding_distance = float('inf')
            
            # 中間点の混雑度を計算
            obj_range = self.population[-1].objectives[obj_index] - self.population[0].objectives[obj_index]
            if obj_range == 0:
                continue
                
            for i in range(1, len(self.population) - 1):
                distance = (self.population[i+1].objectives[obj_index] - self.population[i-1].objectives[obj_index]) / obj_range
                self.population[i].crowding_distance += distance
                
    def tournament_selection(self):
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
        for _ in range(self.pop_size):
            # 2個体を選択
            candidates = random.sample(self.population, 2)
            winner = crowded_comparison(candidates[0], candidates[1])
            selected.append(winner)
            
        return selected
        
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """一点交叉"""
        if random.random() > self.crossover_prob:
            return parent1, parent2
            
        point = random.randint(1, len(parent1.chromosome) - 1)
        child1_chromosome = parent1.chromosome[:point] + parent2.chromosome[point:]
        child2_chromosome = parent2.chromosome[:point] + parent1.chromosome[point:]
        
        child1 = Individual(child1_chromosome)
        child2 = Individual(child2_chromosome)
        
        return child1, child2
        
    def mutation(self, individual: Individual):
        """突然変異"""
        for i in range(len(individual.chromosome)):
            if random.random() < self.mutation_prob:
                individual.chromosome[i] = 1 - individual.chromosome[i]  # 0と1を反転
                
    def create_offspring(self):
        """子孫集団の生成"""
        self.offspring = []
        parents = self.tournament_selection()
        
        # 交配して子孫を生成
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child1, child2 = self.crossover(parents[i], parents[i+1])
                self.mutation(child1)
                self.mutation(child2)
                self.offspring.append(child1)
                self.offspring.append(child2)
            else:
                # 奇数の場合は最後の親をそのまま追加
                self.offspring.append(parents[i])
                
    def save_pareto_front(self, generation: int):
        """現在のパレートフロントを保存"""
        # ランク1の個体を抽出
        pareto_front = [ind for ind in self.population if ind.rank == 1]
        
        if not pareto_front:
            print(f"警告: 世代 {generation} でパレートフロントが空です")
            # 空のパレートフロントの場合、前の世代のデータを使用
            if self.history['pareto_fronts']:
                self.history['pareto_fronts'].append(self.history['pareto_fronts'][-1])
            else:
                # 初期状態で空の場合、ダミーデータを追加
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
        all_solutions = np.array([ind.objectives for ind in self.population])
        
        self.history['pareto_fronts'].append(objectives)
        self.history['all_solutions'].append(all_solutions)
        
        # デバッグ出力
        if generation % 10 == 0 or generation == 0:
            print(f"世代 {generation} のパレートフロント: {objectives.shape}")
        
    def visualize_progress(self, save_dir: str = "nsga2_results"):
        """最適化の進捗を可視化する（改良版）"""
        import matplotlib.pyplot as plt
        
        os.makedirs(save_dir, exist_ok=True)
        
        # タイムスタンプ
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. パレートフロントの進化をプロット
        plt.figure(figsize=(10, 8))
        
        # カラーマップの設定
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.history['pareto_fronts'])))
        
        for i, front in enumerate(self.history['pareto_fronts']):
            if i % max(1, len(self.history['pareto_fronts']) // 10) == 0:  # 表示数を制限
                plt.scatter(front[:, 1], front[:, 0], c=[colors[i]], label=f"Gen {i}", alpha=0.7)
        
        # 最終世代は必ず表示
        if len(self.history['pareto_fronts']) > 0:
            plt.scatter(
                self.history['pareto_fronts'][-1][:, 1],
                self.history['pareto_fronts'][-1][:, 0],
                c=[colors[-1]], label=f"Gen {len(self.history['pareto_fronts'])-1}",
                alpha=0.7, s=100, edgecolor='black'
            )
        
        plt.title("Evolution of Pareto Front Using NSGA-II")
        plt.xlabel("Waiting Time (Makespan)")
        plt.ylabel("Cost")
        plt.legend(loc='upper right')
        plt.grid(True)
        
        plt.savefig(f"{save_dir}/pareto_evolution_{timestamp}.png")
        plt.close()
        
        # 2. ハイパーボリュームの推移をプロット
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.history['hypervolume'])), self.history['hypervolume'], 'b-', linewidth=2)
        plt.title("Hypervolume Progression")
        plt.xlabel("Generation")
        plt.ylabel("Hypervolume")
        plt.grid(True)
        
        # 再スタート世代をマーク
        restart_gens = []
        for i in range(1, len(self.history['hypervolume'])):
            if i > self.stagnation_threshold and self.history['hypervolume'][i] < self.history['hypervolume'][i-1]:
                restart_gens.append(i)
        
        for gen in restart_gens:
            plt.axvline(x=gen, color='r', linestyle='--', alpha=0.5)
        
        plt.savefig(f"{save_dir}/hypervolume_progress_{timestamp}.png")
        plt.close()
        
        # アニメーションGIFの作成
        try:
            import matplotlib.animation as animation
            import matplotlib.pyplot as plt
            
            # 図のセットアップ
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 全世代のデータ範囲を取得
            all_pareto_points = []
            for front in self.history['pareto_fronts']:
                if len(front) > 0:
                    all_pareto_points.extend(front)
            
            all_pareto_points = np.array(all_pareto_points)
            
            if len(all_pareto_points) > 0:
                x_min, x_max = np.min(all_pareto_points[:, 1]), np.max(all_pareto_points[:, 1])
                y_min, y_max = np.min(all_pareto_points[:, 0]), np.max(all_pareto_points[:, 0])
                
                # マージンを追加
                x_margin = (x_max - x_min) * 0.1 if x_max > x_min else 0.1
                y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
                
                # 軸の範囲を設定
                ax.set_xlim(x_min - x_margin, x_max + x_margin)
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
            else:
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            
            # 軸ラベルとタイトル
            ax.set_xlabel("Waiting Time (Makespan)")
            ax.set_ylabel("Cost")
            ax.set_title("Generation 0")
            ax.grid(True)
            
            # 初期の空のスキャッタープロット
            scat = ax.scatter([], [])
            
            # 更新関数を完全に書き直し
            def update(frame):
                ax.clear()
                ax.set_xlabel("Waiting Time (Makespan)")
                ax.set_ylabel("Cost")
                ax.grid(True)
                
                # 軸の範囲を一貫して保持
                if len(all_pareto_points) > 0:
                    ax.set_xlim(x_min - x_margin, x_max + x_margin)
                    ax.set_ylim(y_min - y_margin, y_max + y_margin)
                
                # 現在のフレームのパレートフロントを取得
                if frame < len(self.history['pareto_fronts']) and len(self.history['pareto_fronts'][frame]) > 0:
                    front = self.history['pareto_fronts'][frame]
                    # データをプロット (x: makespan, y: cost)
                    ax.scatter(front[:, 1], front[:, 0], 
                               color=colors[frame], 
                               alpha=0.7, 
                               s=50)
                
                ax.set_title(f"Generation {frame}")
                
                # 戻り値はアーティストのリスト
                return [ax]
            
            # アニメーションの作成と保存
            ani = animation.FuncAnimation(
                fig, update, 
                frames=len(self.history['pareto_fronts']),
                interval=400, 
                blit=False
            )
            
            ani.save(f"{save_dir}/pareto_animation_{timestamp}.gif", writer='pillow', fps=2, dpi=100)
            plt.close()
            
            print(f"アニメーションを保存しました: {save_dir}/pareto_animation_{timestamp}.gif")
        except Exception as e:
            print(f"アニメーション作成中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
        
    def get_final_pareto_front(self):
        """最終的なパレートフロントの取得"""
        # ランク1の個体を抽出
        pareto_front = [ind for ind in self.population if ind.rank == 1]
        objectives = np.array([ind.objectives for ind in pareto_front])
        chromosomes = [ind.chromosome for ind in pareto_front]
        
        return {
            'objectives': objectives,
            'chromosomes': chromosomes
        }
        
    def run(self, env, nb_jobs: int, verbose: bool = True, n_jobs=-1):
        """NSGA-IIによる最適化を実行（並列処理対応）"""
        # 初期集団の生成
        self.initialize_population(nb_jobs)
        
        # 初期集団の評価（並列）
        self.evaluate_population_parallel(env, n_jobs)
        
        # 非支配ソートと混雑度計算
        self.non_dominated_sort()
        self.calculate_crowding_distance()
        
        # 初期パレートフロントの保存
        self.save_pareto_front(0)
        
        if verbose:
            print(f"世代 0 - パレートフロントサイズ: {len([ind for ind in self.population if ind.rank == 1])}")
        
        # 世代を進める
        for generation in range(1, self.num_generations + 1):
            # 子孫集団の生成
            self.create_offspring()
            
            # 子孫集団の評価（並列）
            offspring_copy = self.offspring.copy()
            self.population = offspring_copy
            self.evaluate_population_parallel(env, n_jobs)
            evaluated_offspring = self.population.copy()
            
            # 親と子の集団を結合
            self.population = self.population + evaluated_offspring
            
            # 非支配ソートと混雑度計算
            self.non_dominated_sort()
            self.calculate_crowding_distance()
            
            # ランクと混雑度でソート
            self.population.sort(key=lambda x: (x.rank, -x.crowding_distance))
            
            # 上位pop_size個体を選択
            self.population = self.population[:self.pop_size]
            
            # パレートフロントの保存
            self.save_pareto_front(generation)
            
            if verbose and (generation % 10 == 0 or generation == self.num_generations):
                pareto_front = [ind for ind in self.population if ind.rank == 1]
                print(f"世代 {generation} - パレートフロントサイズ: {len(pareto_front)}")
                print(f"目的関数値の例: {pareto_front[0].objectives if pareto_front else 'なし'}")
                
        # 最終パレートフロントの可視化
        self.visualize_progress()
        
        return self.get_final_pareto_front()

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
