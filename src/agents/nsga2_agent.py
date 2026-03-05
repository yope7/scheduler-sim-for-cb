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
# Rayは削除、multiprocessingを使用

# C言語実装版NSGA-IIをインポート
try:
    import nsga2_core
    C_NSGA2_AVAILABLE = True
except ImportError:
    C_NSGA2_AVAILABLE = False
    print("警告: NSGA-II C言語実装が利用できません。Numba実装を使用します。")

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Numbaが利用できない場合は空のデコレータを定義
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(x):
        return range(x)

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

def _evaluate_individual_worker_multiprocessing(args):
    """multiprocessing用の個体評価関数（グローバル関数、各プロセスで実行）"""
    chromosome, env_params = args
    
    # 各プロセスで環境を作成（C実装版環境を使用）
    from src.envs.c_scheduling_env.scheduling_env_cache_optimized import SchedulingEnvCacheOptimized
    env = SchedulingEnvCacheOptimized(**env_params)
    
    # 評価実行
    result_tuple = evaluate_individual_pure_step_time(env, chromosome)
    
    # evaluate_individual_pure_step_timeは ([cost, avg_waiting_time], pure_step_time) を返す
    if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
        objectives, _ = result_tuple
        return np.array(objectives, dtype=np.float64)
    else:
        # フォールバック
        return np.array(result_tuple, dtype=np.float64)

# NumPy/Numba化された高速な支配関係判定関数
@njit(cache=True, fastmath=True)
def dominates_numba(objectives1, objectives2):
    """NumPy配列での高速な支配関係判定（最小化問題）"""
    better_in_any = False
    worse_in_any = False
    
    for i in range(len(objectives1)):
        if objectives1[i] < objectives2[i]:
            better_in_any = True
        elif objectives1[i] > objectives2[i]:
            worse_in_any = True
    
    return better_in_any and not worse_in_any

# NumPy/Numba化された非支配ソート
@njit(cache=True, fastmath=True)
def non_dominated_sort_numba(objectives_matrix):
    """
    NumPy配列での高速な非支配ソート
    Args:
        objectives_matrix: (n_pop, n_obj) のNumPy配列
    Returns:
        ranks: (n_pop,) のランク配列
    """
    n_pop = objectives_matrix.shape[0]
    n_obj = objectives_matrix.shape[1]
    ranks = np.zeros(n_pop, dtype=np.int32)
    
    # 支配関係マトリクスを計算（iがjを支配する場合dominates_matrix[i, j] = 1）
    # 最大サイズで事前割り当て（各要素が支配する最大数を仮定）
    max_dominated = n_pop  # 最大で全個体を支配する可能性
    dominated_matrix = np.zeros((n_pop, n_pop), dtype=np.int32)
    dominated_counts = np.zeros(n_pop, dtype=np.int32)
    domination_counts = np.zeros(n_pop, dtype=np.int32)
    
    # O(N²)で支配関係を計算（並列化はデータ依存のため通常ループ）
    for i in range(n_pop):
        for j in range(n_pop):
            if i == j:
                continue
            
            better_any = False
            worse_any = False
            
            for k in range(n_obj):
                if objectives_matrix[i, k] < objectives_matrix[j, k]:
                    better_any = True
                elif objectives_matrix[i, k] > objectives_matrix[j, k]:
                    worse_any = True
            
            if better_any and not worse_any:
                # iがjを支配
                dominated_matrix[i, dominated_counts[i]] = j
                dominated_counts[i] += 1
            elif not better_any and worse_any:
                # jがiを支配
                domination_counts[i] += 1
    
    # 第1フロントを決定
    current_rank = 1
    for i in range(n_pop):
        if domination_counts[i] == 0:
            ranks[i] = current_rank
    
    # 残りのフロントを決定
    while True:
        current_members = np.zeros(n_pop, dtype=np.int32)
        current_members_size = 0
        
        for i in range(n_pop):
            if ranks[i] == current_rank:
                current_members[current_members_size] = i
                current_members_size += 1
        
        if current_members_size == 0:
            break
        
        next_front_size = 0
        next_front = np.zeros(n_pop, dtype=np.int32)
        
        for i_idx in range(current_members_size):
            i = current_members[i_idx]
            for j_idx in range(dominated_counts[i]):
                j = dominated_matrix[i, j_idx]
                domination_counts[j] -= 1
                if domination_counts[j] == 0 and ranks[j] == 0:
                    ranks[j] = current_rank + 1
                    next_front[next_front_size] = j
                    next_front_size += 1
        
        if next_front_size == 0:
            break
        
        current_rank += 1
    
    return ranks

# NumPy化された混雑度計算
@njit(cache=True, fastmath=True)
def calculate_crowding_distance_numba(objectives_matrix):
    """
    NumPy配列での高速な混雑度計算
    Args:
        objectives_matrix: (n_pop, n_obj) のNumPy配列
    Returns:
        crowding_distances: (n_pop,) の混雑度配列
    """
    n_pop = objectives_matrix.shape[0]
    n_obj = objectives_matrix.shape[1]
    crowding_distances = np.zeros(n_pop, dtype=np.float64)
    
    # 各目的関数について計算
    for obj_idx in range(n_obj):
        # ソートインデックスを取得
        sorted_indices = np.argsort(objectives_matrix[:, obj_idx])
        
        # 最小値と最大値は無限大の混雑度
        crowding_distances[sorted_indices[0]] = np.inf
        crowding_distances[sorted_indices[-1]] = np.inf
        
        # 目的関数の範囲を計算
        obj_range = (objectives_matrix[sorted_indices[-1], obj_idx] - 
                    objectives_matrix[sorted_indices[0], obj_idx])
        
        if obj_range == 0:
            continue
        
        # 中間の個体の混雑度を計算
        for i in range(1, n_pop - 1):
            idx = sorted_indices[i]
            prev_idx = sorted_indices[i - 1]
            next_idx = sorted_indices[i + 1]
            
            distance = (objectives_matrix[next_idx, obj_idx] - 
                       objectives_matrix[prev_idx, obj_idx]) / obj_range
            crowding_distances[idx] += distance
    
    return crowding_distances

# JITウォームアップ関数（代表的な形状で事前コンパイル）
def warmup_numba_functions():
    """
    Numba関数のウォームアップを実行（初回JITコンパイルを事前に完了）
    代表的なサイズのダミー入力で1回ずつ実行してコンパイル済みにする
    """
    if not NUMBA_AVAILABLE:
        return
    
    try:
        # 代表的なサイズを想定（実際の使用状況に合わせて調整）
        typical_pop_size = 200
        n_obj = 2
        
        # ダミーデータ生成
        dummy_objectives1 = np.array([100.0, 10.0], dtype=np.float64)
        dummy_objectives2 = np.array([110.0, 8.0], dtype=np.float64)
        dummy_matrix = np.random.rand(typical_pop_size, n_obj).astype(np.float64) * 1000.0
        
        # 各関数を1回ずつ実行してJITコンパイル
        _ = dominates_numba(dummy_objectives1, dummy_objectives2)
        _ = non_dominated_sort_numba(dummy_matrix)
        _ = calculate_crowding_distance_numba(dummy_matrix)
        
        if NUMBA_AVAILABLE:
            # デバッグ: コンパイル状況を確認
            try:
                sigs_dom = dominates_numba.nopython_signatures
                sigs_sort = non_dominated_sort_numba.nopython_signatures
                sigs_dist = calculate_crowding_distance_numba.nopython_signatures
                # print(f"[Numbaウォームアップ] コンパイル済みシグネチャ数: dominates={len(sigs_dom)}, sort={len(sigs_sort)}, distance={len(sigs_dist)}")
            except AttributeError:
                pass  # シグネチャ確認はオプショナル
    except Exception as e:
        print(f"[Numbaウォームアップ警告] {e}")

@dataclass
class Individual:
    """個体クラス"""
    chromosome: List[int]
    objectives: Optional[np.ndarray] = None
    rank: int = 0
    crowding_distance: float = 0.0
    
    def dominates(self, other: 'Individual') -> bool:
        """自分が他の個体を支配するかどうか判定（NumPy/Numba化版）"""
        # objectivesがNoneの場合は比較できない
        if self.objectives is None or other.objectives is None:
            return False
        
        # NumPy配列として処理
        if isinstance(self.objectives, np.ndarray) and isinstance(other.objectives, np.ndarray):
            return dominates_numba(self.objectives, other.objectives)
        
        # フォールバック（既存の実装）
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
        
        # multiprocessing用の変数
        self.pool = None  # プロセスプール
        self.env_params = None  # 環境パラメータ
        
        # 実行ディレクトリの作成
        self.execution_dir = self.create_execution_directory()
        
        # Numba関数のウォームアップ（メインプロセス側で事前コンパイル）
        if NUMBA_AVAILABLE:
            try:
                warmup_numba_functions()
            except Exception as e:
                print(f"[警告] Numbaウォームアップ失敗: {e}")
        
        # C実装版の並列評価スレッドを初期化
        if C_NSGA2_AVAILABLE:
            try:
                n_threads = max(1, os.cpu_count() // 4)  # CPUコア数の半分を使用
                nsga2_core.init_evaluation_threads(n_threads)
                print(f"[NSGA-II C実装] 並列評価スレッド数を設定: {n_threads}")
            except Exception as e:
                print(f"[警告] C実装版の並列評価初期化失敗: {e}")
    
    def __del__(self):
        """デストラクタ: プロセスプールをクリーンアップ"""
        if self.pool is not None:
            try:
                self.pool.close()
                self.pool.join()
            except Exception:
                pass  # クリーンアップ中のエラーは無視
    
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
        """多様性を考慮した初期集団の生成（NumPy化版）"""
        self.population = []
        
        # 1. 基本的なランダム個体（30%）- NumPy化
        num_random = int(self.pop_size * 0.3)
        if num_random > 0:
            random_chromosomes = np.random.randint(0, 2, size=(num_random, nb_jobs), dtype=np.int32)
            for chrom in random_chromosomes:
                self.population.append(Individual(chrom.tolist()))
        
        # 2. 重み付き個体（30%）
        weighted_individuals = self.create_weighted_individuals(nb_jobs, int(self.pop_size * 0.3))
        self.population.extend(weighted_individuals)
        
        # 3. 極端な個体（20%）- NumPy化
        num_extreme = int(self.pop_size * 0.1)
        for _ in range(num_extreme):
            # すべてオンプレミス
            all_on_premise = np.zeros(nb_jobs, dtype=np.int32)
            self.population.append(Individual(all_on_premise.tolist()))
            
            # すべてクラウド
            all_cloud = np.ones(nb_jobs, dtype=np.int32)
            self.population.append(Individual(all_cloud.tolist()))
        
        # 4. パターン個体（20%）
        pattern_individuals = self.create_pattern_individuals(nb_jobs, int(self.pop_size * 0.2))
        self.population.extend(pattern_individuals)
        
        # 重複排除を実行
        self.population = self.eliminate_duplicate_individuals(self.population)
        
        # 重複排除後、必要に応じて個体数を調整 - NumPy化
        while len(self.population) < self.pop_size:
            chromosome = np.random.randint(0, 2, size=nb_jobs, dtype=np.int32)
            new_individual = Individual(chromosome.tolist())
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

    def evaluate_population_openmp(self, env, n_jobs=-1):
        """
        multiprocessing版の個体評価（C実装版環境を使用）
        - 各プロセスで環境を作成して評価
        - OpenMPと同様の並列処理を実現
        """
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
        
        # 環境パラメータが変わった場合はプロセスプールを再作成
        if self.env_params != env_params:
            self.env_params = env_params
            # 既存のプールをクリーンアップ
            if self.pool is not None:
                self.pool.close()
                self.pool.join()
                self.pool = None
        
        # プロセスプールの作成（必要に応じて）
        if self.pool is None:
            num_workers = max(1, os.cpu_count() // 2) if n_jobs == -1 else max(1, n_jobs)
            self.pool = mp.Pool(processes=num_workers)
        
        # 評価タスクの準備
        tasks = [(ind.chromosome, self.env_params) for ind in individuals_to_evaluate]
        
        # 並列評価実行
        t_start = time.perf_counter()
        results = self.pool.map(_evaluate_individual_worker_multiprocessing, tasks)
        t_end = time.perf_counter()
        
        # 結果を個体に割り当て
        for i, result in enumerate(results):
            if isinstance(result, np.ndarray):
                individuals_to_evaluate[i].objectives = result
            elif isinstance(result, (list, tuple)):
                individuals_to_evaluate[i].objectives = np.array(result, dtype=np.float64)
            else:
                # エラーが発生した場合
                individuals_to_evaluate[i].objectives = np.array([1e6, 1e6], dtype=np.float64)

    def non_dominated_sort(self):
        """非支配ソーティング（C実装版優先）"""
        # 評価済み個体のみを対象
        evaluated_indices = []
        evaluated_objectives = []
        
        for i, individual in enumerate(self.population):
            if individual.objectives is not None:
                evaluated_indices.append(i)
                evaluated_objectives.append(individual.objectives)
        
        if not evaluated_objectives:
            return
        
        # NumPy配列に変換
        objectives_matrix = np.array(evaluated_objectives, dtype=np.float64)
        
        # C実装版を優先使用
        if C_NSGA2_AVAILABLE:
            ranks_array = nsga2_core.non_dominated_sort(objectives_matrix)
        elif NUMBA_AVAILABLE:
            ranks_array = non_dominated_sort_numba(objectives_matrix)
        else:
            # Numbaが利用できない場合はフォールバック
            ranks_array = self._non_dominated_sort_fallback(objectives_matrix)
        
        # 結果を個体に割り当て
        for idx, i in enumerate(evaluated_indices):
            self.population[i].rank = int(ranks_array[idx])
    
    def _non_dominated_sort_fallback(self, objectives_matrix):
        """Numbaが利用できない場合のフォールバック実装"""
        n_pop = len(objectives_matrix)
        ranks = np.zeros(n_pop, dtype=np.int32)
        domination_counts = np.zeros(n_pop, dtype=np.int32)
        dominated_sets = [[] for _ in range(n_pop)]
        
        # 支配関係を計算
        for i in range(n_pop):
            for j in range(n_pop):
                if i == j:
                    continue
                
                better_any = False
                worse_any = False
                
                for k in range(objectives_matrix.shape[1]):
                    if objectives_matrix[i, k] < objectives_matrix[j, k]:
                        better_any = True
                    elif objectives_matrix[i, k] > objectives_matrix[j, k]:
                        worse_any = True
                
                if better_any and not worse_any:
                    dominated_sets[i].append(j)
                elif not better_any and worse_any:
                    domination_counts[i] += 1
        
        # 第1フロントを決定
        current_rank = 1
        for i in range(n_pop):
            if domination_counts[i] == 0:
                ranks[i] = current_rank
        
        # 残りのフロントを決定
        while True:
            current_members = [i for i in range(n_pop) if ranks[i] == current_rank]
            if not current_members:
                break
            
            next_front = []
            for i in current_members:
                for j in dominated_sets[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        ranks[j] = current_rank + 1
                        next_front.append(j)
            
            if not next_front:
                break
            
            current_rank += 1
        
        return ranks
            
    def calculate_crowding_distance(self):
        """混雑度の計算（C実装版優先）"""
        # 評価済み個体のみを対象
        evaluated_indices = []
        evaluated_objectives = []
        
        for i, individual in enumerate(self.population):
            if individual.objectives is not None:
                evaluated_indices.append(i)
                evaluated_objectives.append(individual.objectives)
        
        if not evaluated_objectives:
            # 全ての個体の混雑度を0に設定
            for individual in self.population:
                individual.crowding_distance = 0.0
            return
        
        # NumPy配列に変換
        objectives_matrix = np.array(evaluated_objectives, dtype=np.float64)
        
        # C実装版を優先使用
        if C_NSGA2_AVAILABLE:
            crowding_distances = nsga2_core.calculate_crowding_distance(objectives_matrix)
        elif NUMBA_AVAILABLE:
            crowding_distances = calculate_crowding_distance_numba(objectives_matrix)
        else:
            # Numbaが利用できない場合はフォールバック
            crowding_distances = self._calculate_crowding_distance_fallback(objectives_matrix)
        
        # 結果を個体に割り当て
        for idx, i in enumerate(evaluated_indices):
            self.population[i].crowding_distance = float(crowding_distances[idx])
        
        # 評価されていない個体の混雑度は0のまま
    
    def _calculate_crowding_distance_fallback(self, objectives_matrix):
        """Numbaが利用できない場合のフォールバック実装"""
        n_pop = objectives_matrix.shape[0]
        n_obj = objectives_matrix.shape[1]
        crowding_distances = np.zeros(n_pop, dtype=np.float64)
        
        # 各目的関数について計算
        for obj_idx in range(n_obj):
            # ソートインデックスを取得
            sorted_indices = np.argsort(objectives_matrix[:, obj_idx])
            
            # 最小値と最大値は無限大の混雑度
            crowding_distances[sorted_indices[0]] = np.inf
            crowding_distances[sorted_indices[-1]] = np.inf
            
            # 目的関数の範囲を計算
            obj_range = (objectives_matrix[sorted_indices[-1], obj_idx] - 
                        objectives_matrix[sorted_indices[0], obj_idx])
            
            if obj_range == 0:
                continue
            
            # 中間の個体の混雑度を計算
            for i in range(1, n_pop - 1):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i - 1]
                next_idx = sorted_indices[i + 1]
                
                distance = (objectives_matrix[next_idx, obj_idx] - 
                           objectives_matrix[prev_idx, obj_idx]) / obj_range
                crowding_distances[idx] += distance
        
        return crowding_distances
                
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
        """多様な交叉戦略（NumPy化版）"""
        if random.random() > self.crossover_prob:
            return parent1, parent2
        
        # NumPy配列に変換
        p1 = np.array(parent1.chromosome, dtype=np.int32)
        p2 = np.array(parent2.chromosome, dtype=np.int32)
        n = len(p1)
        
        crossover_type = random.random()
        
        if crossover_type < 0.4:
            # 一点交叉
            point = random.randint(1, n - 1)
            child1 = np.concatenate([p1[:point], p2[point:]])
            child2 = np.concatenate([p2[:point], p1[point:]])
        
        elif crossover_type < 0.7:
            # 二点交叉
            points = sorted(random.sample(range(1, n), 2))
            child1 = np.concatenate([p1[:points[0]], p2[points[0]:points[1]], p1[points[1]:]])
            child2 = np.concatenate([p2[:points[0]], p1[points[0]:points[1]], p2[points[1]:]])
        
        else:
            # 一様交叉
            mask = np.random.randint(0, 2, size=n, dtype=np.int32)
            child1 = np.where(mask, p1, p2)
            child2 = np.where(mask, p2, p1)
        
        # リストに変換してIndividualオブジェクトを作成
        child1 = Individual(child1.tolist())
        child2 = Individual(child2.tolist())
        
        return child1, child2
        
    def mutation(self, individual: Individual):
        """強化された突然変異戦略（NumPy化版）"""
        # NumPy配列に変換
        chromosome = np.array(individual.chromosome, dtype=np.int32)
        n = len(chromosome)
        
        # 基本突然変異
        mutation_mask = np.random.random(n) < self.mutation_prob
        chromosome[mutation_mask] = 1 - chromosome[mutation_mask]
        
        # 追加の突然変異戦略
        # 1. ビット反転（確率0.1）
        if random.random() < 0.1:
            flip_count = random.randint(1, 3)  # 1-3ビットを反転
            flip_indices = np.random.choice(n, size=min(flip_count, n), replace=False)
            chromosome[flip_indices] = 1 - chromosome[flip_indices]
        
        # 2. 部分的な再生成（確率0.05）
        if random.random() < 0.05:
            start = random.randint(0, max(0, n - 8))
            end = min(start + 8, n)
            chromosome[start:end] = np.random.randint(0, 2, size=end-start, dtype=np.int32)
        
        # リストに戻す
        individual.chromosome = chromosome.tolist()
                
    def create_weighted_individuals(self, nb_jobs: int, num_individuals: int = 20):
        """重み付き目的関数による多様な個体生成（NumPy化版）"""
        weighted_individuals = []
        
        if num_individuals == 0:
            return weighted_individuals
        
        # コストと待ち時間の重みをランダムに設定（ベクトル化）
        cost_weights = np.random.uniform(0.1, 0.9, size=num_individuals)
        
        # 異なる重みの組み合わせで個体を生成
        for i in range(num_individuals):
            cost_weight = cost_weights[i]
            
            # 重みに基づいて染色体を生成（NumPy化）
            random_values = np.random.random(nb_jobs)
            chromosome = np.where(random_values < cost_weight, 0, 1).astype(np.int32)
            
            weighted_individuals.append(Individual(chromosome.tolist()))
        
        return weighted_individuals

    def create_pattern_individuals(self, nb_jobs: int, num_individuals: int):
        """パターンに基づく個体生成（NumPy化版）"""
        patterns = []
        
        # 交互パターン
        alternating = np.arange(nb_jobs, dtype=np.int32) % 2
        patterns.append(Individual(alternating.tolist()))
        
        # ブロックパターン
        block_size = max(1, nb_jobs // 4)
        for i in range(num_individuals - 1):
            pattern = (np.arange(nb_jobs, dtype=np.int32) // block_size) % 2
            patterns.append(Individual(pattern.tolist()))
        
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
        self.evaluate_population_openmp(env, n_jobs)
        
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
            self.evaluate_population_openmp(env, n_jobs)

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
        # self.create_execution_summary()

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
        """多様性維持のためのランダム個体注入（NumPy化版）"""
        # 最下位の10%をランダム個体で置き換え
        replace_count = max(1, self.pop_size // 10)
        
        for i in range(replace_count):
            if len(self.population) > replace_count:
                # ランダム個体を生成（NumPy化）
                new_chromosome = np.random.randint(0, 2, size=nb_jobs, dtype=np.int32)
                new_individual = Individual(new_chromosome.tolist())
                
                # 最下位の個体を置き換え
                self.population[-(i+1)] = new_individual
