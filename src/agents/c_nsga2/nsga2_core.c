#include "nsga2_core.h"
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// 簡易的な乱数生成器（線形合同法）
static uint32_t g_seed = 1;

static void set_seed(uint32_t seed) {
    g_seed = seed;
}

static uint32_t rand_int(uint32_t max) {
    g_seed = g_seed * 1103515245 + 12345;
    return (g_seed / 2) % max;
}

static double rand_double() {
    g_seed = g_seed * 1103515245 + 12345;
    return ((double)(g_seed / 2) / (double)UINT32_MAX);
}

bool dominates(const double* objectives1, const double* objectives2, int32_t n_obj) {
    bool better_in_any = false;
    bool worse_in_any = false;
    
    for (int32_t i = 0; i < n_obj; i++) {
        if (objectives1[i] < objectives2[i]) {
            better_in_any = true;
        } else if (objectives1[i] > objectives2[i]) {
            worse_in_any = true;
        }
    }
    
    return better_in_any && !worse_in_any;
}

void non_dominated_sort(
    const double* objectives_matrix,
    int32_t* ranks,
    int32_t n_pop,
    int32_t n_obj
) {
    // ランクを初期化（0は未割り当て）
    for (int32_t i = 0; i < n_pop; i++) {
        ranks[i] = 0;
    }
    
    // 支配関係を計算
    // dominated_by[i] = iを支配する個体の数
    int32_t* dominated_by = (int32_t*)calloc(n_pop, sizeof(int32_t));
    // dominated_list[i][j] = iが支配する個体jのリスト（簡易版：最大n_pop個）
    int32_t** dominated_list = (int32_t**)malloc(n_pop * sizeof(int32_t*));
    int32_t* dominated_count = (int32_t*)calloc(n_pop, sizeof(int32_t));
    
    // 各リストのメモリを割り当て
    for (int32_t i = 0; i < n_pop; i++) {
        dominated_list[i] = (int32_t*)malloc(n_pop * sizeof(int32_t));
    }
    
    // O(N²)で支配関係を計算（OpenMPで並列化）
    // 注意: dominated_listとdominated_countへの書き込みが競合する可能性があるため、
    // 並列化は慎重に行う必要がある。各スレッドが異なるiを処理するため、
    // dominated_list[i]とdominated_count[i]への書き込みは安全だが、
    // dominated_by[j]への書き込みは競合する可能性がある。
    // そのため、atomic操作を使用するか、後で集計する必要がある。
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n_pop; i++) {
        // 各スレッドが独立して処理するため、dominated_list[i]とdominated_count[i]は安全
        for (int32_t j = 0; j < n_pop; j++) {
            if (i == j) continue;
            
            const double* obj_i = objectives_matrix + i * n_obj;
            const double* obj_j = objectives_matrix + j * n_obj;
            
            if (dominates(obj_i, obj_j, n_obj)) {
                // iがjを支配
                int32_t idx;
#ifdef _OPENMP
                #pragma omp atomic capture
#endif
                {
                    idx = dominated_count[i];
                    dominated_count[i]++;
                }
                dominated_list[i][idx] = j;
            } else if (dominates(obj_j, obj_i, n_obj)) {
                // jがiを支配
#ifdef _OPENMP
                #pragma omp atomic
#endif
                dominated_by[i]++;
            }
        }
    }
#else
    // OpenMPが利用できない場合は逐次実行
    for (int32_t i = 0; i < n_pop; i++) {
        for (int32_t j = 0; j < n_pop; j++) {
            if (i == j) continue;
            
            const double* obj_i = objectives_matrix + i * n_obj;
            const double* obj_j = objectives_matrix + j * n_obj;
            
            if (dominates(obj_i, obj_j, n_obj)) {
                // iがjを支配
                dominated_list[i][dominated_count[i]] = j;
                dominated_count[i]++;
            } else if (dominates(obj_j, obj_i, n_obj)) {
                // jがiを支配
                dominated_by[i]++;
            }
        }
    }
#endif
    
    // 第1フロントを決定（dominated_by[i] == 0の個体）
    int32_t current_rank = 1;
    int32_t* current_front = (int32_t*)malloc(n_pop * sizeof(int32_t));
    int32_t current_front_size = 0;
    
    for (int32_t i = 0; i < n_pop; i++) {
        if (dominated_by[i] == 0) {
            ranks[i] = current_rank;
            current_front[current_front_size++] = i;
        }
    }
    
    // 残りのフロントを決定
    while (current_front_size > 0) {
        current_rank++;
        int32_t* next_front = (int32_t*)malloc(n_pop * sizeof(int32_t));
        int32_t next_front_size = 0;
        
        // 現在のフロントの各個体が支配する個体のdominated_byを減らす
        for (int32_t f = 0; f < current_front_size; f++) {
            int32_t i = current_front[f];
            for (int32_t d = 0; d < dominated_count[i]; d++) {
                int32_t j = dominated_list[i][d];
                dominated_by[j]--;
                if (dominated_by[j] == 0 && ranks[j] == 0) {
                    ranks[j] = current_rank;
                    next_front[next_front_size++] = j;
                }
            }
        }
        
        free(current_front);
        current_front = next_front;
        current_front_size = next_front_size;
    }
    
    free(current_front);
    
    // メモリを解放
    for (int32_t i = 0; i < n_pop; i++) {
        free(dominated_list[i]);
    }
    free(dominated_list);
    free(dominated_count);
    free(dominated_by);
}

void calculate_crowding_distance(
    const double* objectives_matrix,
    double* crowding_distances,
    int32_t n_pop,
    int32_t n_obj
) {
    // 混雑度を初期化
    for (int32_t i = 0; i < n_pop; i++) {
        crowding_distances[i] = 0.0;
    }
    
    // 各目的関数について計算（OpenMPで並列化）
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int32_t obj_idx = 0; obj_idx < n_obj; obj_idx++) {
        // ソートインデックスを取得（バブルソートで簡易実装）
        int32_t* sorted_indices = (int32_t*)malloc(n_pop * sizeof(int32_t));
        for (int32_t i = 0; i < n_pop; i++) {
            sorted_indices[i] = i;
        }
        
        // バブルソート
        for (int32_t i = 0; i < n_pop - 1; i++) {
            for (int32_t j = 0; j < n_pop - i - 1; j++) {
                double val_j = objectives_matrix[sorted_indices[j] * n_obj + obj_idx];
                double val_j1 = objectives_matrix[sorted_indices[j + 1] * n_obj + obj_idx];
                if (val_j > val_j1) {
                    int32_t temp = sorted_indices[j];
                    sorted_indices[j] = sorted_indices[j + 1];
                    sorted_indices[j + 1] = temp;
                }
            }
        }
        
        // 最小値と最大値は無限大の混雑度
        crowding_distances[sorted_indices[0]] = DBL_MAX;
        crowding_distances[sorted_indices[n_pop - 1]] = DBL_MAX;
        
        // 目的関数の範囲を計算
        double obj_min = objectives_matrix[sorted_indices[0] * n_obj + obj_idx];
        double obj_max = objectives_matrix[sorted_indices[n_pop - 1] * n_obj + obj_idx];
        double obj_range = obj_max - obj_min;
        
        if (obj_range > 1e-10) {  // ゼロ除算を避ける
        // 中間の個体の混雑度を計算（OpenMPで並列化）
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int32_t i = 1; i < n_pop - 1; i++) {
                int32_t idx = sorted_indices[i];
                int32_t prev_idx = sorted_indices[i - 1];
                int32_t next_idx = sorted_indices[i + 1];
                
                double distance = (objectives_matrix[next_idx * n_obj + obj_idx] - 
                                 objectives_matrix[prev_idx * n_obj + obj_idx]) / obj_range;
                crowding_distances[idx] += distance;
            }
        }
        
        free(sorted_indices);
    }
}

void tournament_selection(
    const double* objectives_matrix,
    const int32_t* ranks,
    const double* crowding_distances,
    int32_t* selected_indices,
    int32_t n_pop,
    int32_t n_obj,
    int32_t selection_size,
    int32_t tournament_size,
    uint32_t random_seed
) {
    set_seed(random_seed);
    
    // トーナメント選択（並列化可能だが、乱数の一貫性のため順次実行）
    for (int32_t s = 0; s < selection_size; s++) {
        // トーナメントの候補をランダムに選択
        int32_t* candidates = (int32_t*)malloc(tournament_size * sizeof(int32_t));
        for (int32_t t = 0; t < tournament_size; t++) {
            candidates[t] = rand_int(n_pop);
        }
        
        // 最良の候補を選択（ランクが低い、またはランクが同じなら混雑度が大きい）
        int32_t winner = candidates[0];
        for (int32_t t = 1; t < tournament_size; t++) {
            int32_t candidate = candidates[t];
            if (ranks[candidate] < ranks[winner]) {
                winner = candidate;
            } else if (ranks[candidate] == ranks[winner] && 
                      crowding_distances[candidate] > crowding_distances[winner]) {
                winner = candidate;
            }
        }
        
        selected_indices[s] = winner;
        free(candidates);
    }
}

void single_point_crossover(
    const int32_t* parent1,
    const int32_t* parent2,
    int32_t* child1,
    int32_t* child2,
    int32_t chromosome_length,
    int32_t crossover_point
) {
    // child1 = parent1[:point] + parent2[point:]
    memcpy(child1, parent1, crossover_point * sizeof(int32_t));
    memcpy(child1 + crossover_point, parent2 + crossover_point, 
           (chromosome_length - crossover_point) * sizeof(int32_t));
    
    // child2 = parent2[:point] + parent1[point:]
    memcpy(child2, parent2, crossover_point * sizeof(int32_t));
    memcpy(child2 + crossover_point, parent1 + crossover_point, 
           (chromosome_length - crossover_point) * sizeof(int32_t));
}

void two_point_crossover(
    const int32_t* parent1,
    const int32_t* parent2,
    int32_t* child1,
    int32_t* child2,
    int32_t chromosome_length,
    int32_t point1,
    int32_t point2
) {
    // child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    memcpy(child1, parent1, point1 * sizeof(int32_t));
    memcpy(child1 + point1, parent2 + point1, (point2 - point1) * sizeof(int32_t));
    memcpy(child1 + point2, parent1 + point2, (chromosome_length - point2) * sizeof(int32_t));
    
    // child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    memcpy(child2, parent2, point1 * sizeof(int32_t));
    memcpy(child2 + point1, parent1 + point1, (point2 - point1) * sizeof(int32_t));
    memcpy(child2 + point2, parent2 + point2, (chromosome_length - point2) * sizeof(int32_t));
}

void uniform_crossover(
    const int32_t* parent1,
    const int32_t* parent2,
    int32_t* child1,
    int32_t* child2,
    const int32_t* mask,
    int32_t chromosome_length
) {
    for (int32_t i = 0; i < chromosome_length; i++) {
        if (mask[i]) {
            child1[i] = parent1[i];
            child2[i] = parent2[i];
        } else {
            child1[i] = parent2[i];
            child2[i] = parent1[i];
        }
    }
}

void mutation(
    int32_t* chromosome,
    const int32_t* mutation_mask,
    int32_t chromosome_length
) {
    for (int32_t i = 0; i < chromosome_length; i++) {
        if (mutation_mask[i]) {
            chromosome[i] = 1 - chromosome[i];  // ビット反転
        }
    }
}

void eliminate_duplicates(
    const int32_t* chromosomes,
    int32_t* unique_indices,
    int32_t* n_unique,
    int32_t n_pop,
    int32_t chromosome_length
) {
    *n_unique = 0;
    bool* is_unique = (bool*)calloc(n_pop, sizeof(bool));
    
    // 重複チェック（並列化可能だが、順次実行で一貫性を保つ）
    for (int32_t i = 0; i < n_pop; i++) {
        bool duplicate = false;
        
        // 既に追加された個体と比較
        for (int32_t j = 0; j < *n_unique; j++) {
            int32_t idx = unique_indices[j];
            bool same = true;
            for (int32_t k = 0; k < chromosome_length; k++) {
                if (chromosomes[i * chromosome_length + k] != 
                    chromosomes[idx * chromosome_length + k]) {
                    same = false;
                    break;
                }
            }
            if (same) {
                duplicate = true;
                break;
            }
        }
        
        if (!duplicate) {
            unique_indices[*n_unique] = i;
            (*n_unique)++;
        }
    }
    
    free(is_unique);
}

// 並列評価のためのスレッドプール初期化
void init_evaluation_threads(int32_t n_threads) {
#ifdef _OPENMP
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }
    // デフォルトではOpenMPが自動的にスレッド数を決定
#endif
}

// 並列評価のクリーンアップ
void cleanup_evaluation_threads(void) {
    // 現在は特にクリーンアップ処理は不要
    // 将来的にスレッドプールを使用する場合はここでクリーンアップ
}

// 単一の個体を評価（コールバック関数を使用）
void evaluate_individual_c(
    const int32_t* chromosome,
    double* objectives,
    int32_t chromosome_length,
    EvaluateCallback evaluate_callback
) {
    // コールバック関数を呼び出して評価
    evaluate_callback(chromosome, chromosome_length, objectives);
}

// 複数の個体を並列評価（コールバック関数を使用）
void evaluate_population_c(
    const int32_t* chromosomes,
    double* objectives,
    int32_t n_pop,
    int32_t chromosome_length,
    EvaluateCallback evaluate_callback,
    int32_t n_threads
) {
    // OpenMPで並列化
#ifdef _OPENMP
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }
    
    #pragma omp parallel for
    for (int32_t i = 0; i < n_pop; i++) {
        const int32_t* chromosome = chromosomes + i * chromosome_length;
        double* obj = objectives + i * 2;  // [cost, makespan]
        evaluate_callback(chromosome, chromosome_length, obj);
    }
#else
    // OpenMPが利用できない場合は逐次実行
    for (int32_t i = 0; i < n_pop; i++) {
        const int32_t* chromosome = chromosomes + i * chromosome_length;
        double* obj = objectives + i * 2;  // [cost, makespan]
        evaluate_callback(chromosome, chromosome_length, obj);
    }
#endif
}

