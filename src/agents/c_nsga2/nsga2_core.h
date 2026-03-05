#ifndef NSGA2_CORE_H
#define NSGA2_CORE_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// 支配関係の判定
// objectives1がobjectives2を支配する場合にtrueを返す
bool dominates(const double* objectives1, const double* objectives2, int32_t n_obj);

// 非支配ソート
// objectives_matrix: (n_pop, n_obj) の配列
// ranks: (n_pop,) の出力配列（ランクを格納）
void non_dominated_sort(
    const double* objectives_matrix,  // (n_pop, n_obj)
    int32_t* ranks,                    // (n_pop,) 出力
    int32_t n_pop,                     // 集団サイズ
    int32_t n_obj                      // 目的関数数
);

// 混雑度計算
// objectives_matrix: (n_pop, n_obj) の配列
// crowding_distances: (n_pop,) の出力配列
void calculate_crowding_distance(
    const double* objectives_matrix,   // (n_pop, n_obj)
    double* crowding_distances,        // (n_pop,) 出力
    int32_t n_pop,                     // 集団サイズ
    int32_t n_obj                      // 目的関数数
);

// トーナメント選択
// objectives_matrix: (n_pop, n_obj) の配列
// ranks: (n_pop,) のランク配列
// crowding_distances: (n_pop,) の混雑度配列
// selected_indices: (selection_size,) の出力配列（選択された個体のインデックス）
// random_seed: 乱数シード
void tournament_selection(
    const double* objectives_matrix,
    const int32_t* ranks,
    const double* crowding_distances,
    int32_t* selected_indices,         // (selection_size,) 出力
    int32_t n_pop,
    int32_t n_obj,
    int32_t selection_size,
    int32_t tournament_size,
    uint32_t random_seed
);

// 一点交叉
// parent1, parent2: (chromosome_length,) の親染色体
// child1, child2: (chromosome_length,) の出力配列
// crossover_point: 交叉点（1 <= crossover_point < chromosome_length）
void single_point_crossover(
    const int32_t* parent1,
    const int32_t* parent2,
    int32_t* child1,                   // 出力
    int32_t* child2,                   // 出力
    int32_t chromosome_length,
    int32_t crossover_point
);

// 二点交叉
// parent1, parent2: (chromosome_length,) の親染色体
// child1, child2: (chromosome_length,) の出力配列
// point1, point2: 交叉点（0 <= point1 < point2 < chromosome_length）
void two_point_crossover(
    const int32_t* parent1,
    const int32_t* parent2,
    int32_t* child1,                   // 出力
    int32_t* child2,                   // 出力
    int32_t chromosome_length,
    int32_t point1,
    int32_t point2
);

// 一様交叉
// parent1, parent2: (chromosome_length,) の親染色体
// child1, child2: (chromosome_length,) の出力配列
// mask: (chromosome_length,) のマスク配列（1ならparent1から、0ならparent2から）
void uniform_crossover(
    const int32_t* parent1,
    const int32_t* parent2,
    int32_t* child1,                   // 出力
    int32_t* child2,                   // 出力
    const int32_t* mask,
    int32_t chromosome_length
);

// 突然変異
// chromosome: (chromosome_length,) の染色体（入力出力）
// mutation_mask: (chromosome_length,) の突然変異マスク（1なら突然変異を適用）
void mutation(
    int32_t* chromosome,               // 入力出力
    const int32_t* mutation_mask,
    int32_t chromosome_length
);

// 重複排除（染色体の比較）
// chromosomes: (n_pop, chromosome_length) の配列
// unique_indices: (n_pop,) の出力配列（ユニークな個体のインデックス）
// n_unique: 出力（ユニークな個体数）
void eliminate_duplicates(
    const int32_t* chromosomes,        // (n_pop, chromosome_length)
    int32_t* unique_indices,            // (n_pop,) 出力
    int32_t* n_unique,                  // 出力
    int32_t n_pop,
    int32_t chromosome_length
);

// 評価関数のコールバック型（Pythonから環境のstep()を呼び出すための関数ポインタ）
// 戻り値: [cost, makespan] の配列（2要素）
typedef void (*EvaluateCallback)(const int32_t* chromosome, int32_t chromosome_length, double* objectives);

// 単一の個体を評価（コールバック関数を使用）
// chromosome: (chromosome_length,) の配列
// objectives: (2,) の出力配列 [cost, makespan]
// evaluate_callback: 環境のstep()を呼び出すコールバック関数
void evaluate_individual_c(
    const int32_t* chromosome,        // (chromosome_length,)
    double* objectives,               // (2,) 出力 [cost, makespan]
    int32_t chromosome_length,
    EvaluateCallback evaluate_callback
);

// 複数の個体を並列評価（コールバック関数を使用）
// chromosomes: (n_pop, chromosome_length) の配列
// objectives: (n_pop, 2) の出力配列
// evaluate_callback: 環境のstep()を呼び出すコールバック関数
// n_threads: 使用するスレッド数（0の場合は自動）
void evaluate_population_c(
    const int32_t* chromosomes,       // (n_pop, chromosome_length)
    double* objectives,               // (n_pop, 2) 出力
    int32_t n_pop,
    int32_t chromosome_length,
    EvaluateCallback evaluate_callback,
    int32_t n_threads
);

// 並列評価のためのスレッドプール初期化
// n_threads: 使用するスレッド数（0の場合は自動）
void init_evaluation_threads(int32_t n_threads);

// 並列評価のクリーンアップ
void cleanup_evaluation_threads(void);

#ifdef __cplusplus
}
#endif

#endif // NSGA2_CORE_H

