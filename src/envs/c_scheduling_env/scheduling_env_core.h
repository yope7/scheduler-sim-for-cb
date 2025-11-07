#ifndef SCHEDULING_ENV_CORE_H
#define SCHEDULING_ENV_CORE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// 空きノードリストの構造体（サイズ情報を含む）
typedef struct {
    int32_t* nodes;     // ノードインデックスの配列
    int32_t size;       // 配列のサイズ
} FreeNodesList;

// キャッシュ構造体
typedef struct {
    int32_t* free_per_col;      // 各列の空きノード数
    int32_t* prefix_sum;        // 2D累積和 (H+1) x (W+1)
    FreeNodesList* free_nodes_list;  // 各列の空きノードリスト（サイズ情報を含む）
    int32_t* occ;               // 占有マトリックス
    int32_t H;                  // 高さ
    int32_t W;                  // 幅
    int32_t version;           // バージョン番号
} WindowCache;

// 位置情報（連続割り当て）
typedef struct {
    int32_t i;  // 行開始位置
    int32_t a;  // 列開始位置
} Position2D;

// 位置情報（分散割り当て）
typedef struct {
    int32_t i;              // 行開始位置（分散割り当てでは0）
    int32_t a;              // 列開始位置
    int32_t* node_allocation; // ノード割り当てリスト（フラット化）
    int32_t allocation_size; // 割り当てサイズ
} PositionDistributed;

// 位置情報（統合）
typedef struct {
    bool is_distributed;    // 分散割り当てかどうか
    union {
        Position2D continuous;
        PositionDistributed distributed;
    } pos;
} Position;

// 結果構造体
typedef struct {
    bool found;             // 位置が見つかったか
    Position position;       // 位置情報
    double waiting_time;     // 待ち時間
} AllocationResult;

// キャッシュの構築
WindowCache* build_cache(
    const int32_t* window_status,  // ウィンドウの状態 (H x W)
    int32_t H,                      // 高さ
    int32_t W                       // 幅
);

// キャッシュの解放
void free_cache(WindowCache* cache);

// 割り当て位置の探索
AllocationResult find_allocation_position(
    const WindowCache* cache,    // キャッシュ
    int32_t job_width,           // ジョブの幅（処理時間）
    int32_t job_height,          // ジョブの高さ（ノード数）
    int32_t when_submitted,      // 提出時刻
    int32_t current_time         // 現在時刻
);

// 時間遷移（スライドウィンドウ）
void time_transition(
    int32_t* window_status,      // ウィンドウの状態 (H x W) - 入力出力
    int32_t* window_job_id,      // ウィンドウのジョブID (H x W) - 入力出力
    int32_t H,                   // 高さ
    int32_t W,                   // 幅
    bool slide                   // スライドするかどうか
);

// ジョブのスケジュール実行
void do_schedule(
    int32_t* window_status,      // ウィンドウの状態 (H x W) - 入力出力
    int32_t* window_job_id,      // ウィンドウのジョブID (H x W) - 入力出力
    int32_t H,                   // 高さ
    int32_t W,                   // 幅
    int32_t job_width,           // ジョブの幅
    int32_t job_height,          // ジョブの高さ
    int32_t job_id,              // ジョブID
    const Position* position     // 位置情報
);

// ユニークなジョブIDの取得
int32_t* get_unique_job_ids(
    const int32_t* history_matrix,  // 履歴マトリックス (H x W)
    int32_t H,                       // 高さ
    int32_t W,                       // 幅
    int32_t max_job_id,              // 最大ジョブID
    int32_t* count                   // 出力: ユニークなジョブIDの数
);

// makespanの計算
int32_t calculate_makespan(
    const int32_t* window_matrix,    // ウィンドウマトリックス (H x W)
    int32_t H,                       // 高さ
    int32_t W                        // 幅
);

// キャッシュの差分更新（ジョブ追加時）
void update_cache_incremental(
    WindowCache* cache,              // キャッシュ（入力出力）
    const int32_t* window_status,   // ウィンドウの状態 (H x W)
    int32_t i_start,                // 行開始位置
    int32_t i_end,                  // 行終了位置（含まない）
    int32_t a_start,                // 列開始位置
    int32_t a_end                   // 列終了位置（含まない）
);

// キャッシュの差分更新（時間遷移時）
void update_cache_time_transition(
    WindowCache* cache,              // キャッシュ（入力出力）
    const int32_t* window_status    // ウィンドウの状態 (H x W)
);

#ifdef __cplusplus
}
#endif

#endif // SCHEDULING_ENV_CORE_H

