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
    int32_t* scratch_mins;     // find用スクラッチ (size W)
    int32_t* scratch_deque;    // sliding_window_min用 (size W)
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
    int32_t current_time,        // 現在時刻
    bool continuous_only         // true: 連続矩形のみ（クラウド向け）
);

// 時間遷移（スライドウィンドウ）
void time_transition(
    int32_t* window_status,      // ウィンドウの状態 (H x W) - 入力出力
    int32_t* window_job_id,      // ウィンドウのジョブID (H x W) - 入力出力
    int32_t H,                   // 高さ
    int32_t W,                   // 幅
    bool slide                   // スライドするかどうか
);

// リングバッファ版: キャッシュ構築（head=最古列の物理インデックス）
WindowCache* build_cache_from_ringbuffer(
    const int32_t* window_status,
    int32_t H,
    int32_t W,
    int32_t head
);

// リングバッファ版: 時間遷移（列headをクリアし、headを進める）
// 戻り値: 新しいhead = (head+1) % W
int32_t time_transition_ringbuffer(
    int32_t* window_status,
    int32_t* window_job_id,
    int32_t H,
    int32_t W,
    int32_t head
);

// リングバッファ版: ジョブスケジュール（positionのaは論理列）
void do_schedule_ringbuffer(
    int32_t* window_status,
    int32_t* window_job_id,
    int32_t H,
    int32_t W,
    int32_t job_width,
    int32_t job_height,
    int32_t job_id,
    const Position* position,
    int32_t head
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

// リングバッファ版: キャッシュの差分更新（時間遷移時、ウィンドウ参照不要・シフトのみ）
void update_cache_time_transition_ringbuffer(WindowCache* cache);

// リングバッファ版: キャッシュの差分更新（ジョブ追加時、論理列→物理列マッピング）
void update_cache_incremental_ringbuffer(
    WindowCache* cache,
    const int32_t* window_status,
    int32_t i_start,
    int32_t i_end,
    int32_t a_start,
    int32_t a_end,
    int32_t head
);

// キャッシュの再構築（最適化版: バージョンチェックと差分更新を含む）
// cache: 既存のキャッシュ（NULLの場合は新規作成）
// window_status: ウィンドウの状態 (H x W)
// H, W: 高さと幅
// current_version: 現在のバージョン
// cache_version: キャッシュのバージョン（入力出力）
// window_changed: 変更フラグ（入力出力）
// 戻り値: 更新されたキャッシュ（新規作成または既存）
WindowCache* rebuild_cache_if_needed(
    WindowCache* cache,              // 既存のキャッシュ（NULL可）
    const int32_t* window_status,    // ウィンドウの状態 (H x W)
    int32_t H,                       // 高さ
    int32_t W,                       // 幅
    int32_t current_version,         // 現在のバージョン
    int32_t* cache_version,          // キャッシュのバージョン（入力出力）
    bool* window_changed            // 変更フラグ（入力出力）
);

// 観測データの作成（C側で高速化）
// output: 事前確保されたfloat32バッファ（サイズ = H_onpre*obs_window_size + H_cloud*obs_window_size + 40）
// ウィンドウの右端obs_window_size列を抽出し、ジョブキュー5件(8属性)と連結
void get_observation(
    const int32_t* onpre_status,     // オンプレミスウィンドウ (H_onpre x W)
    const int32_t* cloud_status,    // クラウドウィンドウ (H_cloud x W)
    const double* job_queue,        // ジョブキュー (5 x 8) 行優先
    int32_t H_onpre,
    int32_t H_cloud,
    int32_t W,
    int32_t obs_window_size,
    float* output                   // 出力バッファ
);

// リングバッファ版: 生のウィンドウ+headから直接観測を構築（Python側の配列構築を省略）
void get_observation_ringbuffer(
    const int32_t* onpre_status,
    const int32_t* cloud_status,
    const double* job_queue,
    int32_t H_onpre,
    int32_t H_cloud,
    int32_t W,
    int32_t head_onpre,
    int32_t head_cloud,
    int32_t obs_window_size,
    float* output
);

/* イベント観測: events は行優先 (n_events × 6): start,end,duration,use_cloud,start_node,job_height */
void get_observation_event(
    const double* events,
    int32_t n_events,
    int32_t current_time,
    int32_t n_window,
    double norm_time,
    double norm_nodes,
    int32_t n_events_obs,
    int32_t event_features,
    int32_t job_queue_len,
    const double* job_queue,
    float* output
);

/* do_schedule ごとのイベント追記用バッファ（Python list/tuple を避ける） */
typedef struct SchedulingEventBuffer SchedulingEventBuffer;

SchedulingEventBuffer* scheduling_event_buffer_create(int32_t initial_capacity_rows);
void scheduling_event_buffer_free(SchedulingEventBuffer* b);
void scheduling_event_buffer_reset(SchedulingEventBuffer* b);
int32_t scheduling_event_buffer_count(const SchedulingEventBuffer* b);
const double* scheduling_event_buffer_data(const SchedulingEventBuffer* b);
/* 0 成功 / -1 メモリ不足 */
int scheduling_event_buffer_append6(
    SchedulingEventBuffer* b,
    double s,
    double e,
    double d,
    double uc,
    double sn,
    double jh);

#ifdef __cplusplus
}
#endif

#endif // SCHEDULING_ENV_CORE_H

