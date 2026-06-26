#include "scheduling_env_core.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// キャッシュの構築
WindowCache* build_cache(
    const int32_t* window_status,
    int32_t H,
    int32_t W
) {
    WindowCache* cache = (WindowCache*)malloc(sizeof(WindowCache));
    if (!cache) return NULL;
    
    cache->H = H;
    cache->W = W;
    cache->version = 0;
    
    // free_per_colの計算
    cache->free_per_col = (int32_t*)malloc(W * sizeof(int32_t));
    if (!cache->free_per_col) {
        free(cache);
        return NULL;
    }
    
    for (int32_t col = 0; col < W; col++) {
        int32_t free_count = 0;
        for (int32_t row = 0; row < H; row++) {
            if (window_status[row * W + col] == 0) {
                free_count++;
            }
        }
        cache->free_per_col[col] = free_count;
    }
    
    // occの計算
    cache->occ = (int32_t*)malloc(H * W * sizeof(int32_t));
    if (!cache->occ) {
        free(cache->free_per_col);
        free(cache);
        return NULL;
    }
    
    for (int32_t i = 0; i < H * W; i++) {
        cache->occ[i] = (window_status[i] != 0) ? 1 : 0;
    }
    
    // prefix_sumの計算（2D累積和）
    cache->prefix_sum = (int32_t*)calloc((H + 1) * (W + 1), sizeof(int32_t));
    if (!cache->prefix_sum) {
        free(cache->occ);
        free(cache->free_per_col);
        free(cache);
        return NULL;
    }
    
    // 累積和の計算
    for (int32_t i = 1; i <= H; i++) {
        for (int32_t j = 1; j <= W; j++) {
            int32_t idx = i * (W + 1) + j;
            int32_t occ_idx = (i - 1) * W + (j - 1);
            cache->prefix_sum[idx] = cache->occ[occ_idx]
                + cache->prefix_sum[(i - 1) * (W + 1) + j]
                + cache->prefix_sum[i * (W + 1) + (j - 1)]
                - cache->prefix_sum[(i - 1) * (W + 1) + (j - 1)];
        }
    }
    
    // find用スクラッチバッファ（malloc削減）
    cache->scratch_mins = (int32_t*)malloc(W * sizeof(int32_t));
    cache->scratch_deque = (int32_t*)malloc(W * sizeof(int32_t));
    if (!cache->scratch_mins || !cache->scratch_deque) {
        if (cache->scratch_mins) free(cache->scratch_mins);
        if (cache->scratch_deque) free(cache->scratch_deque);
        free(cache->prefix_sum);
        free(cache->occ);
        free(cache->free_per_col);
        free(cache);
        return NULL;
    }
    
    // free_nodes_listの構築（サイズ情報を含む）
    cache->free_nodes_list = (FreeNodesList*)malloc(W * sizeof(FreeNodesList));
    if (!cache->free_nodes_list) {
        free(cache->scratch_mins);
        free(cache->scratch_deque);
        free(cache->prefix_sum);
        free(cache->occ);
        free(cache->free_per_col);
        free(cache);
        return NULL;
    }
    
    for (int32_t col = 0; col < W; col++) {
        // まず空きノード数をカウント
        int32_t free_count = cache->free_per_col[col];
        cache->free_nodes_list[col].size = free_count;
        cache->free_nodes_list[col].nodes = (int32_t*)malloc(free_count * sizeof(int32_t));
        if (!cache->free_nodes_list[col].nodes) {
            // メモリ不足の場合、既に確保したものを解放
            for (int32_t c = 0; c < col; c++) {
                free(cache->free_nodes_list[c].nodes);
            }
            free(cache->free_nodes_list);
            free(cache->scratch_mins);
            free(cache->scratch_deque);
            free(cache->prefix_sum);
            free(cache->occ);
            free(cache->free_per_col);
            free(cache);
            return NULL;
        }
        
        // 空きノードのインデックスを収集（既存のPython実装と完全に同じ方法）
        int32_t idx = 0;
        for (int32_t row = 0; row < H; row++) {
            if (window_status[row * W + col] == 0) {
                cache->free_nodes_list[col].nodes[idx++] = row;
            }
        }
    }
    
    return cache;
}

// キャッシュの解放
void free_cache(WindowCache* cache) {
    if (!cache) return;
    
    if (cache->scratch_mins) free(cache->scratch_mins);
    if (cache->scratch_deque) free(cache->scratch_deque);
    if (cache->free_per_col) free(cache->free_per_col);
    if (cache->prefix_sum) free(cache->prefix_sum);
    if (cache->occ) free(cache->occ);
    
    if (cache->free_nodes_list) {
        for (int32_t i = 0; i < cache->W; i++) {
            if (cache->free_nodes_list[i].nodes) free(cache->free_nodes_list[i].nodes);
        }
        free(cache->free_nodes_list);
    }
    
    free(cache);
}

// スライディングウィンドウの最小値計算（最適化版: O(n)、スクラッチバッファ使用）
static void sliding_window_min(
    const int32_t* arr,
    int32_t n,
    int32_t k,
    int32_t* mins,
    int32_t* deque
) {
    if (k <= 0 || n < k || !mins || !deque) {
        return;
    }
    
    int32_t front = 0, back = 0;
    
    // 最初のウィンドウを処理
    for (int32_t i = 0; i < k; i++) {
        // デックの後ろから、現在の要素より大きい要素を削除
        while (back > front && arr[deque[back - 1]] >= arr[i]) {
            back--;
        }
        deque[back++] = i;
    }
    
    // 最初の最小値
    mins[0] = arr[deque[front]];
    
    // 残りの要素を処理
    for (int32_t i = k; i < n; i++) {
        // ウィンドウから外れた要素を削除
        while (front < back && deque[front] <= i - k) {
            front++;
        }
        
        // デックの後ろから、現在の要素より大きい要素を削除
        while (back > front && arr[deque[back - 1]] >= arr[i]) {
            back--;
        }
        deque[back++] = i;
        
        // 最小値を記録
        mins[i - k + 1] = arr[deque[front]];
    }
}

// 割り当て位置の探索
AllocationResult find_allocation_position(
    const WindowCache* cache,
    int32_t job_width,
    int32_t job_height,
    int32_t when_submitted,
    int32_t current_time,
    bool continuous_only
) {
    AllocationResult result;
    result.found = false;
    result.waiting_time = INFINITY;
    
    if (!cache || job_width <= 0 || job_height <= 0) {
        return result;
    }
    
    int32_t H = cache->H;
    int32_t W = cache->W;
    
    // ジョブサイズが大きすぎる場合
    if (job_width > W || job_height > H) {
        return result;
    }
    
    int32_t k = job_width;
    int32_t need = job_height;
    
    // スライディングウィンドウの最小値計算（スクラッチバッファ使用、malloc不要）
    int32_t* mins = cache->scratch_mins;
    int32_t* deque = cache->scratch_deque;
    if (!mins || !deque) {
        return result;
    }
    
    sliding_window_min(cache->free_per_col, W, k, mins, deque);
    
    // 現在列 (a=0) のみ。未来列への先取り予約はしない
    {
        int32_t a = 0;
        if (mins[a] < need) {
            return result;
        }
        
        int32_t a2 = a + k;
        int32_t max_i = H - job_height + 1;
        
        // 連続割り当ての探索（既存のPython実装と完全に同じ順序）
        for (int32_t i = 0; i < max_i; i++) {
            int32_t i2 = i + job_height;
            int32_t ps_idx1 = i2 * (W + 1) + a2;
            int32_t ps_idx2 = i * (W + 1) + a2;
            int32_t ps_idx3 = i2 * (W + 1) + a;
            int32_t ps_idx4 = i * (W + 1) + a;
            
            int32_t occ_sum = cache->prefix_sum[ps_idx1]
                - cache->prefix_sum[ps_idx2]
                - cache->prefix_sum[ps_idx3]
                + cache->prefix_sum[ps_idx4];
            
            if (occ_sum == 0) {
                // 位置が見つかった（連続割り当て）
                result.found = true;
                result.position.is_distributed = false;
                result.position.pos.continuous.i = i;
                result.position.pos.continuous.a = a;
                result.waiting_time = (double)(current_time + a - when_submitted);
                return result;
            }
        }
        
        // クラウドは連続矩形のみ。オンプレの分散割当は開始列で固定したノード集合を全期間使用
        if (continuous_only) {
            return result;
        }

        if (cache->free_nodes_list[a].size < need) {
            return result;
        }

        int32_t* fixed_nodes = (int32_t*)malloc((size_t)need * sizeof(int32_t));
        if (!fixed_nodes) {
            return result;
        }
        for (int32_t j = 0; j < need; j++) {
            fixed_nodes[j] = cache->free_nodes_list[a].nodes[j];
        }

        bool ok = true;
        for (int32_t col_offset = 0; col_offset < k; col_offset++) {
            int32_t col = a + col_offset;
            for (int32_t j = 0; j < need; j++) {
                int32_t node = fixed_nodes[j];
                if (cache->occ[node * W + col] != 0) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                break;
            }
        }

        if (ok) {
            result.found = true;
            result.position.is_distributed = true;
            result.position.pos.distributed.i = 0;
            result.position.pos.distributed.a = a;
            result.position.pos.distributed.allocation_size = k * need;
            result.position.pos.distributed.node_allocation =
                (int32_t*)malloc((size_t)(k * need) * sizeof(int32_t));

            if (!result.position.pos.distributed.node_allocation) {
                free(fixed_nodes);
                result.found = false;
                return result;
            }

            int32_t idx = 0;
            for (int32_t col_offset = 0; col_offset < k; col_offset++) {
                for (int32_t j = 0; j < need; j++) {
                    result.position.pos.distributed.node_allocation[idx++] = fixed_nodes[j];
                }
            }
            free(fixed_nodes);

            result.waiting_time = (double)(current_time + a - when_submitted);
            return result;
        }
        free(fixed_nodes);
    }
    
    return result;
}

// 時間遷移（スライドウィンドウ、最適化版）
void time_transition(
    int32_t* window_status,
    int32_t* window_job_id,
    int32_t H,
    int32_t W,
    bool slide
) {
    if (!slide || !window_status || !window_job_id) {
        return;
    }
    
    // 左シフト（各列を1つ左に移動、最適化版）
    // 各行ごとにmemmoveを使用してメモリコピーを最適化
    const int32_t shift_size = (W - 1) * sizeof(int32_t);
    
    for (int32_t i = 0; i < H; i++) {
        int32_t* status_row = window_status + i * W;
        int32_t* job_id_row = window_job_id + i * W;
        
        // memmoveを使用して左シフト（オーバーラップを考慮）
        memmove(status_row, status_row + 1, shift_size);
        memmove(job_id_row, job_id_row + 1, shift_size);
        
        // 最後の列をクリア
        status_row[W - 1] = 0;
        job_id_row[W - 1] = -1;
    }
}

// リングバッファ版: キャッシュ構築（論理列0=物理列head）
WindowCache* build_cache_from_ringbuffer(
    const int32_t* window_status,
    int32_t H,
    int32_t W,
    int32_t head
) {
    if (!window_status || head < 0 || head >= W) {
        return NULL;
    }
    WindowCache* cache = (WindowCache*)malloc(sizeof(WindowCache));
    if (!cache) return NULL;
    
    cache->H = H;
    cache->W = W;
    cache->version = 0;
    
    cache->scratch_mins = (int32_t*)malloc(W * sizeof(int32_t));
    cache->scratch_deque = (int32_t*)malloc(W * sizeof(int32_t));
    if (!cache->scratch_mins || !cache->scratch_deque) {
        if (cache->scratch_mins) free(cache->scratch_mins);
        if (cache->scratch_deque) free(cache->scratch_deque);
        free(cache);
        return NULL;
    }
    
    cache->free_per_col = (int32_t*)malloc(W * sizeof(int32_t));
    if (!cache->free_per_col) {
        free(cache->scratch_mins);
        free(cache->scratch_deque);
        free(cache);
        return NULL;
    }
    
    for (int32_t log_col = 0; log_col < W; log_col++) {
        int32_t phys_col = (head + log_col) % W;
        int32_t free_count = 0;
        for (int32_t row = 0; row < H; row++) {
            if (window_status[row * W + phys_col] == 0) {
                free_count++;
            }
        }
        cache->free_per_col[log_col] = free_count;
    }
    
    cache->occ = (int32_t*)malloc(H * W * sizeof(int32_t));
    if (!cache->occ) {
        free(cache->free_per_col);
        free(cache->scratch_mins);
        free(cache->scratch_deque);
        free(cache);
        return NULL;
    }
    
    for (int32_t i = 0; i < H; i++) {
        for (int32_t log_col = 0; log_col < W; log_col++) {
            int32_t phys_col = (head + log_col) % W;
            cache->occ[i * W + log_col] = (window_status[i * W + phys_col] != 0) ? 1 : 0;
        }
    }
    
    cache->prefix_sum = (int32_t*)calloc((H + 1) * (W + 1), sizeof(int32_t));
    if (!cache->prefix_sum) {
        free(cache->occ);
        free(cache->free_per_col);
        free(cache->scratch_mins);
        free(cache->scratch_deque);
        free(cache);
        return NULL;
    }
    
    for (int32_t i = 1; i <= H; i++) {
        for (int32_t j = 1; j <= W; j++) {
            int32_t idx = i * (W + 1) + j;
            int32_t occ_idx = (i - 1) * W + (j - 1);
            cache->prefix_sum[idx] = cache->occ[occ_idx]
                + cache->prefix_sum[(i - 1) * (W + 1) + j]
                + cache->prefix_sum[i * (W + 1) + (j - 1)]
                - cache->prefix_sum[(i - 1) * (W + 1) + (j - 1)];
        }
    }
    
    cache->free_nodes_list = (FreeNodesList*)malloc(W * sizeof(FreeNodesList));
    if (!cache->free_nodes_list) {
        free(cache->prefix_sum);
        free(cache->occ);
        free(cache->free_per_col);
        free(cache->scratch_mins);
        free(cache->scratch_deque);
        free(cache);
        return NULL;
    }
    
    for (int32_t log_col = 0; log_col < W; log_col++) {
        int32_t phys_col = (head + log_col) % W;
        int32_t free_count = cache->free_per_col[log_col];
        cache->free_nodes_list[log_col].size = free_count;
        cache->free_nodes_list[log_col].nodes = (int32_t*)malloc((size_t)free_count * sizeof(int32_t));
        if (!cache->free_nodes_list[log_col].nodes) {
            for (int32_t c = 0; c < log_col; c++) {
                free(cache->free_nodes_list[c].nodes);
            }
            free(cache->free_nodes_list);
            free(cache->prefix_sum);
            free(cache->occ);
            free(cache->free_per_col);
            free(cache->scratch_mins);
            free(cache->scratch_deque);
            free(cache);
            return NULL;
        }
        int32_t idx = 0;
        for (int32_t row = 0; row < H; row++) {
            if (window_status[row * W + phys_col] == 0) {
                cache->free_nodes_list[log_col].nodes[idx++] = row;
            }
        }
    }
    
    return cache;
}

// リングバッファ版: 時間遷移（列headをクリア、O(H)のみ）
int32_t time_transition_ringbuffer(
    int32_t* window_status,
    int32_t* window_job_id,
    int32_t H,
    int32_t W,
    int32_t head
) {
    if (!window_status || !window_job_id) {
        return head;
    }
    for (int32_t i = 0; i < H; i++) {
        window_status[i * W + head] = 0;
        window_job_id[i * W + head] = -1;
    }
    return (head + 1) % W;
}

// リングバッファ版: ジョブスケジュール（論理列a→物理列(head+a)%W）
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
) {
    if (!window_status || !window_job_id || !position) {
        return;
    }
    
    if (!position->is_distributed) {
        int32_t i = position->pos.continuous.i;
        int32_t a = position->pos.continuous.a;
        for (int32_t row = 0; row < job_height; row++) {
            for (int32_t col = 0; col < job_width; col++) {
                int32_t phys_col = (head + a + col) % W;
                int32_t idx = (i + row) * W + phys_col;
                window_status[idx] = 1;
                window_job_id[idx] = job_id;
            }
        }
    } else {
        int32_t a = position->pos.distributed.a;
        int32_t* node_allocation = position->pos.distributed.node_allocation;
        int32_t idx = 0;
        for (int32_t col_offset = 0; col_offset < job_width; col_offset++) {
            int32_t phys_col = (head + a + col_offset) % W;
            for (int32_t j = 0; j < job_height; j++) {
                int32_t node = node_allocation[idx++];
                int32_t pos_idx = node * W + phys_col;
                window_status[pos_idx] = 1;
                window_job_id[pos_idx] = job_id;
            }
        }
    }
}

// ジョブのスケジュール実行
void do_schedule(
    int32_t* window_status,
    int32_t* window_job_id,
    int32_t H,
    int32_t W,
    int32_t job_width,
    int32_t job_height,
    int32_t job_id,
    const Position* position
) {
    if (!window_status || !window_job_id || !position) {
        return;
    }
    
    if (!position->is_distributed) {
        // 連続割り当て
        int32_t i = position->pos.continuous.i;
        int32_t a = position->pos.continuous.a;
        
        for (int32_t row = 0; row < job_height; row++) {
            for (int32_t col = 0; col < job_width; col++) {
                int32_t idx = (i + row) * W + (a + col);
                window_status[idx] = 1;
                window_job_id[idx] = job_id;
            }
        }
    } else {
        // 分散割り当て
        int32_t a = position->pos.distributed.a;
        int32_t* node_allocation = position->pos.distributed.node_allocation;
        // allocation_sizeは使用されていない（job_width * job_heightで計算可能）
        // int32_t allocation_size = position->pos.distributed.allocation_size;
        
        int32_t idx = 0;
        for (int32_t col_offset = 0; col_offset < job_width; col_offset++) {
            int32_t col = a + col_offset;
            for (int32_t j = 0; j < job_height; j++) {
                int32_t node = node_allocation[idx++];
                int32_t pos_idx = node * W + col;
                window_status[pos_idx] = 1;
                window_job_id[pos_idx] = job_id;
            }
        }
    }
}

// ユニークなジョブIDの取得
int32_t* get_unique_job_ids(
    const int32_t* history_matrix,
    int32_t H,
    int32_t W,
    int32_t max_job_id,
    int32_t* count
) {
    *count = 0;
    
    if (!history_matrix || max_job_id <= 0) {
        return NULL;
    }
    
    // 見つかったジョブIDを記録（既存のPython実装と完全に同じ: int8を使用）
    // Numbaではnp.bool_がサポートされていない場合があるため、int8を使用
    int8_t* seen = (int8_t*)calloc(max_job_id, sizeof(int8_t));
    if (!seen) {
        return NULL;
    }
    
    int32_t* temp_ids = (int32_t*)malloc(max_job_id * sizeof(int32_t));
    if (!temp_ids) {
        free(seen);
        return NULL;
    }
    
    // ヒストリーマトリックスを走査
    for (int32_t i = 0; i < H; i++) {
        for (int32_t j = 0; j < W; j++) {
            int32_t job_id = history_matrix[i * W + j];
            // 既存のPython実装と完全に同じチェック: seen[job_id] == 0
            if (job_id >= 0 && job_id < max_job_id && seen[job_id] == 0) {
                seen[job_id] = 1;
                temp_ids[(*count)++] = job_id;
                if (*count >= max_job_id) {
                    break;
                }
            }
        }
        if (*count >= max_job_id) {
            break;
        }
    }
    
    // 結果を配列に格納
    int32_t* result = (int32_t*)malloc(*count * sizeof(int32_t));
    if (!result) {
        free(seen);
        free(temp_ids);
        return NULL;
    }
    
    memcpy(result, temp_ids, *count * sizeof(int32_t));
    
    free(seen);
    free(temp_ids);
    
    return result;
}

// makespanの計算
int32_t calculate_makespan(
    const int32_t* window_matrix,
    int32_t H,
    int32_t W
) {
    int32_t makespan = -1;
    
    if (!window_matrix) {
        return makespan;
    }
    
    // 各行で右端の有効な列を探索
    for (int32_t i = 0; i < H; i++) {
        for (int32_t j = W - 1; j >= 0; j--) {
            if (window_matrix[i * W + j] != -1) {
                if (j > makespan) {
                    makespan = j;
                }
                break;
            }
        }
    }
    
    return makespan;
}

// キャッシュの差分更新（ジョブ追加時）
void update_cache_incremental(
    WindowCache* cache,
    const int32_t* window_status,
    int32_t i_start,
    int32_t i_end,
    int32_t a_start,
    int32_t a_end
) {
    if (!cache || !window_status) {
        return;
    }
    
    int32_t H = cache->H;
    int32_t W = cache->W;
    
    // 範囲チェック
    if (i_start < 0 || i_end > H || a_start < 0 || a_end > W || 
        i_start >= i_end || a_start >= a_end) {
        return;
    }
    
    // 1. free_per_colの更新（影響を受ける列のみ）
    for (int32_t col = a_start; col < a_end; col++) {
        int32_t free_count = 0;
        for (int32_t row = 0; row < H; row++) {
            if (window_status[row * W + col] == 0) {
                free_count++;
            }
        }
        cache->free_per_col[col] = free_count;
    }
    
    // 2. occの更新（影響を受ける領域のみ）
    for (int32_t i = i_start; i < i_end; i++) {
        for (int32_t j = a_start; j < a_end; j++) {
            int32_t idx = i * W + j;
            cache->occ[idx] = (window_status[idx] != 0) ? 1 : 0;
        }
    }
    
    // 3. prefix_sumの更新（影響を受ける領域のみ）
    // 行方向の累積和を更新
    for (int32_t i = i_start + 1; i <= i_end; i++) {
        for (int32_t j = 1; j <= W; j++) {
            int32_t idx = i * (W + 1) + j;
            int32_t occ_idx = (i - 1) * W + (j - 1);
            cache->prefix_sum[idx] = cache->occ[occ_idx]
                + cache->prefix_sum[(i - 1) * (W + 1) + j]
                + cache->prefix_sum[i * (W + 1) + (j - 1)]
                - cache->prefix_sum[(i - 1) * (W + 1) + (j - 1)];
        }
    }
    
    // 列方向の累積和を更新（影響を受ける列以降）
    for (int32_t i = i_end + 1; i <= H; i++) {
        for (int32_t j = a_start + 1; j <= W; j++) {
            int32_t idx = i * (W + 1) + j;
            int32_t occ_idx = (i - 1) * W + (j - 1);
            cache->prefix_sum[idx] = cache->occ[occ_idx]
                + cache->prefix_sum[(i - 1) * (W + 1) + j]
                + cache->prefix_sum[i * (W + 1) + (j - 1)]
                - cache->prefix_sum[(i - 1) * (W + 1) + (j - 1)];
        }
    }
    
    // 4. free_nodes_listの更新（影響を受ける列のみ）
    for (int32_t col = a_start; col < a_end; col++) {
        // 既存のリストを解放
        if (cache->free_nodes_list[col].nodes) {
            free(cache->free_nodes_list[col].nodes);
        }
        
        // 新しいリストを構築
        int32_t free_count = cache->free_per_col[col];
        cache->free_nodes_list[col].size = free_count;
        cache->free_nodes_list[col].nodes = (int32_t*)malloc(free_count * sizeof(int32_t));
        
        if (cache->free_nodes_list[col].nodes) {
            int32_t idx = 0;
            for (int32_t row = 0; row < H; row++) {
                if (window_status[row * W + col] == 0) {
                    cache->free_nodes_list[col].nodes[idx++] = row;
                }
            }
        }
    }
}

// キャッシュの差分更新（時間遷移時、最適化版）
void update_cache_time_transition(
    WindowCache* cache,
    const int32_t* window_status
) {
    if (!cache || !window_status) {
        return;
    }
    
    int32_t H = cache->H;
    int32_t W = cache->W;
    
    // 時間遷移は左にシフトするため、全列を更新する必要がある
    // 最適化: メモリ割り当てを削減し、計算を効率化
    
    // 1. free_per_colの更新（全列を更新）
    for (int32_t col = 0; col < W; col++) {
        int32_t free_count = 0;
        for (int32_t row = 0; row < H; row++) {
            if (window_status[row * W + col] == 0) {
                free_count++;
            }
        }
        cache->free_per_col[col] = free_count;
    }
    
    // 2. occの更新（全領域を更新、最適化: ループを1つに統合）
    for (int32_t i = 0; i < H * W; i++) {
        cache->occ[i] = (window_status[i] != 0) ? 1 : 0;
    }
    
    // 3. prefix_sumの更新（全領域を更新、最適化: メモリアクセスを最適化）
    for (int32_t i = 1; i <= H; i++) {
        int32_t prefix_row = i * (W + 1);
        int32_t prefix_row_prev = (i - 1) * (W + 1);
        int32_t occ_row = (i - 1) * W;
        
        for (int32_t j = 1; j <= W; j++) {
            int32_t idx = prefix_row + j;
            int32_t occ_idx = occ_row + (j - 1);
            cache->prefix_sum[idx] = cache->occ[occ_idx]
                + cache->prefix_sum[prefix_row_prev + j]
                + cache->prefix_sum[prefix_row + (j - 1)]
                - cache->prefix_sum[prefix_row_prev + (j - 1)];
        }
    }
    
    // 4. free_nodes_listの更新（全列を更新、最適化: メモリ再割り当てを削減）
    for (int32_t col = 0; col < W; col++) {
        int32_t free_count = cache->free_per_col[col];
        
        // 既存のメモリサイズをチェック（再割り当てを削減）
        if (cache->free_nodes_list[col].size != free_count) {
            // サイズが変わった場合のみ再割り当て
            if (cache->free_nodes_list[col].nodes) {
                free(cache->free_nodes_list[col].nodes);
            }
            cache->free_nodes_list[col].size = free_count;
            cache->free_nodes_list[col].nodes = (int32_t*)malloc(free_count * sizeof(int32_t));
        }
        // サイズが同じ場合は既存のメモリを再利用
        
        // 新しいリストを構築
        if (cache->free_nodes_list[col].nodes) {
            int32_t idx = 0;
            for (int32_t row = 0; row < H; row++) {
                if (window_status[row * W + col] == 0) {
                    cache->free_nodes_list[col].nodes[idx++] = row;
                }
            }
        }
    }
}

// リングバッファ版: 時間遷移時のキャッシュ差分更新（シフトのみ、ウィンドウ参照不要）
void update_cache_time_transition_ringbuffer(WindowCache* cache) {
    if (!cache) return;
    int32_t H = cache->H;
    int32_t W = cache->W;

    /* 1. free_per_col を左シフト（memmove） */
    memmove(cache->free_per_col, cache->free_per_col + 1, (size_t)(W - 1) * sizeof(int32_t));
    cache->free_per_col[W - 1] = H;

    /* 2. occ を左シフト（行ごとにmemmove） */
    for (int32_t i = 0; i < H; i++) {
        int32_t* row = cache->occ + i * W;
        memmove(row, row + 1, (size_t)(W - 1) * sizeof(int32_t));
        row[W - 1] = 0;
    }

    /* 3. prefix_sum を差分更新（左シフト: new[i][j]=old[i][j+1]-col0 for j<W, new[i][W]=old[i][W]-col0） */
    for (int32_t i = 1; i <= H; i++) {
        int32_t* prow = cache->prefix_sum + i * (W + 1);
        int32_t col0 = prow[1];
        int32_t last = prow[W];
        for (int32_t j = W - 1; j >= 1; j--) {
            prow[j] = prow[j + 1] - col0;
        }
        prow[W] = last - col0;
    }

    /* 4. free_nodes_list を左シフト、最終列は全行 */
    if (cache->free_nodes_list[0].nodes) {
        free(cache->free_nodes_list[0].nodes);
        cache->free_nodes_list[0].nodes = NULL;
    }
    for (int32_t col = 0; col < W - 1; col++) {
        cache->free_nodes_list[col].nodes = cache->free_nodes_list[col + 1].nodes;
        cache->free_nodes_list[col].size = cache->free_nodes_list[col + 1].size;
        cache->free_nodes_list[col + 1].nodes = NULL;
    }
    cache->free_nodes_list[W - 1].size = H;
    cache->free_nodes_list[W - 1].nodes = (int32_t*)malloc((size_t)H * sizeof(int32_t));
    if (cache->free_nodes_list[W - 1].nodes) {
        for (int32_t row = 0; row < H; row++) {
            cache->free_nodes_list[W - 1].nodes[row] = row;
        }
    }
}

// リングバッファ版: ジョブ追加時のキャッシュ差分更新（論理列→物理列マッピング）
void update_cache_incremental_ringbuffer(
    WindowCache* cache,
    const int32_t* window_status,
    int32_t i_start,
    int32_t i_end,
    int32_t a_start,
    int32_t a_end,
    int32_t head
) {
    if (!cache || !window_status) return;
    int32_t H = cache->H;
    int32_t W = cache->W;

    if (i_start < 0 || i_end > H || a_start < 0 || a_end > W ||
        i_start >= i_end || a_start >= a_end) {
        return;
    }

    /* 1. free_per_col の更新（論理列 a_start..a_end-1） */
    for (int32_t log_col = a_start; log_col < a_end; log_col++) {
        int32_t phys_col = (head + log_col) % W;
        int32_t free_count = 0;
        for (int32_t row = 0; row < H; row++) {
            if (window_status[row * W + phys_col] == 0) {
                free_count++;
            }
        }
        cache->free_per_col[log_col] = free_count;
    }

    /* 2. occ の更新 */
    for (int32_t i = i_start; i < i_end; i++) {
        for (int32_t log_col = a_start; log_col < a_end; log_col++) {
            int32_t phys_col = (head + log_col) % W;
            int32_t idx = i * W + log_col;
            cache->occ[idx] = (window_status[i * W + phys_col] != 0) ? 1 : 0;
        }
    }

    /* 3. prefix_sum の更新 */
    for (int32_t i = i_start + 1; i <= i_end; i++) {
        for (int32_t j = 1; j <= W; j++) {
            int32_t idx = i * (W + 1) + j;
            int32_t occ_idx = (i - 1) * W + (j - 1);
            cache->prefix_sum[idx] = cache->occ[occ_idx]
                + cache->prefix_sum[(i - 1) * (W + 1) + j]
                + cache->prefix_sum[i * (W + 1) + (j - 1)]
                - cache->prefix_sum[(i - 1) * (W + 1) + (j - 1)];
        }
    }
    for (int32_t i = i_end + 1; i <= H; i++) {
        for (int32_t j = a_start + 1; j <= W; j++) {
            int32_t idx = i * (W + 1) + j;
            int32_t occ_idx = (i - 1) * W + (j - 1);
            cache->prefix_sum[idx] = cache->occ[occ_idx]
                + cache->prefix_sum[(i - 1) * (W + 1) + j]
                + cache->prefix_sum[i * (W + 1) + (j - 1)]
                - cache->prefix_sum[(i - 1) * (W + 1) + (j - 1)];
        }
    }

    /* 4. free_nodes_list の更新 */
    for (int32_t log_col = a_start; log_col < a_end; log_col++) {
        int32_t phys_col = (head + log_col) % W;
        if (cache->free_nodes_list[log_col].nodes) {
            free(cache->free_nodes_list[log_col].nodes);
        }
        int32_t free_count = cache->free_per_col[log_col];
        cache->free_nodes_list[log_col].size = free_count;
        cache->free_nodes_list[log_col].nodes = (int32_t*)malloc((size_t)free_count * sizeof(int32_t));
        if (cache->free_nodes_list[log_col].nodes) {
            int32_t idx = 0;
            for (int32_t row = 0; row < H; row++) {
                if (window_status[row * W + phys_col] == 0) {
                    cache->free_nodes_list[log_col].nodes[idx++] = row;
                }
            }
        }
    }
}

// キャッシュの再構築（最適化版: バージョンチェックと差分更新を含む）
WindowCache* rebuild_cache_if_needed(
    WindowCache* cache,
    const int32_t* window_status,
    int32_t H,
    int32_t W,
    int32_t current_version,
    int32_t* cache_version,
    bool* window_changed
) {
    // 初回構築またはバージョン不一致の場合は全面再構築
    if (cache == NULL || *cache_version != current_version) {
        // 既存のキャッシュを解放
        if (cache != NULL) {
            free_cache(cache);
        }
        
        // 新規キャッシュを構築
        cache = build_cache(window_status, H, W);
        if (cache) {
            cache->version = current_version;
            *cache_version = current_version;
            *window_changed = false;  // フラグをリセット
        }
        return cache;
    }
    
    // 変更フラグが立っている場合は差分更新
    if (*window_changed) {
        update_cache_time_transition(cache, window_status);
        *window_changed = false;  // フラグをリセット
    }
    
    return cache;
}

// 観測データの作成（ウィンドウ右端のスライス + ジョブキュー）
void get_observation(
    const int32_t* onpre_status,
    const int32_t* cloud_status,
    const double* job_queue,
    int32_t H_onpre,
    int32_t H_cloud,
    int32_t W,
    int32_t obs_window_size,
    float* output
) {
    int32_t out_idx = 0;
    int32_t col_start = W - obs_window_size;
    if (col_start < 0) col_start = 0;

    /* オンプレミス: 右端 obs_window_size 列を float32 で出力 */
    for (int32_t row = 0; row < H_onpre; row++) {
        for (int32_t col = col_start; col < W; col++) {
            output[out_idx++] = (float)onpre_status[row * W + col];
        }
    }

    /* クラウド: 同様 */
    for (int32_t row = 0; row < H_cloud; row++) {
        for (int32_t col = col_start; col < W; col++) {
            output[out_idx++] = (float)cloud_status[row * W + col];
        }
    }

    /* ジョブキュー: 先頭5件 x 8属性 */
    for (int32_t i = 0; i < 5 * 8 && job_queue; i++) {
        output[out_idx++] = (float)job_queue[i];
    }
}

/* リングバッファ版: 論理列を物理列にマッピングして直接出力 */
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
) {
    int32_t out_idx = 0;
    int32_t col_start = W - obs_window_size;
    if (col_start < 0) col_start = 0;

    /* オンプレミス: 論理列 col_start..W-1 を物理列 (head+col)%W から取得 */
    for (int32_t row = 0; row < H_onpre; row++) {
        for (int32_t log_col = col_start; log_col < W; log_col++) {
            int32_t phys_col = (head_onpre + log_col) % W;
            output[out_idx++] = (float)onpre_status[row * W + phys_col];
        }
    }

    /* クラウド: 同様 */
    for (int32_t row = 0; row < H_cloud; row++) {
        for (int32_t log_col = col_start; log_col < W; log_col++) {
            int32_t phys_col = (head_cloud + log_col) % W;
            output[out_idx++] = (float)cloud_status[row * W + phys_col];
        }
    }

    /* ジョブキュー */
    for (int32_t i = 0; i < 5 * 8 && job_queue; i++) {
        output[out_idx++] = (float)job_queue[i];
    }
}

typedef struct {
    double s, e, d, uc, sn, jh;
} Event6Row;

static int event6_cmp_by_start(const void* a, const void* b) {
    const Event6Row* x = (const Event6Row*)a;
    const Event6Row* y = (const Event6Row*)b;
    if (x->s < y->s) return -1;
    if (x->s > y->s) return 1;
    return 0;
}

static float clip01_scaled(double value, double norm) {
    double t = value / norm;
    if (t < 0.0) return 0.f;
    if (t > 1.0) return 1.f;
    return (float)t;
}

static float clip01_window_time(double value, double window_start, double norm) {
    return clip01_scaled(value - window_start, norm);
}

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
) {
    int32_t window_start = current_time - n_window;
    if (window_start < 0) window_start = 0;

    int32_t out_events_elems = n_events_obs * event_features;

    int32_t nf = 0;
    for (int32_t i = 0; i < n_events; i++) {
        const double* row = events + (size_t)i * 6u;
        if (row[1] >= (double)window_start) {
            nf++;
        }
    }

    Event6Row* filt = NULL;
    if (nf > 0) {
        filt = (Event6Row*)malloc((size_t)nf * sizeof(Event6Row));
        if (!filt) {
            size_t total_floats = (size_t)out_events_elems + (size_t)job_queue_len;
            memset(output, 0, total_floats * sizeof(float));
            float* jq_out = output + out_events_elems;
            for (int32_t i = 0; i < job_queue_len && job_queue; i++) {
                jq_out[i] = (float)job_queue[i];
            }
            return;
        }
        int32_t k = 0;
        for (int32_t i = 0; i < n_events; i++) {
            const double* row = events + (size_t)i * 6u;
            if (row[1] >= (double)window_start) {
                filt[k].s = row[0];
                filt[k].e = row[1];
                filt[k].d = row[2];
                filt[k].uc = row[3];
                filt[k].sn = row[4];
                filt[k].jh = row[5];
                k++;
            }
        }
        qsort(filt, (size_t)nf, sizeof(Event6Row), event6_cmp_by_start);
    }

    size_t total_floats = (size_t)out_events_elems + (size_t)job_queue_len;
    memset(output, 0, total_floats * sizeof(float));

    int32_t n_take = nf;
    if (n_take > n_events_obs) {
        n_take = n_events_obs;
    }
    int32_t take_start = nf - n_take;
    if (take_start < 0) {
        take_start = 0;
    }
    for (int32_t i = 0; i < n_take; i++) {
        const Event6Row* ev = &filt[take_start + i];
        int32_t base = i * event_features;
        output[base + 0] = clip01_window_time(ev->s, (double)window_start, norm_time);
        output[base + 1] = clip01_window_time(ev->e, (double)window_start, norm_time);
        output[base + 2] = clip01_scaled(ev->d, norm_time);
        output[base + 3] = (float)ev->uc;
        output[base + 4] = clip01_scaled(ev->sn, norm_nodes);
        output[base + 5] = clip01_scaled(ev->jh, norm_nodes);
    }
    if (filt) {
        free(filt);
    }

    float* jq_out = output + out_events_elems;
    for (int32_t i = 0; i < job_queue_len && job_queue; i++) {
        jq_out[i] = (float)job_queue[i];
    }
}

struct SchedulingEventBuffer {
    double* storage;
    int32_t capacity;
    int32_t count;
};

SchedulingEventBuffer* scheduling_event_buffer_create(int32_t initial_capacity_rows) {
    if (initial_capacity_rows < 16) {
        initial_capacity_rows = 16;
    }
    SchedulingEventBuffer* b = (SchedulingEventBuffer*)malloc(sizeof(SchedulingEventBuffer));
    if (!b) {
        return NULL;
    }
    b->storage = (double*)malloc((size_t)initial_capacity_rows * 6u * sizeof(double));
    if (!b->storage) {
        free(b);
        return NULL;
    }
    b->capacity = initial_capacity_rows;
    b->count = 0;
    return b;
}

void scheduling_event_buffer_free(SchedulingEventBuffer* b) {
    if (!b) {
        return;
    }
    free(b->storage);
    free(b);
}

void scheduling_event_buffer_reset(SchedulingEventBuffer* b) {
    if (!b) {
        return;
    }
    b->count = 0;
}

int32_t scheduling_event_buffer_count(const SchedulingEventBuffer* b) {
    return b ? b->count : 0;
}

const double* scheduling_event_buffer_data(const SchedulingEventBuffer* b) {
    return b ? b->storage : NULL;
}

static int ensure_event_buffer_rows(SchedulingEventBuffer* b, int32_t need_rows) {
    if (need_rows <= b->capacity) {
        return 0;
    }
    int32_t newcap = b->capacity;
    while (newcap < need_rows) {
        int32_t nxt = newcap * 2;
        if (nxt <= newcap) {
            return -1;
        }
        newcap = nxt;
    }
    double* p = (double*)realloc(b->storage, (size_t)newcap * 6u * sizeof(double));
    if (!p) {
        return -1;
    }
    b->storage = p;
    b->capacity = newcap;
    return 0;
}

int scheduling_event_buffer_append6(
    SchedulingEventBuffer* b,
    double s,
    double e,
    double d,
    double uc,
    double sn,
    double jh
) {
    if (!b) {
        return -1;
    }
    if (ensure_event_buffer_rows(b, b->count + 1) != 0) {
        return -1;
    }
    double* row = b->storage + (size_t)b->count * 6u;
    row[0] = s;
    row[1] = e;
    row[2] = d;
    row[3] = uc;
    row[4] = sn;
    row[5] = jh;
    b->count++;
    return 0;
}

/* ---- event-native sweep 配置探索 (C実装) ---- */

/* (key, index) ペアの昇順比較。key 同値は index 昇順で決定的に(安定sortと等価)。 */
typedef struct { int64_t key; int32_t idx; } EvKIdx;
static int evkidx_cmp(const void* a, const void* b) {
    const EvKIdx* x = (const EvKIdx*)a;
    const EvKIdx* y = (const EvKIdx*)b;
    if (x->key < y->key) return -1;
    if (x->key > y->key) return 1;
    if (x->idx < y->idx) return -1;
    if (x->idx > y->idx) return 1;
    return 0;
}
static int evi64_cmp(const void* a, const void* b) {
    int64_t x = *(const int64_t*)a;
    int64_t y = *(const int64_t*)b;
    if (x < y) return -1;
    if (x > y) return 1;
    return 0;
}

void event_sweep_alloc(
    const int64_t* starts,
    const int64_t* ends,
    const int32_t* nodes_flat,
    const int32_t* node_off,
    int32_t n,
    int32_t width,
    int32_t height,
    int32_t n_nodes,
    bool continuous_only,
    int64_t arrival,
    int64_t* out_start,
    int32_t* out_is_contiguous,
    int32_t* out_nodes,
    int32_t* out_count
) {
    int64_t* cand = (int64_t*)malloc((size_t)(n + 1) * sizeof(int64_t));
    EvKIdx* ks = (EvKIdx*)malloc((size_t)(n > 0 ? n : 1) * sizeof(EvKIdx));
    EvKIdx* ke = (EvKIdx*)malloc((size_t)(n > 0 ? n : 1) * sizeof(EvKIdx));
    int32_t* count = (int32_t*)calloc((size_t)(n_nodes > 0 ? n_nodes : 1), sizeof(int32_t));
    unsigned char* counted = (unsigned char*)calloc((size_t)(n > 0 ? n : 1), sizeof(unsigned char));

    int32_t i;
    /* order_start / order_end: start / end 昇順のイベント index 列。 */
    for (i = 0; i < n; i++) {
        ks[i].key = starts[i]; ks[i].idx = i;
        ke[i].key = ends[i];   ke[i].idx = i;
    }
    qsort(ks, (size_t)n, sizeof(EvKIdx), evkidx_cmp);
    qsort(ke, (size_t)n, sizeof(EvKIdx), evkidx_cmp);

    /* 候補時刻 = {arrival} ∪ {ends[e] >= arrival} を昇順ユニーク化。 */
    int32_t ncand = 0;
    cand[ncand++] = arrival;
    for (i = 0; i < n; i++) {
        if (ends[i] >= arrival) cand[ncand++] = ends[i];
    }
    qsort(cand, (size_t)ncand, sizeof(int64_t), evi64_cmp);
    int32_t m = 0;
    for (i = 0; i < ncand; i++) {
        if (m == 0 || cand[i] != cand[m - 1]) cand[m++] = cand[i];
    }
    ncand = m;

    int32_t i_en = 0, i_ex = 0, free_count = n_nodes;
    int32_t ci;
    for (ci = 0; ci < ncand; ci++) {
        int64_t start = cand[ci];
        int64_t win_end = start + (int64_t)width;
        /* ENTER: start_e < win_end かつ end_e > start のイベントを占有に加える(単調前進)。 */
        while (i_en < n && starts[ks[i_en].idx] < win_end) {
            int32_t idx = ks[i_en].idx;
            if (ends[idx] > start) {
                int32_t k;
                for (k = node_off[idx]; k < node_off[idx + 1]; k++) {
                    int32_t node = nodes_flat[k];
                    if (count[node] == 0) free_count--;
                    count[node]++;
                }
                counted[idx] = 1;
            }
            i_en++;
        }
        /* EXIT: end_e <= start のイベントを占有から外す(単調前進)。 */
        while (i_ex < n && ends[ke[i_ex].idx] <= start) {
            int32_t idx = ke[i_ex].idx;
            if (counted[idx]) {
                int32_t k;
                for (k = node_off[idx]; k < node_off[idx + 1]; k++) {
                    int32_t node = nodes_flat[k];
                    count[node]--;
                    if (count[node] == 0) free_count++;
                }
                counted[idx] = 0;
            }
            i_ex++;
        }
        /* pick: free<height は即不成立。最低位の height 連続 free run を最優先、
         * 無ければ continuous_only は不成立、そうでなければ最低位 height 個の free ノード。 */
        if (free_count >= height) {
            int32_t run_start = -1, found = -1, node;
            for (node = 0; node < n_nodes; node++) {
                if (count[node] != 0) { run_start = -1; continue; }
                if (run_start < 0) run_start = node;
                if (node - run_start + 1 >= height) { found = run_start; break; }
            }
            if (found >= 0) {
                int32_t j;
                for (j = 0; j < height; j++) out_nodes[j] = found + j;
                *out_count = height; *out_is_contiguous = 1; *out_start = start;
                goto cleanup;
            }
            if (!continuous_only) {
                int32_t cnt = 0, j, is_contig = 1;
                for (node = 0; node < n_nodes && cnt < height; node++) {
                    if (count[node] == 0) out_nodes[cnt++] = node;
                }
                for (j = 1; j < height; j++) {
                    if (out_nodes[j] != out_nodes[j - 1] + 1) { is_contig = 0; break; }
                }
                *out_count = height; *out_is_contiguous = is_contig; *out_start = start;
                goto cleanup;
            }
            /* continuous_only かつ連続runなし → 次候補へ */
        }
    }

    /* フォールバック: 全候補不成立 → 最大終了時刻に range(height) を連続割当。 */
    {
        int64_t maxend = arrival;
        int32_t j;
        for (i = 0; i < n; i++) if (ends[i] > maxend) maxend = ends[i];
        for (j = 0; j < height; j++) out_nodes[j] = j;
        *out_count = height; *out_is_contiguous = 1; *out_start = maxend;
    }

cleanup:
    free(cand);
    free(ks);
    free(ke);
    free(count);
    free(counted);
}

#ifdef __cplusplus
}
#endif
