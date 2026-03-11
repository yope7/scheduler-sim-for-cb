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
    
    // free_nodes_listの構築（サイズ情報を含む）
    cache->free_nodes_list = (FreeNodesList*)malloc(W * sizeof(FreeNodesList));
    if (!cache->free_nodes_list) {
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
    
    if (cache->free_per_col) free(cache->free_per_col);
    if (cache->prefix_sum) free(cache->prefix_sum);
    if (cache->occ) free(cache->occ);
    
    if (cache->free_nodes_list) {
        for (int32_t i = 0; i < cache->W; i++) {
            if (cache->free_nodes_list[i].nodes) {
                free(cache->free_nodes_list[i].nodes);
            }
        }
        free(cache->free_nodes_list);
    }
    
    free(cache);
}

// スライディングウィンドウの最小値計算（最適化版: O(n)）
static void sliding_window_min(
    const int32_t* arr,
    int32_t n,
    int32_t k,
    int32_t* mins
) {
    // dequeを使ったO(n)の実装
    if (k <= 0 || n < k) {
        return;
    }
    
    // デックのインデックスを管理（配列で実装）
    int32_t* deque = (int32_t*)malloc(n * sizeof(int32_t));
    if (!deque) {
        // メモリ不足の場合はシンプルな実装にフォールバック
        for (int32_t i = 0; i <= n - k; i++) {
            int32_t min_val = INT_MAX;
            for (int32_t j = 0; j < k; j++) {
                if (arr[i + j] < min_val) {
                    min_val = arr[i + j];
                }
            }
            mins[i] = min_val;
        }
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
    
    free(deque);
}

// 割り当て位置の探索
AllocationResult find_allocation_position(
    const WindowCache* cache,
    int32_t job_width,
    int32_t job_height,
    int32_t when_submitted,
    int32_t current_time
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
    
    // スライディングウィンドウの最小値計算
    int32_t* mins = (int32_t*)malloc((W - k + 1) * sizeof(int32_t));
    if (!mins) {
        return result;
    }
    
    sliding_window_min(cache->free_per_col, W, k, mins);
    
    // First-Fit探索
    int32_t limit_a = W - k + 1;
    for (int32_t a = 0; a < limit_a; a++) {
        if (mins[a] < need) {
            continue;
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
                free(mins);
                return result;
            }
        }
        
        // 連続割り当てが見つからなかった場合、分散割り当てを探索
        // （既存のPython実装と完全に同じ順序とロジック）
        bool ok = true;
        for (int32_t col_offset = 0; col_offset < k; col_offset++) {
            int32_t col = a + col_offset;
            // 既存のPython実装と完全に同じチェック: nodes.size < job_height
            if (col >= W || cache->free_nodes_list[col].size < need) {
                ok = false;
                break;
            }
        }
        
        if (ok) {
            // 分散割り当てが可能（既存のPython実装と完全に同じ方法）
            result.found = true;
            result.position.is_distributed = true;
            result.position.pos.distributed.i = 0;
            result.position.pos.distributed.a = a;
            result.position.pos.distributed.allocation_size = k * need;
            result.position.pos.distributed.node_allocation = 
                (int32_t*)malloc(k * need * sizeof(int32_t));
            
            if (!result.position.pos.distributed.node_allocation) {
                result.found = false;
                free(mins);
                return result;
            }
            
            // ノード割り当てを構築（既存のPython実装と完全に同じ方法: nodes[:job_height]）
            int32_t idx = 0;
            for (int32_t col_offset = 0; col_offset < k; col_offset++) {
                int32_t col = a + col_offset;
                int32_t* nodes = cache->free_nodes_list[col].nodes;
                // nodes[:job_height]をコピー（既存のPython実装と完全に同じ）
                for (int32_t j = 0; j < need; j++) {
                    result.position.pos.distributed.node_allocation[idx++] = nodes[j];
                }
            }
            
            result.waiting_time = (double)(current_time + a - when_submitted);
            free(mins);
            return result;
        }
    }
    
    free(mins);
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

#ifdef __cplusplus
}
#endif

