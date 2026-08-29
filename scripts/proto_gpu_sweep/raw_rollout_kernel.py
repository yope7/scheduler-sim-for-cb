"""raw_rollout_kernel — 1スレッド=1エピソードの GPU 完結 rollout カーネル(numba CUDA)。

設計書: scripts/proto_gpu_sweep/raw_rollout_design.md
正確性の基準: src/envs/scheduling_variants/event_native_env.py の
  step / _find_event_allocation / _find_event_allocation_sweep / _pick_free_from_count /
  _record_event_schedule / _compute_event_cost

初版スコープ: 行動列は事前生成して渡す(state非依存で完全等価)。defer(action=2)は対象外。
obs構築はスコープ外。1スレッドが T ジョブ分の配置探索(候補時刻sweep)を全部行う。

データ構造(実装上の設計判断: 設計書の「combined ev_*配列+is_cloudフラグ」ではなく、
オンプレ/クラウドを完全に別配列(ev_on_*, ev_cl_*)に分離した。理由: 参照実装
(_resource_events={False:[],True:[]})が資源ごとに独立したイベント列を持つのと対応が
直接的で、GPUカーネル内で "同資源のイベントだけを見る" フィルタ処理(is_cloud判定)を
逐一挟まずに済む。numba は dtype/ndim が一致すれば if/else 分岐でも配列変数のエイリアス
(ev_start_r = ev_cl_start if use_cloud else ev_on_start 相当)を許すため、コード量は
combined 版と同程度に保てる。数学的等価性は設計書と同一(候補列・占有判定・pick規則は不変)。

候補列: order_end_on/order_end_cl (end昇順に挿入維持) を持ち、各 find_alloc 呼び出しでは
「{arrival} ∪ {end>=arrivalのend}」を 二分探索+末尾までの線形走査 で昇順に辿る。
CPU参照コードのコメント(_find_event_allocation 297-312行)が示す通り、sweepループが動的に
生成する next_end は常にこの初期候補集合の要素なので、静的にこの集合を辿るだけで十分
(この等価性は本番コードの _FAST_ENV_SWEEP 系のコメントで既に検証済みの事実)。

占有カウントは増分sweep(v2): リファレンス _find_event_allocation_sweep(435-490行)と同一
構造で、order_start/order_end の2順序配列を候補間で単調前進する。
  ENTER: start_e < win_end のイベントを取り込み(end_e > start の生存分のみ計数、counted立て)
  EXIT:  end_e <= start のイベントを除去(counted が立っている分のみ減算)
counted フラグが二重計上を防ぐ(リファレンスの counted リストと同一)。1クエリ(1ジョブ)を
終えるとき、ENTERが訪れた範囲(order_start[0:i_en])の counted イベントを減算して count を
全ゼロへ戻す(ここだけ O(E) だがクエリごと1回)。free_count(count==0のノード数)も
リファレンス同様に増分維持し、pick 冒頭の free_count < height 早期棄却を再現する。
pick は _pick_free_from_count と同一規則: 連続run昇順最初(あればrun先頭からheight個)、
なければ onprem は free先頭height個(断片化, 断片数>K で ovf)、cloud は失敗して次候補へ。
pick の断片リストはグローバルメモリのスクラッチ (B,K) に置く(cuda.local.array(K) は
K=128 でローカルメモリ溢れ=レジスタ退避を起こし全体を約10倍遅化させた実測がある)。
K は配列 shape から取る実行時値になったのでカーネルは1本(コンパイル1回)。

prune(償却間引き): リファレンス _find_event_allocation 257-265行と同一。配置クエリの冒頭で
資源別イベント数が prune_at(初期64)以上なら end >= suffix_min_arrival[現ジョブindex] の
イベントだけ前詰めで残し、prune_at = max(64, 2*残数+32) に更新。間引かれるイベントは
end < thr <= 以後の全クエリ start+1 なので占有判定に二度と寄与せず、候補時刻としても
end==arrival の縮退ケースは {arrival} 自身で常に代替される=結果不変(リファレンスで
検証済みの同じ論理)。suffix_min_arrival はリファレンス同様 float の submit 列で計算
(int切り捨て前の値。比較 end >= thr も同じ float 比較)。
"""
from __future__ import annotations

import numpy as np
from numba import cuda, int64

# ---------------------------------------------------------------------------
# カーネル本体。K(断片上限)は pick スクラッチ配列の shape から取る実行時値なので
# カーネルは1本だけビルドすればよい。
# ---------------------------------------------------------------------------
_KERNEL = None


def _build_kernel():
    @cuda.jit
    def _kernel(
        jobs, actions, suffix_min, B, T, n_on, n_cl, E_MAX,
        ev_on_start, ev_on_end, ev_on_nfrag, ev_on_frag_lo, ev_on_frag_hi,
        order_end_on, order_start_on, counted_on,
        ev_cl_start, ev_cl_end, ev_cl_nfrag, ev_cl_frag_lo, ev_cl_frag_hi,
        order_end_cl, order_start_cl, counted_cl,
        count_on, count_cl, remap, pick_lo, pick_hi,
        start_times, waits, costs, ovf_flag, peak_ev_on, peak_ev_cl,
    ):
        e = cuda.grid(1)
        if e >= B:
            return
        ovf_flag[e] = 0
        peak_ev_on[e] = 0
        peak_ev_cl[e] = 0
        K = pick_lo.shape[1]

        # count はエピソード開始時に一度だけゼロ化(以後はクエリ末尾の巻き戻しでゼロ不変)。
        for nd in range(n_on):
            count_on[e, nd] = 0
        for nd in range(n_cl):
            count_cl[e, nd] = 0

        # 資源別イベント数(=次に書き込む配列の空きスロット index でもある)。
        n_ev_on = 0
        n_ev_cl = 0
        # 償却 prune の発火閾値(リファレンス _evt_prune_at={False:64,True:64} と同一)。
        prune_at_on = 64
        prune_at_cl = 64

        for j in range(T):
            arrival = int64(jobs[j, 0])
            width = int64(jobs[j, 1])
            height = int64(jobs[j, 2])
            use_cloud = actions[e, j] != 0

            # --- 資源別の配列・件数をエイリアス(dtype/ndim が一致するので numba が型統一できる) ---
            if use_cloud:
                n_ev_r = n_ev_cl
                prune_at_r = prune_at_cl
                ev_start_r = ev_cl_start
                ev_end_r = ev_cl_end
                ev_nfrag_r = ev_cl_nfrag
                ev_frag_lo_r = ev_cl_frag_lo
                ev_frag_hi_r = ev_cl_frag_hi
                order_end_r = order_end_cl
                order_start_r = order_start_cl
                counted_r = counted_cl
                count_r = count_cl
                n_nodes = n_cl
            else:
                n_ev_r = n_ev_on
                prune_at_r = prune_at_on
                ev_start_r = ev_on_start
                ev_end_r = ev_on_end
                ev_nfrag_r = ev_on_nfrag
                ev_frag_lo_r = ev_on_frag_lo
                ev_frag_hi_r = ev_on_frag_hi
                order_end_r = order_end_on
                order_start_r = order_start_on
                counted_r = counted_on
                count_r = count_on
                n_nodes = n_on

            # --- prune(償却間引き): リファレンス257-265行と同一条件・同一結果 ---
            if n_ev_r >= prune_at_r:
                thr = suffix_min[j]
                m = 0
                for i in range(n_ev_r):
                    if ev_end_r[e, i] >= thr:
                        remap[e, i] = m
                        if m != i:
                            ev_start_r[e, m] = ev_start_r[e, i]
                            ev_end_r[e, m] = ev_end_r[e, i]
                            nf = ev_nfrag_r[e, i]
                            ev_nfrag_r[e, m] = nf
                            for f in range(nf):
                                ev_frag_lo_r[e, m, f] = ev_frag_lo_r[e, i, f]
                                ev_frag_hi_r[e, m, f] = ev_frag_hi_r[e, i, f]
                        m += 1
                    else:
                        remap[e, i] = -1
                # order_end / order_start を前詰め再構築(昇順の相対順は filter で保存される)。
                # counted はクエリ末尾で毎回全クリア済みなので移送不要(全て0)。
                w = 0
                for oi in range(n_ev_r):
                    nm = remap[e, order_end_r[e, oi]]
                    if nm >= 0:
                        order_end_r[e, w] = nm
                        w += 1
                w = 0
                for oi in range(n_ev_r):
                    nm = remap[e, order_start_r[e, oi]]
                    if nm >= 0:
                        order_start_r[e, w] = nm
                        w += 1
                n_ev_r = m
                pa = 2 * m + 32
                if pa < 64:
                    pa = 64
                if use_cloud:
                    n_ev_cl = m
                    prune_at_cl = pa
                else:
                    n_ev_on = m
                    prune_at_on = pa

            # --- 候補列: order_end_r 中で end>=arrival となる先頭 index を二分探索 ---
            lo_idx = 0
            hi_idx = n_ev_r
            while lo_idx < hi_idx:
                mid = (lo_idx + hi_idx) // 2
                oe = order_end_r[e, mid]
                if ev_end_r[e, oe] >= arrival:
                    hi_idx = mid
                else:
                    lo_idx = mid + 1
            start_idx = lo_idx

            # --- 候補時刻を昇順に走査: cand_off=-1 は arrival 自身、以降は order_end_r 順 ---
            # 占有カウントは増分維持(リファレンス _find_event_allocation_sweep 457-479行と同一):
            # i_en/i_ex を単調前進し、counted フラグで二重計上を防ぐ。free_cnt(count==0 の
            # ノード数)も同時維持し、pick 冒頭の free_count<height 早期棄却を再現する。
            success = False
            best_start = int64(0)
            best_nfrag = 0
            cand_off = -1
            i_en = 0
            i_ex = 0
            free_cnt = n_nodes
            while True:
                if cand_off == -1:
                    cur_start = arrival
                else:
                    oi = start_idx + cand_off
                    if oi >= n_ev_r:
                        break  # 候補が尽きた(通常は起きない: 保険フォールバックへ)
                    idxe = order_end_r[e, oi]
                    cur_start = ev_end_r[e, idxe]

                win_end = cur_start + width

                # ENTER: start_e < win_end のイベントを取り込み(end_e > start の生存分のみ計数)。
                while i_en < n_ev_r:
                    ii = order_start_r[e, i_en]
                    if ev_start_r[e, ii] >= win_end:
                        break
                    if ev_end_r[e, ii] > cur_start:
                        counted_r[e, ii] = 1
                        nf = ev_nfrag_r[e, ii]
                        for f in range(nf):
                            flo = ev_frag_lo_r[e, ii, f]
                            fhi = ev_frag_hi_r[e, ii, f]
                            for nd in range(flo, fhi):
                                c = count_r[e, nd]
                                if c == 0:
                                    free_cnt -= 1
                                count_r[e, nd] = c + 1
                    i_en += 1
                # EXIT: end_e <= start のイベントを除去(counted 分のみ減算)。
                while i_ex < n_ev_r:
                    ii = order_end_r[e, i_ex]
                    if ev_end_r[e, ii] > cur_start:
                        break
                    if counted_r[e, ii] != 0:
                        counted_r[e, ii] = 0
                        nf = ev_nfrag_r[e, ii]
                        for f in range(nf):
                            flo = ev_frag_lo_r[e, ii, f]
                            fhi = ev_frag_hi_r[e, ii, f]
                            for nd in range(flo, fhi):
                                c = count_r[e, nd] - 1
                                count_r[e, nd] = c
                                if c == 0:
                                    free_cnt += 1
                    i_ex += 1

                # --- pick: _pick_free_from_count と同一規則(free_count<height は早期棄却) ---
                run_found = False
                fh_count = 0
                fh_frag_n = 0
                if free_cnt >= height:
                    run_start = -1
                    run_lo = -1
                    need_head = not use_cloud
                    fh_last_node = -2
                    for nd in range(n_nodes):
                        if count_r[e, nd] != 0:
                            run_start = -1
                            continue
                        if run_start < 0:
                            run_start = nd
                        if nd - run_start + 1 >= height:
                            run_found = True
                            run_lo = run_start
                            break
                        if need_head and fh_count < height:
                            if fh_count == 0 or nd != fh_last_node + 1:
                                if fh_frag_n >= K:
                                    ovf_flag[e] = 2  # 断片数 > K (K を上げて再実行)
                                    return
                                pick_lo[e, fh_frag_n] = nd
                                pick_hi[e, fh_frag_n] = nd + 1
                                fh_frag_n += 1
                            else:
                                pick_hi[e, fh_frag_n - 1] = nd + 1
                            fh_last_node = nd
                            fh_count += 1

                    if run_found:
                        best_start = cur_start
                        pick_lo[e, 0] = run_lo
                        pick_hi[e, 0] = run_lo + height
                        best_nfrag = 1
                        success = True
                        break
                    elif need_head and fh_count >= height:
                        best_start = cur_start
                        best_nfrag = fh_frag_n
                        success = True
                        break

                cand_off += 1

            # RESET: このクエリで ENTER が訪れた範囲(order_start[0:i_en])の counted イベントを
            # 減算し count を全ゼロへ戻す(クエリごと1回の O(E))。
            for t in range(i_en):
                ii = order_start_r[e, t]
                if counted_r[e, ii] != 0:
                    counted_r[e, ii] = 0
                    nf = ev_nfrag_r[e, ii]
                    for f in range(nf):
                        flo = ev_frag_lo_r[e, ii, f]
                        fhi = ev_frag_hi_r[e, ii, f]
                        for nd in range(flo, fhi):
                            count_r[e, nd] -= 1

            if not success:
                # 保険: すべての既存イベント終了後なら必ず空く(参照実装315-317行と同じ)。
                if n_ev_r > 0:
                    last_idx = order_end_r[e, n_ev_r - 1]
                    max_end = ev_end_r[e, last_idx]
                    if max_end < arrival:
                        max_end = arrival
                else:
                    max_end = arrival
                best_start = max_end
                best_nfrag = 1
                pick_lo[e, 0] = 0
                pick_hi[e, 0] = height

            start = best_start
            wait = start - arrival
            if use_cloud:
                cost = width * height
            else:
                cost = int64(0)

            start_times[e, j] = start
            waits[e, j] = wait
            costs[e, j] = cost

            # --- record_event: 末尾に追加 + order_end/order_start へ挿入(二分探索+シフト) ---
            idx = n_ev_r
            if idx >= E_MAX:
                ovf_flag[e] = 1  # イベント数 > E_MAX (E_MAX を上げて再実行)
                return
            ev_start_r[e, idx] = start
            end_val = start + width
            ev_end_r[e, idx] = end_val
            ev_nfrag_r[e, idx] = best_nfrag
            counted_r[e, idx] = 0
            for f in range(best_nfrag):
                ev_frag_lo_r[e, idx, f] = pick_lo[e, f]
                ev_frag_hi_r[e, idx, f] = pick_hi[e, f]

            lo2 = 0
            hi2 = n_ev_r
            while lo2 < hi2:
                mid2 = (lo2 + hi2) // 2
                if ev_end_r[e, order_end_r[e, mid2]] <= end_val:
                    lo2 = mid2 + 1
                else:
                    hi2 = mid2
            p = n_ev_r
            while p > lo2:
                order_end_r[e, p] = order_end_r[e, p - 1]
                p -= 1
            order_end_r[e, lo2] = idx

            # order_start へも二分挿入(同値は後ろ=タイ順は占有集合に影響しない)。
            lo3 = 0
            hi3 = n_ev_r
            while lo3 < hi3:
                mid3 = (lo3 + hi3) // 2
                if ev_start_r[e, order_start_r[e, mid3]] <= start:
                    lo3 = mid3 + 1
                else:
                    hi3 = mid3
            p = n_ev_r
            while p > lo3:
                order_start_r[e, p] = order_start_r[e, p - 1]
                p -= 1
            order_start_r[e, lo3] = idx

            if use_cloud:
                n_ev_cl = idx + 1
                if n_ev_cl > peak_ev_cl[e]:
                    peak_ev_cl[e] = n_ev_cl
            else:
                n_ev_on = idx + 1
                if n_ev_on > peak_ev_on[e]:
                    peak_ev_on[e] = n_ev_on

    return _kernel


def run_raw_rollout(
    jobs, actions, n_on: int, n_cl: int, e_max: int = 8192, k: int = 16, tpb: int = 32
) -> dict:
    """GPU完結rolloutを実行する。

    Args:
        jobs: (T,8) 配列。列=[submit,pt,nodes,can_cloud,user,id,waiting(-1),pad(0)]
              (JobGenerator.generate_job と同一の raw 形式。to_queue 変換前)。
        actions: (B,T) 配列。0=onprem, 1=cloud(defer非対応)。
        n_on, n_cl: オンプレ/クラウドのノード数。
        e_max: 1エピソードあたりのイベント上限(資源ごとに e_max 確保)。
              prune 実装後は「同時に生きているイベント数」の上限で足りる
              (溢れたら ovf=1。peak_ev_on/cl で実測ピークを確認できる)。
        k: ノード断片上限。
        tpb: threads_per_block。1エピソード=1スレッドで各スレッドの分岐が完全に
             バラけるため、tpb=1(1スレッド=1ブロック=warp内分岐ゼロ)が速いことがある。
             結果はどちらでも同一。

    Returns:
        dict(start_times, waits, costs: (B,T) int64,
             objectives: (B,2) float64 = (total_cost, mean_wait),
             ovf: (B,) int32 (0=正常 / 1=E_MAX溢れ / 2=断片数>K),
             peak_ev_on, peak_ev_cl: (B,) int32 = 資源別の生存イベント数ピーク)
    """
    jobs = np.ascontiguousarray(jobs, dtype=np.float64)
    actions = np.ascontiguousarray(actions, dtype=np.int8)
    T = int(jobs.shape[0])
    B = int(actions.shape[0])
    if actions.shape[1] != T:
        raise ValueError(f"actions.shape[1]={actions.shape[1]} != T={T}")

    # prune 閾値: リファレンス(reset 94-97行)と同一の float submit 列 suffix-min。
    arrivals = jobs[:, 0].astype(np.float64)
    suffix_min = np.minimum.accumulate(arrivals[::-1])[::-1].copy()

    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _build_kernel()
    kernel = _KERNEL

    d_jobs = cuda.to_device(jobs)
    d_actions = cuda.to_device(actions)
    d_suffix = cuda.to_device(suffix_min)

    d_ev_on_start = cuda.device_array((B, e_max), dtype=np.int64)
    d_ev_on_end = cuda.device_array((B, e_max), dtype=np.int64)
    d_ev_on_nfrag = cuda.device_array((B, e_max), dtype=np.int32)
    d_ev_on_frag_lo = cuda.device_array((B, e_max, k), dtype=np.int32)
    d_ev_on_frag_hi = cuda.device_array((B, e_max, k), dtype=np.int32)
    d_order_end_on = cuda.device_array((B, e_max), dtype=np.int32)
    d_order_start_on = cuda.device_array((B, e_max), dtype=np.int32)
    d_counted_on = cuda.device_array((B, e_max), dtype=np.int8)

    d_ev_cl_start = cuda.device_array((B, e_max), dtype=np.int64)
    d_ev_cl_end = cuda.device_array((B, e_max), dtype=np.int64)
    d_ev_cl_nfrag = cuda.device_array((B, e_max), dtype=np.int32)
    # クラウドは continuous_only=True で pick が連続 run しか返さない(fallback も連続)ため
    # nfrag は常に1=断片配列は幅1で厳密に足りる。(B,e_max,k) を2資源分持つと B=1024 で
    # 100GB 超になるための必須の縮小(結果不変)。
    d_ev_cl_frag_lo = cuda.device_array((B, e_max, 1), dtype=np.int32)
    d_ev_cl_frag_hi = cuda.device_array((B, e_max, 1), dtype=np.int32)
    d_order_end_cl = cuda.device_array((B, e_max), dtype=np.int32)
    d_order_start_cl = cuda.device_array((B, e_max), dtype=np.int32)
    d_counted_cl = cuda.device_array((B, e_max), dtype=np.int8)

    d_count_on = cuda.device_array((B, max(n_on, 1)), dtype=np.int32)
    d_count_cl = cuda.device_array((B, max(n_cl, 1)), dtype=np.int32)
    d_remap = cuda.device_array((B, e_max), dtype=np.int32)
    # pick の断片スクラッチ(グローバルメモリ)。cuda.local.array(K) は K=128 で
    # ローカルメモリ溢れ→全体約10倍遅化のためグローバルに置く。
    d_pick_lo = cuda.device_array((B, max(k, 1)), dtype=np.int32)
    d_pick_hi = cuda.device_array((B, max(k, 1)), dtype=np.int32)

    d_start_times = cuda.device_array((B, T), dtype=np.int64)
    d_waits = cuda.device_array((B, T), dtype=np.int64)
    d_costs = cuda.device_array((B, T), dtype=np.int64)
    d_ovf = cuda.device_array((B,), dtype=np.int32)
    d_peak_on = cuda.device_array((B,), dtype=np.int32)
    d_peak_cl = cuda.device_array((B,), dtype=np.int32)

    threads_per_block = max(1, int(tpb))
    blocks = (B + threads_per_block - 1) // threads_per_block

    kernel[blocks, threads_per_block](
        d_jobs, d_actions, d_suffix, B, T, n_on, n_cl, e_max,
        d_ev_on_start, d_ev_on_end, d_ev_on_nfrag, d_ev_on_frag_lo, d_ev_on_frag_hi,
        d_order_end_on, d_order_start_on, d_counted_on,
        d_ev_cl_start, d_ev_cl_end, d_ev_cl_nfrag, d_ev_cl_frag_lo, d_ev_cl_frag_hi,
        d_order_end_cl, d_order_start_cl, d_counted_cl,
        d_count_on, d_count_cl, d_remap, d_pick_lo, d_pick_hi,
        d_start_times, d_waits, d_costs, d_ovf, d_peak_on, d_peak_cl,
    )
    cuda.synchronize()

    start_times = d_start_times.copy_to_host()
    waits = d_waits.copy_to_host()
    costs = d_costs.copy_to_host()
    ovf = d_ovf.copy_to_host()
    peak_ev_on = d_peak_on.copy_to_host()
    peak_ev_cl = d_peak_cl.copy_to_host()

    total_cost = costs.sum(axis=1).astype(np.float64)
    mean_wait = waits.mean(axis=1).astype(np.float64)
    objectives = np.stack([total_cost, mean_wait], axis=1)

    return dict(
        start_times=start_times, waits=waits, costs=costs,
        objectives=objectives, ovf=ovf,
        peak_ev_on=peak_ev_on, peak_ev_cl=peak_ev_cl,
    )
