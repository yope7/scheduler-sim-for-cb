"""raw_rollout_kernel_block — 1ブロック=1エピソードのブロック協調版 rawカーネル(numba CUDA)。

既存の raw_rollout_kernel.py(1スレッド=1エピソード)は一切変更していない。本ファイルは
「1本のエピソードを TPB スレッドで手分けして進める」版で、返り値・引数は既存と同一。

なぜ作るか(実測): 既存版は 1スレッドが 5万ステップを逐次に進めるため、グローバルメモリの
レイテンシ(400-800サイクル)を隠せず 13.7ms/step かかっていた(CPU env の 0.465ms/step の29倍)。
GPU のコアは 10,752 個あるのに B=64 では 64 個しか使っていない。1本をブロックで担当すれば
待ち時間が他スレッドの実行で埋まる。

手分けするのは「イベント×ノードの占有カウント更新」(ENTER/EXIT/RESET)と count のゼロ化で、
ここが仕事量の主。pick(連続runの探索)・prune・イベント表への挿入は本版ではスレッド0が
従来通り逐次で行う(まず主要部の効果を測るため。pick の並列化は次段)。

ビット一致の根拠:
  * count の最終値 = 加算の総和 → atomic の実行順に依らない
  * free_cnt の増減 = 「atomic の戻り値 old が 0 だった回数」の総和。同じノードに複数
    イベントが入っても old==0 は厳密に1回しか観測されない → 順序に依らない
  * ENTER/EXIT の範囲は order_start/order_end が昇順なので二分探索で求まる(逐次前進と同値)
  * pick / prune / 挿入は従来と同じ逐次コード
"""
from __future__ import annotations

import numpy as np
from numba import cuda, int32, int64

TPB = 128

# sh_state の添字
_S_NEV_ON = 0
_S_NEV_CL = 1
_S_PRUNE_ON = 2
_S_PRUNE_CL = 3
_S_IEN = 4
_S_IEX = 5
_S_SUCCESS = 6
_S_OVF = 7
_S_NFRAG = 8
_S_FREE = 9
_S_DEC = 10
_S_DONE = 11

_KERNEL_B = {}


def _build_kernel_block(parallel_pick: bool):
    """parallel_pick=True なら「連続した空きノードの探索」も手分けする。

    分担のしかた: 各スレッドが担当区間を1回走査して
      (先頭から続く空きの長さ, 末尾に続く空きの長さ, 区間内で完結する最初のrun, 区間が全部空きか)
    を出し、スレッド0が区間を順に繋いで「最初に height 個続く位置」を決める。
    区間をまたぐ run は carry で持ち回るので、逐次版と同じ位置が選ばれる。

    逐次版との差が出る唯一の場所: 逐次版は run を見つける前に断片が K 本を超えると
    ovf=2 で打ち切るが、こちらは run を先に判定するので打ち切らない(run があるなら
    断片リストは使われないので結果は正しい)。K=128 の実運用では断片がそこまで出た
    実績がないため差は出ない見込みだが、verify では ovf も突き合わせて確認する。
    """
    PARALLEL_PICK = bool(parallel_pick)

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
        e = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        nthreads = cuda.blockDim.x
        if e >= B:
            return

        sh = cuda.shared.array(16, int32)
        sh_start = cuda.shared.array(2, int64)
        # pick を手分けするときの区間ごとの中間結果(先頭連続/末尾連続/区間内run/全部空き)
        sh_pre = cuda.shared.array(TPB, int32)
        sh_suf = cuda.shared.array(TPB, int32)
        sh_best = cuda.shared.array(TPB, int32)
        sh_all = cuda.shared.array(TPB, int32)
        K = pick_lo.shape[1]

        if tid == 0:
            ovf_flag[e] = 0
            peak_ev_on[e] = 0
            peak_ev_cl[e] = 0
            sh[_S_NEV_ON] = 0
            sh[_S_NEV_CL] = 0
            sh[_S_PRUNE_ON] = 64
            sh[_S_PRUNE_CL] = 64
            sh[_S_OVF] = 0
        # count のゼロ化を手分け(既存版はここも 1スレッドで n_on 回まわしていた)
        for nd in range(tid, n_on, nthreads):
            count_on[e, nd] = 0
        for nd in range(tid, n_cl, nthreads):
            count_cl[e, nd] = 0
        cuda.syncthreads()

        for j in range(T):
            arrival = int64(jobs[j, 0])
            width = int64(jobs[j, 1])
            height = int64(jobs[j, 2])
            use_cloud = actions[e, j] != 0

            if use_cloud:
                n_ev_r = sh[_S_NEV_CL]
                prune_at_r = sh[_S_PRUNE_CL]
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
                n_ev_r = sh[_S_NEV_ON]
                prune_at_r = sh[_S_PRUNE_ON]
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

            # --- prune: 前詰めは順序に依存するのでスレッド0が従来通り行う ---
            if n_ev_r >= prune_at_r:
                if tid == 0:
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
                    pa = 2 * m + 32
                    if pa < 64:
                        pa = 64
                    if use_cloud:
                        sh[_S_NEV_CL] = m
                        sh[_S_PRUNE_CL] = pa
                    else:
                        sh[_S_NEV_ON] = m
                        sh[_S_PRUNE_ON] = pa
                cuda.syncthreads()
                if use_cloud:
                    n_ev_r = sh[_S_NEV_CL]
                else:
                    n_ev_r = sh[_S_NEV_ON]

            # --- 候補列の開始位置(全スレッドが同じ二分探索をする) ---
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

            if tid == 0:
                sh[_S_IEN] = 0
                sh[_S_IEX] = 0
                sh[_S_SUCCESS] = 0
                sh[_S_DONE] = 0
                sh[_S_FREE] = n_nodes
                sh[_S_NFRAG] = 0
                sh_start[0] = 0
            cuda.syncthreads()

            cand_off = -1
            while True:
                if cand_off == -1:
                    cur_start = arrival
                else:
                    oi = start_idx + cand_off
                    if oi >= n_ev_r:
                        if tid == 0:
                            sh[_S_DONE] = 1
                        cuda.syncthreads()
                        break
                    idxe = order_end_r[e, oi]
                    cur_start = ev_end_r[e, idxe]
                win_end = cur_start + width

                # --- ENTER: start_e < win_end の範囲を二分探索で確定し、手分けして計上 ---
                i_en = sh[_S_IEN]
                lo2 = i_en
                hi2 = n_ev_r
                while lo2 < hi2:
                    mid2 = (lo2 + hi2) // 2
                    ii2 = order_start_r[e, mid2]
                    if ev_start_r[e, ii2] >= win_end:
                        hi2 = mid2
                    else:
                        lo2 = mid2 + 1
                i_en2 = lo2
                if tid == 0:
                    sh[_S_DEC] = 0
                cuda.syncthreads()
                dec_local = 0
                for t in range(i_en + tid, i_en2, nthreads):
                    ii = order_start_r[e, t]
                    if ev_end_r[e, ii] > cur_start:
                        counted_r[e, ii] = 1
                        nf = ev_nfrag_r[e, ii]
                        for f in range(nf):
                            flo = ev_frag_lo_r[e, ii, f]
                            fhi = ev_frag_hi_r[e, ii, f]
                            for nd in range(flo, fhi):
                                old = cuda.atomic.add(count_r, (e, nd), 1)
                                if old == 0:
                                    dec_local += 1
                if dec_local != 0:
                    cuda.atomic.add(sh, _S_DEC, dec_local)
                cuda.syncthreads()
                if tid == 0:
                    sh[_S_FREE] -= sh[_S_DEC]
                    sh[_S_IEN] = i_en2
                cuda.syncthreads()

                # --- EXIT: end_e <= cur_start の範囲を二分探索で確定し、手分けして除去 ---
                i_ex = sh[_S_IEX]
                lo3 = i_ex
                hi3 = n_ev_r
                while lo3 < hi3:
                    mid3 = (lo3 + hi3) // 2
                    ii3 = order_end_r[e, mid3]
                    if ev_end_r[e, ii3] > cur_start:
                        hi3 = mid3
                    else:
                        lo3 = mid3 + 1
                i_ex2 = lo3
                if tid == 0:
                    sh[_S_DEC] = 0
                cuda.syncthreads()
                inc_local = 0
                for t in range(i_ex + tid, i_ex2, nthreads):
                    ii = order_end_r[e, t]
                    if counted_r[e, ii] != 0:
                        counted_r[e, ii] = 0
                        nf = ev_nfrag_r[e, ii]
                        for f in range(nf):
                            flo = ev_frag_lo_r[e, ii, f]
                            fhi = ev_frag_hi_r[e, ii, f]
                            for nd in range(flo, fhi):
                                old = cuda.atomic.add(count_r, (e, nd), -1)
                                if old == 1:
                                    inc_local += 1
                if inc_local != 0:
                    cuda.atomic.add(sh, _S_DEC, inc_local)
                cuda.syncthreads()
                if tid == 0:
                    sh[_S_FREE] += sh[_S_DEC]
                    sh[_S_IEX] = i_ex2
                cuda.syncthreads()

                # --- pick ---
                run_lo_par = -1
                if PARALLEL_PICK and sh[_S_FREE] >= height:
                    seg = (n_nodes + nthreads - 1) // nthreads
                    lo_nd = tid * seg
                    hi_nd = lo_nd + seg
                    if hi_nd > n_nodes:
                        hi_nd = n_nodes
                    cur0 = 0
                    pre0 = -1
                    bestp = -1
                    nd = lo_nd
                    while nd < hi_nd:
                        if count_r[e, nd] == 0:
                            cur0 += 1
                            if bestp < 0 and cur0 >= height:
                                bestp = nd - height + 1
                        else:
                            if pre0 < 0:
                                pre0 = cur0
                            cur0 = 0
                        nd += 1
                    allz = 0
                    if pre0 < 0:
                        pre0 = cur0
                        allz = 1
                    if lo_nd >= n_nodes:
                        pre0 = 0
                        cur0 = 0
                        allz = 1
                    sh_pre[tid] = pre0
                    sh_suf[tid] = cur0
                    sh_best[tid] = bestp
                    sh_all[tid] = allz
                    cuda.syncthreads()
                    if tid == 0:
                        carry = 0
                        ans = -1
                        i = 0
                        while i < nthreads:
                            lo_i = i * seg
                            if lo_i >= n_nodes:
                                break
                            hi_i = lo_i + seg
                            if hi_i > n_nodes:
                                hi_i = n_nodes
                            if carry + sh_pre[i] >= height:
                                ans = lo_i - carry
                                break
                            if sh_best[i] >= 0:
                                ans = sh_best[i]
                                break
                            if sh_all[i] != 0:
                                carry += hi_i - lo_i
                            else:
                                carry = sh_suf[i]
                            i += 1
                        sh_best[0] = ans
                    cuda.syncthreads()
                    run_lo_par = sh_best[0]

                if PARALLEL_PICK and run_lo_par >= 0:
                    # 連続runが見つかった: 断片リストは使わないのでスレッド0が結果だけ書く
                    if tid == 0:
                        sh_start[0] = cur_start
                        pick_lo[e, 0] = run_lo_par
                        pick_hi[e, 0] = run_lo_par + height
                        sh[_S_NFRAG] = 1
                        sh[_S_SUCCESS] = 1
                    cuda.syncthreads()
                elif tid == 0:
                    free_cnt = sh[_S_FREE]
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
                                        sh[_S_OVF] = 2
                                        break
                                    pick_lo[e, fh_frag_n] = nd
                                    pick_hi[e, fh_frag_n] = nd + 1
                                    fh_frag_n += 1
                                else:
                                    pick_hi[e, fh_frag_n - 1] = nd + 1
                                fh_last_node = nd
                                fh_count += 1
                        if sh[_S_OVF] == 0:
                            if run_found:
                                sh_start[0] = cur_start
                                pick_lo[e, 0] = run_lo
                                pick_hi[e, 0] = run_lo + height
                                sh[_S_NFRAG] = 1
                                sh[_S_SUCCESS] = 1
                            elif need_head and fh_count >= height:
                                sh_start[0] = cur_start
                                sh[_S_NFRAG] = fh_frag_n
                                sh[_S_SUCCESS] = 1
                cuda.syncthreads()
                if sh[_S_OVF] != 0:
                    if tid == 0:
                        ovf_flag[e] = sh[_S_OVF]
                    return
                if sh[_S_SUCCESS] != 0:
                    break
                cand_off += 1

            # --- RESET: ENTER が訪れた範囲を手分けして巻き戻す ---
            i_en_done = sh[_S_IEN]
            for t in range(tid, i_en_done, nthreads):
                ii = order_start_r[e, t]
                if counted_r[e, ii] != 0:
                    counted_r[e, ii] = 0
                    nf = ev_nfrag_r[e, ii]
                    for f in range(nf):
                        flo = ev_frag_lo_r[e, ii, f]
                        fhi = ev_frag_hi_r[e, ii, f]
                        for nd in range(flo, fhi):
                            cuda.atomic.add(count_r, (e, nd), -1)
            cuda.syncthreads()

            # --- 記録とイベント表への挿入(順序依存なのでスレッド0) ---
            if tid == 0:
                if sh[_S_SUCCESS] != 0:
                    best_start = sh_start[0]
                    best_nfrag = sh[_S_NFRAG]
                else:
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

                idx = n_ev_r
                if idx >= E_MAX:
                    sh[_S_OVF] = 1
                else:
                    ev_start_r[e, idx] = start
                    end_val = start + width
                    ev_end_r[e, idx] = end_val
                    ev_nfrag_r[e, idx] = best_nfrag
                    counted_r[e, idx] = 0
                    for f in range(best_nfrag):
                        ev_frag_lo_r[e, idx, f] = pick_lo[e, f]
                        ev_frag_hi_r[e, idx, f] = pick_hi[e, f]

                    lo4 = 0
                    hi4 = n_ev_r
                    while lo4 < hi4:
                        mid4 = (lo4 + hi4) // 2
                        if ev_end_r[e, order_end_r[e, mid4]] <= end_val:
                            lo4 = mid4 + 1
                        else:
                            hi4 = mid4
                    p = n_ev_r
                    while p > lo4:
                        order_end_r[e, p] = order_end_r[e, p - 1]
                        p -= 1
                    order_end_r[e, lo4] = idx

                    lo5 = 0
                    hi5 = n_ev_r
                    while lo5 < hi5:
                        mid5 = (lo5 + hi5) // 2
                        if ev_start_r[e, order_start_r[e, mid5]] <= start:
                            lo5 = mid5 + 1
                        else:
                            hi5 = mid5
                    p = n_ev_r
                    while p > lo5:
                        order_start_r[e, p] = order_start_r[e, p - 1]
                        p -= 1
                    order_start_r[e, lo5] = idx

                    if use_cloud:
                        sh[_S_NEV_CL] = idx + 1
                        if idx + 1 > peak_ev_cl[e]:
                            peak_ev_cl[e] = idx + 1
                    else:
                        sh[_S_NEV_ON] = idx + 1
                        if idx + 1 > peak_ev_on[e]:
                            peak_ev_on[e] = idx + 1
            cuda.syncthreads()
            if sh[_S_OVF] != 0:
                if tid == 0:
                    ovf_flag[e] = sh[_S_OVF]
                return

    return _kernel


def run_raw_rollout_block(
    jobs, actions, n_on: int, n_cl: int, e_max: int = 8192, k: int = 16, tpb: int = TPB,
    parallel_pick: bool = True,
) -> dict:
    """ブロック協調版。引数・返り値は run_raw_rollout と同一(tpb は 1本を担当するスレッド数)。

    parallel_pick=False にすると「空きノードの探索」だけ逐次に戻る(効果の切り分け用)。
    """
    jobs = np.ascontiguousarray(jobs, dtype=np.float64)
    actions = np.ascontiguousarray(actions, dtype=np.int8)
    T = int(jobs.shape[0])
    B = int(actions.shape[0])
    if actions.shape[1] != T:
        raise ValueError(f"actions.shape[1]={actions.shape[1]} != T={T}")

    arrivals = jobs[:, 0].astype(np.float64)
    suffix_min = np.minimum.accumulate(arrivals[::-1])[::-1].copy()

    key = bool(parallel_pick)
    if key not in _KERNEL_B:
        _KERNEL_B[key] = _build_kernel_block(key)
    kernel = _KERNEL_B[key]

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
    d_ev_cl_frag_lo = cuda.device_array((B, e_max, 1), dtype=np.int32)
    d_ev_cl_frag_hi = cuda.device_array((B, e_max, 1), dtype=np.int32)
    d_order_end_cl = cuda.device_array((B, e_max), dtype=np.int32)
    d_order_start_cl = cuda.device_array((B, e_max), dtype=np.int32)
    d_counted_cl = cuda.device_array((B, e_max), dtype=np.int8)

    d_count_on = cuda.device_array((B, max(n_on, 1)), dtype=np.int32)
    d_count_cl = cuda.device_array((B, max(n_cl, 1)), dtype=np.int32)
    d_remap = cuda.device_array((B, e_max), dtype=np.int32)
    d_pick_lo = cuda.device_array((B, max(k, 1)), dtype=np.int32)
    d_pick_hi = cuda.device_array((B, max(k, 1)), dtype=np.int32)

    d_start_times = cuda.device_array((B, T), dtype=np.int64)
    d_waits = cuda.device_array((B, T), dtype=np.int64)
    d_costs = cuda.device_array((B, T), dtype=np.int64)
    d_ovf = cuda.device_array((B,), dtype=np.int32)
    d_peak_on = cuda.device_array((B,), dtype=np.int32)
    d_peak_cl = cuda.device_array((B,), dtype=np.int32)

    threads_per_block = max(32, int(tpb))
    kernel[B, threads_per_block](
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
