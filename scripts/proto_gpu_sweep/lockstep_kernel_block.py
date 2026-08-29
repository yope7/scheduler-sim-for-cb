"""lockstep_kernel_block — lockstep カーネルの「1ブロック=1エピソード」版(numba CUDA)。

既存の lockstep_kernel.py(1スレッド=1エピソード)は一切変更していない。本ファイルは同じ
step/obs カーネルを TPB スレッドで手分けして進める版で、alloc_state / 返り値は共通。

なぜ作るか(実測): lockstep の 13.7ms/step はほぼ「配置探索2回」(観測の urgency=オンプレ、
efficiency=クラウド。配置側はこの結果を再利用する)で占められていた。同じ探索を
raw_rollout_kernel_block で手分けしたところ 6.898→0.337 ms/step(20.5倍・全件一致)だったので、
ここにも同じ手分けを入れる。

手分けするもの: 占有カウントの更新(ENTER/EXIT/RESET、atomic.add + shared での集約)と
空きノードの探索(区間分割して各スレッドが1回走査し、スレッド0が繋いで最初の位置を決める)。
逐次のまま: 刈り取りの前詰め、イベント表への二分挿入、観測ベクトルの組み立て。

ビット一致の根拠は raw_rollout_kernel_block と同じ(占有カウントは加算の総和、空きノード数は
atomic の戻り値が0だった回数の総和で、どちらも実行順に依らない)。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from numba import cuda, float64, int32, int64

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.envs.scheduling_variants.event_c_env import (  # noqa: E402
    EFF_COST_K, EFF_GAIN_K, EFF_RATIO_K, EVENT_FEATURES, JOB_QUEUE_SIZE, N_EVENTS_OBS,
)

URGENCY_K = 16.0  # event_c_env._front_job_urgency の log1p 正規化幅
OBS_EVENTS_SIZE = N_EVENTS_OBS * EVENT_FEATURES          # 180

# [PCN_LOCKSTEP_CAND_M] 観測の「直近イベント上位30件」を取るための候補バッファ容量。
# 小さいと生存30件が埋まらず「過去イベント全走査(O(j)、j は最大5万)」に落ちる。
# 5万・実trace weekB の実測(B=2、ブロック版): 64→3.308 / 256→1.801 / 1024→1.321 /
# 2048→1.082 / 4096→1.216 ms/step。2048 が底(それ以上は挿入コストが勝つ)。
# 容量は正確性に無関係(結果は全件一致を確認済み)なので既定を 2048 にする。
_CAND_M_DEFAULT = int(os.environ.get("PCN_LOCKSTEP_CAND_M", "2048"))
OBS_EXTRA_DIM = 1 + 3                                      # urgency(1) + efficiency(3)
OBS_TOTAL_DIM = OBS_EVENTS_SIZE + JOB_QUEUE_SIZE + OBS_EXTRA_DIM  # 224


# ---------------------------------------------------------------------------
# 配置探索(prune + 候補sweep + pick + RESET)を device 関数として1本化。
# raw_rollout_kernel._build_kernel の per-job 本体から record_event(挿入)より前を切り出した。
# 呼び出し側の用途は2通り:
#   (a) 実配置(step_kernel): 戻り値の pick_lo/hi・best_nfrag を使って呼び出し側が record_event する。
#   (b) 読み取り専用プローブ(obs_kernel の urgency/efficiency): best_start だけ使う。
#       count_r は呼び出しの最後に必ずゼロへ戻る(RESET)ので、次の呼び出し(同じ資源への実配置
#       クエリを含む)に影響しない。ただし prune による ev_*_r の前詰め圧縮は不可逆(CPU参照と同じ)。
# ovf_code: 0=正常 / 2=断片数>K(pick_lo/hi の列数)。この場合は呼び出し側が ovf_flag を立てて
#           即座にそのエピソードの処理を中断すること(count_r の巻き戻しは行われない=CPU参照の
#           ハード return と同じ扱い。以後のカーネル呼び出しは ovf_flag を見て何もしない)。
# ---------------------------------------------------------------------------
TPB = 128
# [計測用] 1 にすると観測ベクトルの組み立てだけを飛ばす(探索と配置は通常どおり)。
# 出力は壊れるので時間の内訳を見るときだけ使う。
_SKIP_OBS_BUILD = os.environ.get("PCN_LOCKSTEP_SKIP_OBS", "0") == "1"


@cuda.jit(device=True)
def _find_start(
    e, arrival, width, height, continuous_only, n_nodes, thr,
    n_ev_r, prune_at_r,
    ev_start_r, ev_end_r, ev_nfrag_r, ev_frag_lo_r, ev_frag_hi_r,
    order_end_r, order_start_r, counted_r, count_r,
    remap, pick_lo, pick_hi,
):
    K = pick_lo.shape[1]

    # --- prune(償却間引き): raw_rollout_kernel._build_kernel と同一条件・同一結果 ---
    if n_ev_r >= prune_at_r:
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
        n_ev_r = m
        pa = 2 * m + 32
        if pa < 64:
            pa = 64
        prune_at_r = pa

    # --- 候補列先頭 index(end>=arrival)を二分探索 ---
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
                break
            idxe = order_end_r[e, oi]
            cur_start = ev_end_r[e, idxe]

        win_end = cur_start + width

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

        run_found = False
        fh_count = 0
        fh_frag_n = 0
        ovf_here = False
        if free_cnt >= height:
            run_start = -1
            run_lo = -1
            need_head = not continuous_only
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
                            ovf_here = True
                            break
                        pick_lo[e, fh_frag_n] = nd
                        pick_hi[e, fh_frag_n] = nd + 1
                        fh_frag_n += 1
                    else:
                        pick_hi[e, fh_frag_n - 1] = nd + 1
                    fh_last_node = nd
                    fh_count += 1

            if ovf_here:
                # CPU参照のハード return と同じ扱い(RESET せず即座に呼び出し側へ通知)。
                return best_start, best_nfrag, n_ev_r, prune_at_r, 2
            if run_found:
                best_start = cur_start
                pick_lo[e, 0] = run_lo
                pick_hi[e, 0] = run_lo + height
                best_nfrag = 1
                success = True
            elif need_head and fh_count >= height:
                best_start = cur_start
                best_nfrag = fh_frag_n
                success = True

        if success:
            break
        cand_off += 1

    # --- RESET: ENTER が訪れた範囲の counted イベントを減算し count を全ゼロへ戻す ---
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
        # 保険: すべての既存イベント終了後なら必ず空く(参照実装と同じ)。
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

    return best_start, best_nfrag, n_ev_r, prune_at_r, 0



# ---------------------------------------------------------------------------
# _find_start_block: _find_start と同じ探索を、ブロック内 TPB スレッドで手分けして行う。
# 呼び出し側は「ブロック内の全スレッドが必ずここを通る」ことを守ること(途中に
# cuda.syncthreads があるため)。戻り値は全スレッドで同一。
# sh/sh64/sh_pre/sh_suf/sh_best/sh_all は呼び出し側が確保した shared 配列。
#   sh[0]=i_en sh[1]=i_ex sh[2]=success sh[3]=ovf sh[4]=nfrag sh[5]=free
#   sh[6]=集約用 sh[7]=n_ev sh[8]=prune_at   sh64[0]=best_start
# ---------------------------------------------------------------------------
@cuda.jit(device=True)
def _find_start_block(
    e, tid, nthreads, arrival, width, height, continuous_only, n_nodes, thr,
    n_ev_r, prune_at_r,
    ev_start_r, ev_end_r, ev_nfrag_r, ev_frag_lo_r, ev_frag_hi_r,
    order_end_r, order_start_r, counted_r, count_r,
    remap, pick_lo, pick_hi,
    sh, sh64, sh_pre, sh_suf, sh_best, sh_all,
):
    K = pick_lo.shape[1]
    if tid == 0:
        sh[3] = 0
        sh[7] = n_ev_r
        sh[8] = prune_at_r
    cuda.syncthreads()

    # --- 刈り取り(前詰め)は順序に依存するのでスレッド0が逐次で行う ---
    if n_ev_r >= prune_at_r:
        if tid == 0:
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
            sh[7] = m
            sh[8] = pa
        cuda.syncthreads()
    n_ev_r = sh[7]
    prune_at_r = sh[8]

    # --- 候補列の先頭(end>=arrival)を二分探索(全スレッドが同じ計算) ---
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
        sh[0] = 0
        sh[1] = 0
        sh[2] = 0
        sh[4] = 0
        sh[5] = n_nodes
        sh64[0] = 0
    cuda.syncthreads()

    cand_off = -1
    while True:
        if cand_off == -1:
            cur_start = arrival
        else:
            oi = start_idx + cand_off
            if oi >= n_ev_r:
                break
            idxe = order_end_r[e, oi]
            cur_start = ev_end_r[e, idxe]
        win_end = cur_start + width

        # ENTER: 取り込む範囲を二分探索で確定し、イベントを手分けして計上
        i_en = sh[0]
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
            sh[6] = 0
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
            cuda.atomic.add(sh, 6, dec_local)
        cuda.syncthreads()
        if tid == 0:
            sh[5] -= sh[6]
            sh[0] = i_en2
        cuda.syncthreads()

        # EXIT: 取り除く範囲を二分探索で確定し、手分けして減算
        i_ex = sh[1]
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
            sh[6] = 0
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
            cuda.atomic.add(sh, 6, inc_local)
        cuda.syncthreads()
        if tid == 0:
            sh[5] += sh[6]
            sh[1] = i_ex2
        cuda.syncthreads()

        # --- 空きノードの探索: 各スレッドが担当区間を1回走査し、スレッド0が繋ぐ ---
        run_lo_par = -1
        if sh[5] >= height:
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

        if run_lo_par >= 0:
            if tid == 0:
                sh64[0] = cur_start
                pick_lo[e, 0] = run_lo_par
                pick_hi[e, 0] = run_lo_par + height
                sh[4] = 1
                sh[2] = 1
            cuda.syncthreads()
        elif sh[5] >= height and not continuous_only:
            # 連続した空きが無い: 先頭から height 個の断片リストをスレッド0が作る(従来と同じ)
            if tid == 0:
                fh_count = 0
                fh_frag_n = 0
                fh_last_node = -2
                for nd in range(n_nodes):
                    if count_r[e, nd] != 0:
                        continue
                    if fh_count >= height:
                        break
                    if fh_count == 0 or nd != fh_last_node + 1:
                        if fh_frag_n >= K:
                            sh[3] = 2
                            break
                        pick_lo[e, fh_frag_n] = nd
                        pick_hi[e, fh_frag_n] = nd + 1
                        fh_frag_n += 1
                    else:
                        pick_hi[e, fh_frag_n - 1] = nd + 1
                    fh_last_node = nd
                    fh_count += 1
                if sh[3] == 0 and fh_count >= height:
                    sh64[0] = cur_start
                    sh[4] = fh_frag_n
                    sh[2] = 1
            cuda.syncthreads()

        if sh[3] != 0:
            return int64(0), 0, n_ev_r, prune_at_r, 2
        if sh[2] != 0:
            break
        cand_off += 1

    # --- RESET: ENTER が訪れた範囲を手分けして巻き戻す ---
    i_en_done = sh[0]
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

    if sh[2] == 0:
        # 保険: すべての既存イベント終了後なら必ず空く(参照実装と同じ)
        if tid == 0:
            if n_ev_r > 0:
                last_idx = order_end_r[e, n_ev_r - 1]
                max_end = ev_end_r[e, last_idx]
                if max_end < arrival:
                    max_end = arrival
            else:
                max_end = arrival
            sh64[0] = max_end
            sh[4] = 1
            pick_lo[e, 0] = 0
            pick_hi[e, 0] = height
        cuda.syncthreads()

    return sh64[0], sh[4], n_ev_r, prune_at_r, 0


# ---------------------------------------------------------------------------
# step_kernel: 全エピソードの j 番目のジョブを1個配置する。
#
# probe再利用(ureuse_*/creuse_*): 直前の obs_kernel(j) が urgency(オンプレ)/efficiency
# (クラウド)のプローブで解いた配置クエリの結果 (start, 断片リスト) を再利用し、
# 配置探索を丸ごと省く。CPU参照の PCN_REUSE_URGENCY_ALLOC(既定ON)と同じ論理で、
# さらにクラウド側にも拡張している。正当性: obs_kernel(j) と step_kernel(j) の間で
# イベント表は不変・探索は純関数(count系は必ずゼロへ巻き戻る)・prune はプローブ内で
# 適用済み(直後の同一クエリは prune_at 未達で再prune しない)ため、スキップしても
# 結果・持続状態ともにビット一致。*reuse_j[e] != j(obs_kernel 未実行=B-1単体等)なら
# 従来どおり自前で探索する。
# ---------------------------------------------------------------------------
def _build_step_kernel():
    @cuda.jit
    def _kernel(
        jobs, actions, suffix_min, j, B, n_on, n_cl, e_max,
        ev_on_start, ev_on_end, ev_on_nfrag, ev_on_frag_lo, ev_on_frag_hi,
        order_end_on, order_start_on, counted_on, count_on,
        ev_cl_start, ev_cl_end, ev_cl_nfrag, ev_cl_frag_lo, ev_cl_frag_hi,
        order_end_cl, order_start_cl, counted_cl, count_cl,
        n_ev_on_arr, n_ev_cl_arr, prune_at_on_arr, prune_at_cl_arr,
        remap, pick_lo, pick_hi,
        ureuse_j, ureuse_start, ureuse_nfrag, ureuse_lo, ureuse_hi,
        creuse_j, creuse_start, creuse_lo0, creuse_hi0,
        start_times, waits, costs, node_start, time_state,
        ovf_flag, peak_ev_on, peak_ev_cl,
    ):
        e = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        nthreads = cuda.blockDim.x
        sh = cuda.shared.array(16, int32)
        sh64 = cuda.shared.array(4, int64)
        sh_pre = cuda.shared.array(TPB, int32)
        sh_suf = cuda.shared.array(TPB, int32)
        sh_best = cuda.shared.array(TPB, int32)
        sh_all = cuda.shared.array(TPB, int32)
        if e >= B:
            return
        if ovf_flag[e] != 0:
            return

        arrival = int64(jobs[j, 0])
        width = int64(jobs[j, 1])
        height = int64(jobs[j, 2])
        use_cloud = actions[e, j] != 0
        thr = suffix_min[j]

        if use_cloud:
            n_ev_r = n_ev_cl_arr[e]
            prune_at_r = prune_at_cl_arr[e]
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
            n_ev_r = n_ev_on_arr[e]
            prune_at_r = prune_at_on_arr[e]
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

        # --- probe再利用: obs_kernel(j) が同じクエリを解いていれば探索をスキップ ---
        reuse_hit = False
        best_start = int64(0)
        best_nfrag = 0
        if use_cloud:
            if creuse_j[e] == j:
                best_start = creuse_start[e]
                best_nfrag = 1
                if tid == 0:
                    pick_lo[e, 0] = creuse_lo0[e]
                    pick_hi[e, 0] = creuse_hi0[e]
                reuse_hit = True
        else:
            if ureuse_j[e] == j:
                best_start = ureuse_start[e]
                best_nfrag = ureuse_nfrag[e]
                if tid == 0:
                    for f in range(best_nfrag):
                        pick_lo[e, f] = ureuse_lo[e, f]
                        pick_hi[e, f] = ureuse_hi[e, f]
                reuse_hit = True

        if not reuse_hit:
            best_start, best_nfrag, n_ev_r, prune_at_r, ovf_code = _find_start_block(
                e, tid, nthreads, arrival, width, height, use_cloud, n_nodes, thr,
                n_ev_r, prune_at_r,
                ev_start_r, ev_end_r, ev_nfrag_r, ev_frag_lo_r, ev_frag_hi_r,
                order_end_r, order_start_r, counted_r, count_r,
                remap, pick_lo, pick_hi,
                sh, sh64, sh_pre, sh_suf, sh_best, sh_all,
            )
            if ovf_code != 0:
                if tid == 0:
                    ovf_flag[e] = ovf_code
                return

        if tid == 0:
            idx = n_ev_r
            if idx >= e_max:
                ovf_flag[e] = 1
                return

            ev_start_r[e, idx] = best_start
            end_val = best_start + width
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

            lo3 = 0
            hi3 = n_ev_r
            while lo3 < hi3:
                mid3 = (lo3 + hi3) // 2
                if ev_start_r[e, order_start_r[e, mid3]] <= best_start:
                    lo3 = mid3 + 1
                else:
                    hi3 = mid3
            p = n_ev_r
            while p > lo3:
                order_start_r[e, p] = order_start_r[e, p - 1]
                p -= 1
            order_start_r[e, lo3] = idx

            if use_cloud:
                n_ev_cl_arr[e] = idx + 1
                prune_at_cl_arr[e] = prune_at_r
                if idx + 1 > peak_ev_cl[e]:
                    peak_ev_cl[e] = idx + 1
                cost = width * height
            else:
                n_ev_on_arr[e] = idx + 1
                prune_at_on_arr[e] = prune_at_r
                if idx + 1 > peak_ev_on[e]:
                    peak_ev_on[e] = idx + 1
                cost = int64(0)

            node_start[e, j] = pick_lo[e, 0]
            start_times[e, j] = best_start
            waits[e, j] = best_start - arrival
            costs[e, j] = cost
            time_state[e] = best_start

    return _kernel


# ---------------------------------------------------------------------------
# obs_kernel: job j を配置する「直前」の観測 obs[e, j, :224] を構築する。
# ---------------------------------------------------------------------------
def _build_obs_kernel():
    @cuda.jit
    def _kernel(
        jobs, actions, suffix_min, j, B, T, n_on, n_cl,
        n_window, norm_time, norm_nodes,
        start_times, node_start, time_state,
        ev_on_start, ev_on_end, ev_on_nfrag, ev_on_frag_lo, ev_on_frag_hi,
        order_end_on, order_start_on, counted_on, count_on, n_ev_on_arr, prune_at_on_arr,
        ev_cl_start, ev_cl_end, ev_cl_nfrag, ev_cl_frag_lo, ev_cl_frag_hi,
        order_end_cl, order_start_cl, counted_cl, count_cl, n_ev_cl_arr, prune_at_cl_arr,
        remap, pick_lo, pick_hi,
        buf_start, buf_t,
        cand_start, cand_t, cand_n, last_ins,
        ureuse_j, ureuse_start, ureuse_nfrag, ureuse_lo, ureuse_hi,
        creuse_j, creuse_start, creuse_lo0, creuse_hi0,
        ovf_flag, obs_out,
    ):
        e = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        nthreads = cuda.blockDim.x
        sh = cuda.shared.array(16, int32)
        sh64 = cuda.shared.array(4, int64)
        sh_pre = cuda.shared.array(TPB, int32)
        sh_suf = cuda.shared.array(TPB, int32)
        sh_best = cuda.shared.array(TPB, int32)
        sh_all = cuda.shared.array(TPB, int32)
        if e >= B:
            return
        if ovf_flag[e] != 0:
            return

        if tid == 0 and not _SKIP_OBS_BUILD:
            cur_time = time_state[e]
            window_start = cur_time - n_window
            if window_start < 0:
                window_start = 0

            n_events_obs = buf_start.shape[1]

            # --- 増分候補バッファ: (start,t) 辞書式順の上位 M 件(生死問わず)を step 間で維持 ---
            # 不変条件: cand は「これまでの全イベントのうち (start,t) 最大の cand_n 件」を昇順保持。
            #   満杯時の挿入は最小要素を追い出し(または新イベント自身を破棄)、追い出された/破棄された
            #   イベントは常に現メンバー全員より下位。よって window フィルタ後の上位 n_events_obs 件が
            #   バッファ内で n_events_obs 件見つかれば、それは全体でも正しい上位集合(生存する buffer外
            #   イベントは全メンバーより下位)。見つからない場合、まだ何も捨てていなければ(j<=M)
            #   バッファ=全イベントで正確、捨てた後(j>M)のみ O(j) の全走査へフォールバックする。
            #   window_start は後退し得る(time_state は直前ジョブの start で非単調)ため「死んだ
            #   イベントの恒久削除」はできない=生死問わず保持がこの設計の要。
            M = cand_start.shape[1]
            cn = cand_n[e]
            if j >= 1 and last_ins[e] < j - 1:
                t_new = j - 1
                s_new = start_times[e, t_new]
                if cn < M:
                    pos = cn
                    while pos > 0 and cand_start[e, pos - 1] > s_new:
                        cand_start[e, pos] = cand_start[e, pos - 1]
                        cand_t[e, pos] = cand_t[e, pos - 1]
                        pos -= 1
                    cand_start[e, pos] = s_new
                    cand_t[e, pos] = t_new
                    cn += 1
                    cand_n[e] = cn
                elif s_new >= cand_start[e, 0]:
                    # 最小(index0)を追い出して挿入(同値 start は t_new が最大なので後ろ=<=で送る)
                    pos = 0
                    while pos + 1 < M and cand_start[e, pos + 1] <= s_new:
                        cand_start[e, pos] = cand_start[e, pos + 1]
                        cand_t[e, pos] = cand_t[e, pos + 1]
                        pos += 1
                    cand_start[e, pos] = s_new
                    cand_t[e, pos] = t_new
                # else: top-M 外→破棄(以後も上位に入り得ない)
                last_ins[e] = t_new

            # --- events 部分: 「end>=window_start を start 昇順 stable で並べた末尾 n_events_obs 件」
            #     = (start,t) 辞書式順で最大の n_events_obs 件。まず候補バッファを上から歩いて生存
            #     イベントを集める(O(n_events_obs+スキップ数))。 ---
            kk = 0
            i2 = cn - 1
            while i2 >= 0 and kk < n_events_obs:
                tt0 = cand_t[e, i2]
                s0 = cand_start[e, i2]
                if s0 + int64(jobs[tt0, 1]) >= window_start:
                    buf_start[e, kk] = s0     # 降順で仮置き(後で反転)
                    buf_t[e, kk] = tt0
                    kk += 1
                i2 -= 1

            if kk >= n_events_obs or j <= M:
                # 正確: バッファ内で満了 or 何も捨てていない(バッファ=全イベント)
                n_take = kk
                a2 = 0
                b2 = kk - 1
                while a2 < b2:  # 降順→昇順へ反転
                    ts = buf_start[e, a2]
                    buf_start[e, a2] = buf_start[e, b2]
                    buf_start[e, b2] = ts
                    ti = buf_t[e, a2]
                    buf_t[e, a2] = buf_t[e, b2]
                    buf_t[e, b2] = ti
                    a2 += 1
                    b2 -= 1
            else:
                # フォールバック(稀): 全過去イベントの O(j) 走査(従来実装そのまま=正確性の錨)
                n_take = 0
                for t in range(j):
                    width_t = int64(jobs[t, 1])
                    s_t = start_times[e, t]
                    end_t = s_t + width_t
                    if end_t >= window_start:
                        if n_take < n_events_obs:
                            pos = n_take
                            while pos > 0 and buf_start[e, pos - 1] > s_t:
                                buf_start[e, pos] = buf_start[e, pos - 1]
                                buf_t[e, pos] = buf_t[e, pos - 1]
                                pos -= 1
                            buf_start[e, pos] = s_t
                            buf_t[e, pos] = t
                            n_take += 1
                        elif s_t >= buf_start[e, 0]:
                            pos = 0
                            while pos < n_events_obs - 1 and buf_start[e, pos + 1] <= s_t:
                                buf_start[e, pos] = buf_start[e, pos + 1]
                                buf_t[e, pos] = buf_t[e, pos + 1]
                                pos += 1
                            buf_start[e, pos] = s_t
                            buf_t[e, pos] = t

            event_features = 6
            for i in range(n_events_obs):
                base = i * event_features
                if i < n_take:
                    tt = buf_t[e, i]
                    s_v = buf_start[e, i]
                    width_v = jobs[tt, 1]
                    height_v = jobs[tt, 2]
                    e_v = s_v + int64(width_v)
                    uc_v = actions[e, tt]
                    sn_v = node_start[e, tt]

                    v0 = (float64(s_v) - float64(window_start)) / norm_time
                    if v0 < 0.0:
                        v0 = 0.0
                    elif v0 > 1.0:
                        v0 = 1.0
                    v1 = (float64(e_v) - float64(window_start)) / norm_time
                    if v1 < 0.0:
                        v1 = 0.0
                    elif v1 > 1.0:
                        v1 = 1.0
                    v2 = float64(width_v) / norm_time
                    if v2 < 0.0:
                        v2 = 0.0
                    elif v2 > 1.0:
                        v2 = 1.0
                    v4 = float64(sn_v) / norm_nodes
                    if v4 < 0.0:
                        v4 = 0.0
                    elif v4 > 1.0:
                        v4 = 1.0
                    v5 = float64(height_v) / norm_nodes
                    if v5 < 0.0:
                        v5 = 0.0
                    elif v5 > 1.0:
                        v5 = 1.0

                    obs_out[e, j, base + 0] = v0
                    obs_out[e, j, base + 1] = v1
                    obs_out[e, j, base + 2] = v2
                    obs_out[e, j, base + 3] = float64(uc_v)
                    obs_out[e, j, base + 4] = v4
                    obs_out[e, j, base + 5] = v5
                else:
                    obs_out[e, j, base + 0] = 0.0
                    obs_out[e, j, base + 1] = 0.0
                    obs_out[e, j, base + 2] = 0.0
                    obs_out[e, j, base + 3] = 0.0
                    obs_out[e, j, base + 4] = 0.0
                    obs_out[e, j, base + 5] = 0.0

            events_size = n_events_obs * event_features

            # --- job_queue 部分: jobs[j:j+5] を roll(-1) した生値(5*8=40) ---
            for i in range(5):
                tt = j + i
                base = events_size + i * 8
                if tt < T:
                    obs_out[e, j, base + 0] = jobs[tt, 1]
                    obs_out[e, j, base + 1] = jobs[tt, 2]
                    obs_out[e, j, base + 2] = jobs[tt, 3]
                    obs_out[e, j, base + 3] = jobs[tt, 4]
                    obs_out[e, j, base + 4] = jobs[tt, 5]
                    obs_out[e, j, base + 5] = jobs[tt, 6]
                    obs_out[e, j, base + 6] = jobs[tt, 7]
                    obs_out[e, j, base + 7] = jobs[tt, 0]
                else:
                    for c in range(8):
                        obs_out[e, j, base + c] = 0.0

            extra_base = events_size + 40

            if j >= T:
                obs_out[e, j, extra_base + 0] = 0.0
                obs_out[e, j, extra_base + 1] = 0.0
                obs_out[e, j, extra_base + 2] = 0.0
                obs_out[e, j, extra_base + 3] = 0.0

        cuda.syncthreads()
        # 終端観測(j>=T)は探索不要。抜けるときはブロック全員で抜ける(片方だけ抜けると
        # 残りが cuda.syncthreads で待ち続けるため)。
        if j >= T:
            return
        # 以降(探索)はブロック全員で通る。extra_base は上のブロック内と同じ値を作り直す。
        extra_base = buf_start.shape[1] * 6 + 40
        arrival = int64(jobs[j, 0])
        width = int64(jobs[j, 1])
        height = int64(jobs[j, 2])
        thr = suffix_min[j]

        # --- urgency: 現ジョブをオンプレに置いたときの開始時刻(副作用: prune のみ持続) ---
        n_ev_r = n_ev_on_arr[e]
        prune_at_r = prune_at_on_arr[e]
        s_on, nf_u, n_ev_r, prune_at_r, ovf_code = _find_start_block(
            e, tid, nthreads, arrival, width, height, False, n_on, thr,
            n_ev_r, prune_at_r,
            ev_on_start, ev_on_end, ev_on_nfrag, ev_on_frag_lo, ev_on_frag_hi,
            order_end_on, order_start_on, counted_on, count_on,
            remap, pick_lo, pick_hi,
            sh, sh64, sh_pre, sh_suf, sh_best, sh_all,
        )
        if ovf_code != 0:
            if tid == 0:
                ovf_flag[e] = ovf_code
            return
        n_ev_on_arr[e] = n_ev_r
        prune_at_on_arr[e] = prune_at_r
        # step_kernel(j) の onprem 配置向けに (start, 断片リスト) をキャッシュ
        # (クラウドプローブが pick_lo/hi を上書きする前に退避)。
        ureuse_j[e] = j
        ureuse_start[e] = s_on
        ureuse_nfrag[e] = nf_u
        for f in range(nf_u):
            ureuse_lo[e, f] = pick_lo[e, f]
            ureuse_hi[e, f] = pick_hi[e, f]

        pw = s_on - arrival
        if pw < 0:
            pw = 0
        urgency = math.log1p(float64(pw)) / URGENCY_K
        if urgency < 0.0:
            urgency = 0.0
        elif urgency > 1.0:
            urgency = 1.0
        obs_out[e, j, extra_base + 0] = urgency

        # --- efficiency: a=pt*nodes, b=max(0,s_on-s_cl) ---
        n_ev_rc = n_ev_cl_arr[e]
        prune_at_rc = prune_at_cl_arr[e]
        s_cl, _nf_e, n_ev_rc, prune_at_rc, ovf_code2 = _find_start_block(
            e, tid, nthreads, arrival, width, height, True, n_cl, thr,
            n_ev_rc, prune_at_rc,
            ev_cl_start, ev_cl_end, ev_cl_nfrag, ev_cl_frag_lo, ev_cl_frag_hi,
            order_end_cl, order_start_cl, counted_cl, count_cl,
            remap, pick_lo, pick_hi,
            sh, sh64, sh_pre, sh_suf, sh_best, sh_all,
        )
        if ovf_code2 != 0:
            if tid == 0:
                ovf_flag[e] = ovf_code2
            return
        n_ev_cl_arr[e] = n_ev_rc
        prune_at_cl_arr[e] = prune_at_rc
        # step_kernel(j) の cloud 配置向けキャッシュ(cloud は常に nfrag=1)。
        creuse_j[e] = j
        creuse_start[e] = s_cl
        creuse_lo0[e] = pick_lo[e, 0]
        creuse_hi0[e] = pick_hi[e, 0]

        a = float64(width) * float64(height)
        b = float64(s_on) - float64(s_cl)
        if b < 0.0:
            b = 0.0
        la = math.log1p(a)
        lb = math.log1p(b)

        f0 = la / EFF_COST_K
        if f0 < 0.0:
            f0 = 0.0
        elif f0 > 1.0:
            f0 = 1.0
        f1 = lb / EFF_GAIN_K
        if f1 < 0.0:
            f1 = 0.0
        elif f1 > 1.0:
            f1 = 1.0
        f2 = (lb - la) / EFF_RATIO_K + 0.5
        if f2 < 0.0:
            f2 = 0.0
        elif f2 > 1.0:
            f2 = 1.0

        obs_out[e, j, extra_base + 1] = f0
        obs_out[e, j, extra_base + 2] = f1
        obs_out[e, j, extra_base + 3] = f2

    return _kernel


_STEP_KERNEL = None
_OBS_KERNEL = None


def _get_step_kernel():
    global _STEP_KERNEL
    if _STEP_KERNEL is None:
        _STEP_KERNEL = _build_step_kernel()
    return _STEP_KERNEL


def _get_obs_kernel():
    global _OBS_KERNEL
    if _OBS_KERNEL is None:
        _OBS_KERNEL = _build_obs_kernel()
    return _OBS_KERNEL


# ---------------------------------------------------------------------------
# 状態確保
# ---------------------------------------------------------------------------
class LockstepState:
    """1回の run_lockstep_rollout 呼び出し分の GPU 常駐状態(alloc_state の戻り値)。"""

    def __init__(self, B, T, n_on, n_cl, e_max, k, n_events_obs, collect_obs, cand_m=64):
        self.B, self.T = B, T
        self.n_on, self.n_cl = n_on, n_cl
        self.e_max, self.k = e_max, k
        self.collect_obs = collect_obs

        z_i64 = lambda shape: cuda.to_device(np.zeros(shape, dtype=np.int64))  # noqa: E731
        z_i32 = lambda shape: cuda.to_device(np.zeros(shape, dtype=np.int32))  # noqa: E731
        z_i8 = lambda shape: cuda.to_device(np.zeros(shape, dtype=np.int8))  # noqa: E731

        self.ev_on_start = z_i64((B, e_max))
        self.ev_on_end = z_i64((B, e_max))
        self.ev_on_nfrag = z_i32((B, e_max))
        self.ev_on_frag_lo = z_i32((B, e_max, k))
        self.ev_on_frag_hi = z_i32((B, e_max, k))
        self.order_end_on = z_i32((B, e_max))
        self.order_start_on = z_i32((B, e_max))
        self.counted_on = z_i8((B, e_max))

        self.ev_cl_start = z_i64((B, e_max))
        self.ev_cl_end = z_i64((B, e_max))
        self.ev_cl_nfrag = z_i32((B, e_max))
        self.ev_cl_frag_lo = z_i32((B, e_max, 1))
        self.ev_cl_frag_hi = z_i32((B, e_max, 1))
        self.order_end_cl = z_i32((B, e_max))
        self.order_start_cl = z_i32((B, e_max))
        self.counted_cl = z_i8((B, e_max))

        self.count_on = z_i32((B, max(n_on, 1)))
        self.count_cl = z_i32((B, max(n_cl, 1)))
        self.remap = z_i32((B, e_max))
        self.pick_lo = z_i32((B, max(k, 1)))
        self.pick_hi = z_i32((B, max(k, 1)))

        self.n_ev_on = z_i32((B,))
        self.n_ev_cl = z_i32((B,))
        self.prune_at_on = cuda.to_device(np.full((B,), 64, dtype=np.int32))
        self.prune_at_cl = cuda.to_device(np.full((B,), 64, dtype=np.int32))

        self.start_times = z_i64((B, T))
        self.waits = z_i64((B, T))
        self.costs = z_i64((B, T))
        self.node_start = z_i32((B, T))
        self.time_state = z_i64((B,))

        self.ovf_flag = z_i32((B,))
        self.peak_ev_on = z_i32((B,))
        self.peak_ev_cl = z_i32((B,))

        # probe再利用キャッシュ(obs_kernel が書き step_kernel が読む。*_j==-1 で無効=
        # B-1 単体でも step_kernel は同一シグネチャで安全に動く)。
        self.ureuse_j = cuda.to_device(np.full((B,), -1, dtype=np.int32))
        self.ureuse_start = z_i64((B,))
        self.ureuse_nfrag = z_i32((B,))
        self.ureuse_lo = z_i32((B, max(k, 1)))
        self.ureuse_hi = z_i32((B, max(k, 1)))
        self.creuse_j = cuda.to_device(np.full((B,), -1, dtype=np.int32))
        self.creuse_start = z_i64((B,))
        self.creuse_lo0 = z_i32((B,))
        self.creuse_hi0 = z_i32((B,))

        if collect_obs:
            self.buf_start = z_i64((B, n_events_obs))
            self.buf_t = z_i32((B, n_events_obs))
            # 増分候補バッファ((start,t)上位 cand_m 件を昇順維持)。
            self.cand_m = int(cand_m)
            self.cand_start = z_i64((B, self.cand_m))
            self.cand_t = z_i32((B, self.cand_m))
            self.cand_n = z_i32((B,))
            self.last_ins = cuda.to_device(np.full((B,), -1, dtype=np.int32))
            self.obs = cuda.to_device(np.zeros((B, T, OBS_TOTAL_DIM), dtype=np.float64))
        else:
            self.buf_start = None
            self.buf_t = None
            self.cand_start = None
            self.cand_t = None
            self.cand_n = None
            self.last_ins = None
            self.obs = None


def alloc_state(B, T, n_on, n_cl, e_max=8192, k=16, n_events_obs=N_EVENTS_OBS,
                collect_obs=True, cand_m=_CAND_M_DEFAULT):
    """B-1/B-2 の GPU 常駐状態を確保する(全てゼロ初期化。prune_at_* のみ 64 初期化)。

    cand_m: obs 増分候補バッファの容量(>= n_events_obs)。大きいほどフォールバック
            (O(j) 全走査)が減るが挿入コストが増える。既定 2048(実測の底)。
    """
    if cand_m < n_events_obs:
        raise ValueError(f"cand_m={cand_m} < n_events_obs={n_events_obs}")
    return LockstepState(B, T, n_on, n_cl, e_max, k, n_events_obs, collect_obs, cand_m)


# ---------------------------------------------------------------------------
# ドライバ
# ---------------------------------------------------------------------------
def run_lockstep_rollout_block(
    jobs, actions, n_on: int, n_cl: int, *,
    e_max: int = 8192, k: int = 16, n_window: int = 16, tpb: int = 1,
    collect_obs: bool = True, state: "LockstepState | None" = None,
    return_host: bool = True,
) -> dict:
    """ロックステップ(1step=envカーネル1[+obsカーネル1])で GPU 完結 rollout を実行する。

    Args:
        jobs: (T,8) 配列(raw形式、raw_rollout_kernel.run_raw_rollout と同一)。
        actions: (B,T) 配列。0=onprem, 1=cloud(defer非対応)。
        n_on, n_cl: オンプレ/クラウドのノード数。
        e_max, k: raw_rollout_kernel と同義(資源別イベント上限・断片上限)。
        n_window: 観測の時間窓(event_c_env._norm_time = max(1,n_window) と同じ値を渡すこと。
                  学習/リプレイと不一致だと obs がズレる=設計書の既知の罠)。
        tpb: threads_per_block(既定1。ワープ発散回避、v5実測で32は5倍遅かった)。
        collect_obs: True なら obs_kernel を毎 step 起動し obs[B,T,224] を構築・返す。
                     False なら step_kernel のみを起動する(B-1 単体の性能測定用)。
        state: 既存の LockstepState を使い回す場合(性能測定でアロケーションを除外したい時)。
               None なら新規 alloc_state する。
        return_host: True なら結果を host(numpy)へコピーして返す。False は device 配列のまま
                     (性能測定で PCIe 転送を計測対象から外したい時用)。

    Returns:
        dict(start_times, waits, costs: (B,T) int64, node_start: (B,T) int32,
             objectives: (B,2) float64, ovf: (B,) int32,
             peak_ev_on, peak_ev_cl: (B,) int32,
             obs: (B,T,224) float64 or None)
    """
    jobs = np.ascontiguousarray(jobs, dtype=np.float64)
    actions = np.ascontiguousarray(actions, dtype=np.int8)
    T = int(jobs.shape[0])
    B = int(actions.shape[0])
    if actions.shape[1] != T:
        raise ValueError(f"actions.shape[1]={actions.shape[1]} != T={T}")

    arrivals = jobs[:, 0].astype(np.float64)
    suffix_min = np.minimum.accumulate(arrivals[::-1])[::-1].copy()

    step_kernel = _get_step_kernel()
    obs_kernel = _get_obs_kernel() if collect_obs else None

    d_jobs = cuda.to_device(jobs)
    d_actions = cuda.to_device(actions)
    d_suffix = cuda.to_device(suffix_min)

    if state is None:
        state = alloc_state(B, T, n_on, n_cl, e_max=e_max, k=k, collect_obs=collect_obs)
    s = state

    norm_time = float(max(1, n_window))
    norm_nodes = float(max(1, max(n_on, n_cl)))

    # 1ブロック=1エピソード。tpb はその1本を手分けするスレッド数(shared 配列の形に合わせ TPB 固定)。
    threads_per_block = TPB
    blocks = B

    for j in range(T):
        if collect_obs:
            obs_kernel[blocks, threads_per_block](
                d_jobs, d_actions, d_suffix, j, B, T, n_on, n_cl,
                float(n_window), norm_time, norm_nodes,
                s.start_times, s.node_start, s.time_state,
                s.ev_on_start, s.ev_on_end, s.ev_on_nfrag, s.ev_on_frag_lo, s.ev_on_frag_hi,
                s.order_end_on, s.order_start_on, s.counted_on, s.count_on,
                s.n_ev_on, s.prune_at_on,
                s.ev_cl_start, s.ev_cl_end, s.ev_cl_nfrag, s.ev_cl_frag_lo, s.ev_cl_frag_hi,
                s.order_end_cl, s.order_start_cl, s.counted_cl, s.count_cl,
                s.n_ev_cl, s.prune_at_cl,
                s.remap, s.pick_lo, s.pick_hi,
                s.buf_start, s.buf_t,
                s.cand_start, s.cand_t, s.cand_n, s.last_ins,
                s.ureuse_j, s.ureuse_start, s.ureuse_nfrag, s.ureuse_lo, s.ureuse_hi,
                s.creuse_j, s.creuse_start, s.creuse_lo0, s.creuse_hi0,
                s.ovf_flag, s.obs,
            )
        step_kernel[blocks, threads_per_block](
            d_jobs, d_actions, d_suffix, j, B, n_on, n_cl, e_max,
            s.ev_on_start, s.ev_on_end, s.ev_on_nfrag, s.ev_on_frag_lo, s.ev_on_frag_hi,
            s.order_end_on, s.order_start_on, s.counted_on, s.count_on,
            s.ev_cl_start, s.ev_cl_end, s.ev_cl_nfrag, s.ev_cl_frag_lo, s.ev_cl_frag_hi,
            s.order_end_cl, s.order_start_cl, s.counted_cl, s.count_cl,
            s.n_ev_on, s.n_ev_cl, s.prune_at_on, s.prune_at_cl,
            s.remap, s.pick_lo, s.pick_hi,
            s.ureuse_j, s.ureuse_start, s.ureuse_nfrag, s.ureuse_lo, s.ureuse_hi,
            s.creuse_j, s.creuse_start, s.creuse_lo0, s.creuse_hi0,
            s.start_times, s.waits, s.costs, s.node_start, s.time_state,
            s.ovf_flag, s.peak_ev_on, s.peak_ev_cl,
        )

    cuda.synchronize()

    if not return_host:
        return dict(state=s)

    start_times = s.start_times.copy_to_host()
    waits = s.waits.copy_to_host()
    costs = s.costs.copy_to_host()
    node_start = s.node_start.copy_to_host()
    ovf = s.ovf_flag.copy_to_host()
    peak_ev_on = s.peak_ev_on.copy_to_host()
    peak_ev_cl = s.peak_ev_cl.copy_to_host()
    # 参照実装(get_observation)は内部を float64/double で計算し、最終格納だけ float32 に
    # 丸める(C の `(float)` cast / np.asarray(..., dtype=np.float32))。GPU 側も同様に
    # float64 で計算し、host 転送後にここで float32 へ丸める(IEEE754 round-to-nearest は
    # どちらの経路でも同一なのでビット一致するはず)。
    obs = s.obs.copy_to_host().astype(np.float32) if collect_obs else None

    total_cost = costs.sum(axis=1).astype(np.float64)
    mean_wait = waits.mean(axis=1).astype(np.float64)
    objectives = np.stack([total_cost, mean_wait], axis=1)

    return dict(
        start_times=start_times, waits=waits, costs=costs, node_start=node_start,
        objectives=objectives, ovf=ovf,
        peak_ev_on=peak_ev_on, peak_ev_cl=peak_ev_cl,
        obs=obs,
    )
