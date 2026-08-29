"""イベントを正として資源占有を判定するスケジューリング環境。

この環境ではビットマップの time_transition を使わず、各ジョブ到着時刻に
既存イベントの占有ノードだけを見て配置を決める。
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from src.utils.schedule_map_options import schedule_maps_enabled

from .event_c_env import SchedulingEnvEventObs

# 配置探索の毎ステップ全イベント走査を、未来のクエリ区間に重なり得ないイベント
# （end < 残りジョブの最小到着時刻）を間引いて短縮する。結果は不変（除外イベントは
# 現在/未来の占有判定に決して寄与しない）。0 で旧挙動。
_FAST_ENV = os.environ.get("PCN_FAST_ENV", "1") == "1"
# 候補時刻ループ O(候補×イベント) を sweep-line + per-node 重なりカウントで O(R log R + R·H) に下げる。
# 旧ループが実質訪問する候補列（昇順の {arrival} ∪ {end>=arrival}; miss 時の min_overlap_end は
# 常にこの集合内なので新規候補を増やさない=スキップ無し）を同順に辿り、占有を増分維持するので結果不変。
# 検証完了で既定 ON（env単体 thash 全20セル一致＋NN eval ahash一致＋b2/128実学習で崩壊なし・PF同等）。
# イベントが少ない間は候補ループ(fused)の方が速いので閾値で切替（どちらでも結果は同一）。0 で旧挙動。
_FAST_ENV_SWEEP = _FAST_ENV and os.environ.get("PCN_FAST_ENV_SWEEP", "1") == "1"
_SWEEP_MIN_EVENTS = int(os.environ.get("PCN_SWEEP_MIN_EVENTS", "48"))
# sweep の per-node 重なりカウント（Python list の range(n_nodes) 走査）を numpy 配列の
# ベクトル演算に置換し、ENTER/EXIT の `for k in e["nodes"]` と pick の連続 run 探索から
# Python ループを排除する。占有集合・free run 選択は旧 sweep と同一なので結果はビット一致。
# 1イベント内のノードは重複しないので count[nodes]+=1 の fancy-index は重なり数を正しく数える。0 で旧 sweep。
_FAST_ENV_NP = _FAST_ENV_SWEEP and os.environ.get("PCN_FAST_ENV_NP", "1") == "1"
# urgency 観測のため毎ステップ実行する「先頭ジョブのオンプレ配置クエリ」(_find_event_allocation)を、
# 直後の step が同じジョブを onprem 配置するとき再利用し、二重実行を 1 回に削減する。
# get_observation→次step の間に events は変化しないので (position,start) は同一=ビット一致。0 で旧挙動。
_REUSE_URGENCY_ALLOC = os.environ.get("PCN_REUSE_URGENCY_ALLOC", "1") == "1"
# _to_queue_job の np.roll(arr,-1) を手書きロール(out[:-1]=arr[1:]; out[-1]=arr[0])に置換する。
# np.roll は normalize_axis_tuple 等の汎用前処理＋concatenate で重く、毎ステップ呼ぶ 1次元・左1回転には
# 過剰。手書きは同じ要素配置を float64 のまま生成するのでビット一致。プロファイルで ~7%。0 で旧 np.roll。
_FAST_QUEUE_ROLL = os.environ.get("PCN_FAST_QUEUE_ROLL", "1") == "1"
# 注: sweep_np 内の sorted×3/set 構築の numpy(argsort/unique)化は、イベント間引きで events 数が
# 数百に抑えられる規模では Python の sorted(timsort)+set より遅く（NJ=512 で +6%）撤回した。
# 配置あたり events 数は間引きで頭打ち=大規模でも好転しない。残る塊(sweep本体60%)は C化(段階2)で攻める。
# 段階2: sweep 本体を C 実装(scheduling_env_core.event_sweep_alloc)に置換。候補列・占有count・pick・
# フォールバックを全て C で行い、Python は events から CSR(starts/ends/nodes_flat/node_off)を1度組むだけ。
# 占有集合・候補順・free run 選択は sweep_np と同一なので結果はビット一致。
# 検証完了で既定 ON: env単体 bench thash 全一致(NJ512/1024・P=0/0.5/1.0 両分岐)＋実学習で崩壊なし(TRAIN_EXIT=0)
# ＋greedy eval ohash 一致。速度: eval(推論)が NJ128 1.5×→512 2.9×→1024純env 5.4×と規模で拡大、
# np 段階比 4〜5×・pure-Python比 26〜36×。分散学習(128J)は NN律速のため不変。0 で sweep_np に復帰。
# C 拡張が無い環境では import 失敗→自動的に sweep_np へフォールバック(_c_event_sweep_alloc is None)。
try:
    from scheduling_env_core import event_sweep_alloc as _c_event_sweep_alloc
except Exception:
    _c_event_sweep_alloc = None
_SWEEP_C = (
    _FAST_ENV_SWEEP
    and _c_event_sweep_alloc is not None
    and os.environ.get("PCN_SWEEP_C", "1") == "1"
)
# [PCN_EVENT_CSR_CACHE] sweep_c(C実装)に渡す配列(starts/ends/nodes_flat/node_off)は、
# 従来「イベント表(dict の list)から毎回組み直し」ていた。5万・v9設定の実測では、この
# 組み直しが env.step の 61.4%(0.832ms/step)で、C 本体(0.298ms)の 2.8 倍を占めていた。
# イベント追加のたびに末尾へ追記して持ち回れば、この O(生存イベント数) の Python 走査が消える。
# C へ渡すのは同じ値・同じ順序の配列なので結果はビット一致(PCN_EVENT_CSR_VERIFY=1 で全step照合可)。
_CSR_CACHE = os.environ.get("PCN_EVENT_CSR_CACHE", "1") == "1"
_CSR_VERIFY = os.environ.get("PCN_EVENT_CSR_VERIFY", "0") == "1"


class _EventCSR:
    """sweep_c に渡す CSR を増分で保つ入れ物(容量は倍々に伸ばし、C へは使用長の view を渡す)。"""

    __slots__ = ("starts", "ends", "node_off", "nodes_flat", "n", "m")

    def __init__(self, cap_rows: int = 256, cap_nodes: int = 4096):
        cap_rows = max(16, int(cap_rows))
        cap_nodes = max(64, int(cap_nodes))
        self.starts = np.empty(cap_rows, dtype=np.int64)
        self.ends = np.empty(cap_rows, dtype=np.int64)
        self.node_off = np.empty(cap_rows + 1, dtype=np.int32)
        self.node_off[0] = 0
        self.nodes_flat = np.empty(cap_nodes, dtype=np.int32)
        self.n = 0   # イベント数
        self.m = 0   # nodes_flat の使用長

    def _grow_rows(self) -> None:
        new = 2 * self.starts.shape[0]
        starts = np.empty(new, dtype=np.int64)
        starts[: self.n] = self.starts[: self.n]
        ends = np.empty(new, dtype=np.int64)
        ends[: self.n] = self.ends[: self.n]
        off = np.empty(new + 1, dtype=np.int32)
        off[: self.n + 1] = self.node_off[: self.n + 1]
        self.starts, self.ends, self.node_off = starts, ends, off

    def _grow_nodes(self, need: int) -> None:
        new = max(2 * self.nodes_flat.shape[0], self.m + need)
        flat = np.empty(new, dtype=np.int32)
        flat[: self.m] = self.nodes_flat[: self.m]
        self.nodes_flat = flat

    def append(self, start: int, end: int, nodes_np: np.ndarray) -> None:
        h = int(nodes_np.shape[0])
        if self.n >= self.starts.shape[0]:
            self._grow_rows()
        if self.m + h > self.nodes_flat.shape[0]:
            self._grow_nodes(h)
        self.starts[self.n] = start
        self.ends[self.n] = end
        self.nodes_flat[self.m : self.m + h] = nodes_np
        self.m += h
        self.n += 1
        self.node_off[self.n] = self.m


def _csr_from_events(events) -> "_EventCSR":
    """イベント表(dict の list)から CSR を1度だけ組む(従来の組み直しと同一の内容・順序)。"""
    n = len(events)
    c = _EventCSR(max(256, 2 * n))
    if n:
        c.starts[:n] = np.fromiter((e["start"] for e in events), dtype=np.int64, count=n)
        c.ends[:n] = np.fromiter((e["end"] for e in events), dtype=np.int64, count=n)
        node_lists = [e["nodes_np"] for e in events]
        np.cumsum(
            np.fromiter((x.shape[0] for x in node_lists), dtype=np.int32, count=n),
            out=c.node_off[1 : n + 1],
        )
        m = int(c.node_off[n])
        if m > c.nodes_flat.shape[0]:
            c.nodes_flat = np.empty(max(64, m), dtype=np.int32)
        c.nodes_flat[:m] = np.concatenate(node_lists).astype(np.int32, copy=False)
        c.m = m
    c.n = n
    return c


class SchedulingEnvEventNative(SchedulingEnvEventObs):
    """イベントネイティブなスケジューリング環境。

    - ジョブは到着順に1つずつ処理する
    - 配置可否は絶対時刻上のスケジュール済みイベントから判定する
    - クラウドは連続矩形のみ
    - オンプレの分散割当は、決めたノード集合をジョブ終了まで固定する
    """

    def reset(self):
        if self.multi_algorithm:
            self.jobs = np.asarray(self.jobs_set[self.episode])
        elif not self.job_is_constant:
            raise RuntimeError("event native env requires pre-generated jobs")

        self.time = 0
        self.index_next_job = 0
        self.step_count = 0
        self.done_flag = False
        self._defer_counts = {}  # 後回し回数(job_id→回数, defer時のみ使用)
        self.cost = 0
        self.total_cost = 0
        self.completed_jobs = 0
        self.jobs_processed_count = 0
        self.total_jobs_count = len(self.jobs)
        self.waiting_times = []
        self.job_allocated = []
        self._resource_events = {False: [], True: []}
        self._event_csr = {False: None, True: None}   # [PCN_EVENT_CSR_CACHE]
        self._absolute_schedule_records = []
        self._front_alloc_cache = None  # urgency が計算した先頭ジョブのオンプレ配置を次step で再利用
        # 残りジョブの最小到着時刻（suffix min）。これ未満で終わるイベントは未来の
        # 配置クエリ区間（start >= 未来到着）に重ならないので走査から外せる。
        if _FAST_ENV and len(self.jobs):
            arrivals = np.asarray(self.jobs, dtype=np.float64)[:, 0]
            suffix = np.minimum.accumulate(arrivals[::-1])[::-1]
            self._suffix_min_arrival = np.append(suffix, np.inf)
        else:
            self._suffix_min_arrival = None
        self._evt_prune_at = {False: 64, True: 64}  # 償却的 prune の発火閾値
        # 観測/agent が読む job_queue 先頭件数（[:5] + 余裕）。これだけ毎ステップ更新する。
        self._job_queue_fill_k = max(
            16,
            int(getattr(self, "n_job_queue_obs", 5)) + int(getattr(self, "n_job_queue_bck", 5)) + 2,
        )
        self.job_queue = np.zeros((len(self.jobs), 8))
        self.rear_job_queue = 0
        self._refresh_job_queue()

        if self._event_buf is not None:
            self._event_buf.reset()
        else:
            self._scheduled_events = []

        self.on_premise_window_history_full = None
        self.cloud_window_history_full = None

        # 報酬無次元化: この workload の達成レンジ[all-onprem待ち, all-cloud cost]を1回測りキャッシュ。
        if getattr(self, "_dimless_norm", False) and not getattr(self, "_computing_dimless", False):
            _jid = id(self.jobs)
            if _jid not in self._dimless_cache:
                self._dimless_cache[_jid] = self._compute_dimless_scale()
                return self.reset()  # 計測でepisode消費→クリーンに再reset(jidキャッシュ済で再計算せず)
            self._dimless_scale = self._dimless_cache[_jid]

        return self.get_observation()

    def step(self, action_raw):
        if self.index_next_job >= len(self.jobs):
            self.done_flag = True
            return self.get_observation(), np.zeros(2, dtype=np.float64), False, 0, True

        # 後回し(defer): SCHEDULER_ALLOW_DEFER=1 かつ action==2。現ジョブを配置せず後方へ回す。
        # 巨大ジョブを今は置かず、後続ジョブで資源が動いてから配置できる(=配置タイミングの自由度)。
        # 終了保証: 1ジョブの後回しは _defer_max 回まで。超えたら強制配置(onprem)にフォールバック。
        if getattr(self, "_allow_defer", False) and int(action_raw) == 2:
            row = self.jobs[self.index_next_job]
            jid = int(row[5]) if row.shape[0] > 5 else int(self.index_next_job)
            cnt = self._defer_counts.get(jid, 0)
            remaining = len(self.jobs) - self.index_next_job
            if cnt < self._defer_max and remaining > 1:
                self._defer_counts[jid] = cnt + 1
                self._defer_rotate(self.index_next_job)  # 現ジョブを offset 後ろへ
                self.step_count += 1
                self._refresh_job_queue()
                # 配置していない: scheduled=False・報酬0・index_next_job は不変(次ジョブが前に来る)
                return self.get_observation(), np.zeros(2, dtype=np.float64), False, 0, False
            # cap 到達 or 末尾 → 強制 onprem 配置にフォールバック(進行保証)
            action_raw = 0

        action = self.get_converted_action(action_raw)
        raw_job = self.jobs[self.index_next_job]
        job = self._to_queue_job(raw_job)
        arrival = int(raw_job[0])
        self.time = max(self.time, arrival)

        use_cloud = bool(action[1])
        # 直前の get_observation(urgency) が同じジョブのオンプレ配置を解いていれば再利用（events 不変=同一結果）。
        cache = getattr(self, "_front_alloc_cache", None)
        if (
            _REUSE_URGENCY_ALLOC
            and not use_cloud
            and cache is not None
            and cache[0] == self.index_next_job
            and cache[1] == arrival
        ):
            position, start_time = cache[2], cache[3]
        else:
            position, start_time = self._find_event_allocation(job, use_cloud, arrival)
        self._front_alloc_cache = None  # 配置で events が変化するため 1 回限り
        waiting_time = int(start_time - arrival)
        self.time = int(start_time)
        self._record_event_schedule(job, use_cloud, position, start_time)

        # 待ち指標: slowdown モードでは待ち÷処理時間(stretch)。env が報酬・目的値の両方で同じ量を
        # 返すので、学習・NSGA・eval が一括で同じ座標系になる。OFF(wait) では生 waiting_time。
        if self._wait_metric == "slowdown":
            wmetric = float(waiting_time) / max(1.0, float(raw_job[1]))
        else:
            wmetric = float(waiting_time)
        cost = self._compute_event_cost(job, use_cloud)
        # 報酬無次元化(PCN_DIMLESS_NORM): wmetric/cost を達成レンジ[all-onprem待ち, all-cloud cost]で割る。
        # 累積(total_cost/waiting_times)・目的値(calc_objective_values)・報酬・指令が全て一貫して無次元に。
        # 計測中(_computing_dimless)は生値。既定OFF(_dimless_scale=None)で不変=ビット一致。
        if self._dimless_scale is not None and not self._computing_dimless:
            wmetric = wmetric / float(self._dimless_scale[0])
            cost = cost / float(self._dimless_scale[1])
        self.waiting_times.append(wmetric)
        self.cost += cost
        self.total_cost = self.cost
        self.completed_jobs += 1
        self.jobs_processed_count += 1
        self.index_next_job += 1
        self.step_count += 1
        self._refresh_job_queue()

        done = self.index_next_job >= len(self.jobs)
        self.done_flag = done
        rewards = np.array([-wmetric, -cost], dtype=np.float64)  # wmetric/cost は上で無次元化済(dimless時)
        return self.get_observation(), rewards, True, waiting_time, done

    def step_with_start_hint(self, action_raw, start_time_hint):
        """開始時刻が既知(GPUカーネル等で事前計算済み)なときの step。

        通常の step() と全く同じ処理を行うが、配置探索(_find_event_allocation の候補時刻
        ループ)を「start=start_time_hint に対するノード選択1回」に置き換える(_find_event_
        allocation_at_hint)。配置規則は決定論なので、同じイベント表・同じ start に対しては
        通常経路と同一のノード選択・同一のイベント表・同一の観測になる(scripts/
        proto_gpu_sweep/verify_replay_obs.py で全ステップ一致を検証済み)。

        defer(action_raw==2)は対象外(呼び出し側は 0/1 のみ渡す前提。GPU rawカーネルも
        defer非対応)。start_time_hint が誤っている(その時刻で height 個のノードが空かない)
        場合は ValueError を送出する(検証で不一致を即座に検出するため)。
        既存の step() は一切変更していない(この決定論リプレイ専用の新規メソッド)。
        """
        if self.index_next_job >= len(self.jobs):
            self.done_flag = True
            return self.get_observation(), np.zeros(2, dtype=np.float64), False, 0, True

        action = self.get_converted_action(action_raw)
        raw_job = self.jobs[self.index_next_job]
        job = self._to_queue_job(raw_job)
        arrival = int(raw_job[0])
        self.time = max(self.time, arrival)

        use_cloud = bool(action[1])
        position, start_time = self._find_event_allocation_at_hint(
            job, use_cloud, arrival, start_time_hint
        )
        self._front_alloc_cache = None  # 配置で events が変化するため 1 回限り
        waiting_time = int(start_time - arrival)
        self.time = int(start_time)
        self._record_event_schedule(job, use_cloud, position, start_time)

        # 待ち指標: slowdown モードでは待ち÷処理時間(stretch)。step() と同一処理。
        if self._wait_metric == "slowdown":
            wmetric = float(waiting_time) / max(1.0, float(raw_job[1]))
        else:
            wmetric = float(waiting_time)
        cost = self._compute_event_cost(job, use_cloud)
        # 報酬無次元化(PCN_DIMLESS_NORM): step() と同一処理。
        if self._dimless_scale is not None and not self._computing_dimless:
            wmetric = wmetric / float(self._dimless_scale[0])
            cost = cost / float(self._dimless_scale[1])
        self.waiting_times.append(wmetric)
        self.cost += cost
        self.total_cost = self.cost
        self.completed_jobs += 1
        self.jobs_processed_count += 1
        self.index_next_job += 1
        self.step_count += 1
        self._refresh_job_queue()

        done = self.index_next_job >= len(self.jobs)
        self.done_flag = done
        rewards = np.array([-wmetric, -cost], dtype=np.float64)
        return self.get_observation(), rewards, True, waiting_time, done

    def _defer_rotate(self, i: int) -> None:
        """位置 i のジョブを offset 後ろへ回す(配置順の入れ替え)。defer 専用。
        self.jobs の順序が変わるため suffix_min を再計算する。"""
        j = min(i + self._defer_offset, len(self.jobs) - 1)
        if j <= i:
            return
        jobs = self.jobs
        row = jobs[i].copy()
        jobs = np.delete(jobs, i, axis=0)
        jobs = np.insert(jobs, j, row, axis=0)
        self.jobs = jobs
        if _FAST_ENV and len(self.jobs):
            arrivals = np.asarray(self.jobs, dtype=np.float64)[:, 0]
            suffix = np.minimum.accumulate(arrivals[::-1])[::-1]
            self._suffix_min_arrival = np.append(suffix, np.inf)

    def _to_queue_job(self, raw_job: np.ndarray) -> np.ndarray:
        arr = np.asarray(raw_job, dtype=float)
        if not _FAST_QUEUE_ROLL:
            return np.roll(arr, -1)
        # np.roll(arr,-1) と同一: result[i]=arr[(i+1)%n]（左1回転）。汎用前処理を省く。
        out = np.empty_like(arr)
        out[:-1] = arr[1:]
        out[-1] = arr[0]
        return out

    def _refresh_job_queue(self) -> None:
        if _FAST_ENV:
            # 観測が読むのは job_queue[:5]（しきい値agentは[0]）だけ。毎ステップ全 n_jobs 件を
            # 埋める O(n_jobs) を、読み出される先頭 K 件のみに限定して O(1) 化（結果不変）。
            # rear_job_queue は旧実装の len(next_jobs)=残ジョブ数を厳密に再現する。
            k = self._job_queue_fill_k
            self.job_queue[:k] = 0.0
            next_jobs = self.jobs[self.index_next_job : self.index_next_job + k]
            for i, raw_job in enumerate(next_jobs):
                self.job_queue[i] = self._to_queue_job(raw_job)
            self.rear_job_queue = max(0, len(self.jobs) - self.index_next_job)
            return
        self.job_queue.fill(0)
        next_jobs = self.jobs[self.index_next_job : self.index_next_job + len(self.job_queue)]
        for i, raw_job in enumerate(next_jobs):
            self.job_queue[i] = self._to_queue_job(raw_job)
        self.rear_job_queue = len(next_jobs)

    def _resource_event_list(self, use_cloud: bool) -> list[dict]:
        return self._resource_events[bool(use_cloud)]

    def _find_event_allocation(self, job: np.ndarray, use_cloud: bool, arrival: int):
        width = int(job[0])
        height = int(job[1])
        n_nodes = self.n_cloud_node if use_cloud else self.n_on_premise_node
        if width <= 0 or height <= 0 or height > n_nodes:
            raise ValueError(f"invalid job size for resource: width={width}, height={height}")

        events = self._resource_event_list(use_cloud)
        # 償却的に「未来の配置に二度と重ならない」過去イベントを間引く（結果不変）。
        if (
            _FAST_ENV
            and self._suffix_min_arrival is not None
            and len(events) >= self._evt_prune_at[use_cloud]
        ):
            thr = float(self._suffix_min_arrival[self.index_next_job])
            events[:] = [e for e in events if e["end"] >= thr]
            self._evt_prune_at[use_cloud] = max(64, 2 * len(events) + 32)
            # [PCN_EVENT_CSR_CACHE] 表が縮んだので持ち回りは破棄(次の探索で1度だけ組み直す)
            _csr = getattr(self, "_event_csr", None)
            if _csr is not None:
                _csr[bool(use_cloud)] = None

        # イベントが多いときだけ sweep-line（O(R log R + R·H)）。少数なら候補ループ(fused)。結果は同一。
        if _FAST_ENV_SWEEP and len(events) >= _SWEEP_MIN_EVENTS:
            if _SWEEP_C:
                return self._find_event_allocation_sweep_c(
                    events, width, height, n_nodes, use_cloud, arrival
                )
            if _FAST_ENV_NP:
                return self._find_event_allocation_sweep_np(
                    events, width, height, n_nodes, use_cloud, arrival
                )
            return self._find_event_allocation_sweep(
                events, width, height, n_nodes, use_cloud, arrival
            )

        candidates = {int(arrival)}
        for event in events:
            if event["end"] >= arrival:
                candidates.add(int(event["end"]))

        while candidates:
            start = min(candidates)
            candidates.remove(start)
            nodes, min_overlap_end = self._find_nodes_free_for_interval(
                events, start, start + width, height, n_nodes, continuous_only=use_cloud
            )
            if nodes is not None:
                if self._is_contiguous(nodes):
                    return (nodes[0], 0), start
                node_cols = [nodes] * width
                return (0, 0, node_cols), start

            if _FAST_ENV:
                # fused: occupied 走査中に得た min_overlap_end を次候補に（旧 min(overlapping_ends) と同値）。
                next_end = min_overlap_end
            else:
                overlapping_ends = [
                    int(event["end"])
                    for event in events
                    if event["start"] < start + width and event["end"] > start and event["end"] > start
                ]
                next_end = min(overlapping_ends) if overlapping_ends else None
            if next_end is None:
                # 現在時刻で入らないが重複イベントがない場合は、次候補がない
                # （サイズ不正以外では通常ここには来ない）
                break
            candidates.add(int(next_end))

        # すべての既存イベント終了後なら必ず空く
        start = max([arrival] + [int(event["end"]) for event in events])
        nodes = list(range(height))
        return (nodes[0], 0), start

    def _find_event_allocation_at_hint(
        self, job: np.ndarray, use_cloud: bool, arrival: int, start_hint
    ):
        """start(開始時刻)が既知のときの配置クエリ(決定論リプレイ用)。

        既存の _find_event_allocation を arrival=start_hint で呼ぶだけ。候補列は
        {start_hint} ∪ {end >= start_hint} の昇順なので、正しい hint なら先頭候補
        start_hint が即 feasible となり 1 候補で確定する。占有判定・pick 規則・prune は
        通常経路と完全に同一実装(C sweep / sweep_np / fused を含む)を通るため、
        返り値 (position, start) は通常経路とビット一致で、候補時刻ループの反復
        (通常経路の計算の大半)だけが省かれる。hint が誤り(その時刻に height 個の
        ノードが空かない)なら、より遅い開始時刻が返るので検出して ValueError
        (検証で不一致を即座に見つけるための契約)。arrival は待ち時間計算に使う
        実到着時刻(呼び出し側が持つ)で、ここでは診断メッセージにのみ使う。
        """
        position, start_time = self._find_event_allocation(job, use_cloud, int(start_hint))
        if int(start_time) != int(start_hint):
            raise ValueError(
                f"start_time_hint infeasible: hint={int(start_hint)} -> feasible "
                f"start={int(start_time)} (arrival={int(arrival)}, width={int(job[0])}, "
                f"height={int(job[1])}, use_cloud={bool(use_cloud)})"
            )
        return position, start_time

    def _find_nodes_free_for_interval(
        self,
        events: list[dict],
        start: int,
        end: int,
        height: int,
        n_nodes: int,
        continuous_only: bool,
    ):
        """戻り値 (nodes_or_None, min_overlap_end_or_None)。

        fused: occupied を作る同じ走査で、区間に重なるイベントの最小 end も取得して返す
        （_FAST_ENV のときのみ。呼び出し側が2回目の overlapping_ends 走査を省ける）。
        旧来の overlapping_ends 述語と同一集合の min なので値は一致。レガシー経路は None を返し、
        呼び出し側が従来どおり別走査する（バイト一致のベースライン維持）。
        """
        occupied = set()
        min_overlap_end = None
        track_moe = _FAST_ENV
        for event in events:
            if event["start"] < end and event["end"] > start:
                occupied.update(event["nodes"])
                if track_moe:
                    ee = int(event["end"])
                    if min_overlap_end is None or ee < min_overlap_end:
                        min_overlap_end = ee

        if not _FAST_ENV:
            free = [node for node in range(n_nodes) if node not in occupied]
            if len(free) < height:
                return None, None

            run: list[int] = []
            prev = None
            for node in free:
                if prev is None or node == prev + 1:
                    run.append(node)
                else:
                    run = [node]
                if len(run) >= height:
                    return run[:height], None
                prev = node

            if continuous_only:
                return None, None
            return free[:height], None

        # fast: range(n_nodes) を 1 パス。最初の height 連続 free run を見つけたら即 return
        # （free 全列挙 O(n_nodes) を回避）。値は旧実装と同一（昇順最初の run / 非連続は free[:height]）。
        if n_nodes - len(occupied) < height:
            return None, min_overlap_end
        run_start = -1
        free_head: list[int] = []
        need_head = not continuous_only
        for node in range(n_nodes):
            if node in occupied:
                run_start = -1
                continue
            if run_start < 0:
                run_start = node
            if node - run_start + 1 >= height:
                return list(range(run_start, run_start + height)), min_overlap_end
            if need_head and len(free_head) < height:
                free_head.append(node)
        if continuous_only:
            return None, min_overlap_end
        return free_head[:height], min_overlap_end

    def _pick_free_from_count(self, count, free_count, height, n_nodes, continuous_only):
        """count[k]==0 ⇔ node 空き、として height 個を選ぶ（_find_nodes_free_for_interval fast と同値）。

        count は「区間に重なるイベントが node k を占有する個数」。占有集合が同じなら選択も同一。
        """
        if free_count < height:
            return None
        run_start = -1
        free_head: list[int] = []
        need_head = not continuous_only
        for node in range(n_nodes):
            if count[node] != 0:
                run_start = -1
                continue
            if run_start < 0:
                run_start = node
            if node - run_start + 1 >= height:
                return list(range(run_start, run_start + height))
            if need_head and len(free_head) < height:
                free_head.append(node)
        if continuous_only:
            return None
        return free_head[:height]

    def _pick_free_from_count_np(self, count, free_count, height, n_nodes, continuous_only):
        """_pick_free_from_count の numpy 版（free run 探索をベクトル化, 返り値はビット一致）。

        free_count<height の早期 return は高競合時に下の連続 run 探索（配列確保+diff）を丸ごと
        省くための要の最適化。これを外すと pick が毎回フル実行され全体が大きく遅くなる（実測 1.7x 遅化）。
        """
        if free_count < height:
            return None
        free = count == 0
        # 連続 free run の境界を一括検出。padded の diff 非零位置が各 run の開始/終了端。
        padded = np.empty(n_nodes + 2, dtype=np.int8)
        padded[0] = 0
        padded[-1] = 0
        padded[1:-1] = free
        bounds = np.flatnonzero(np.diff(padded))
        starts = bounds[0::2]
        ok = np.flatnonzero((bounds[1::2] - starts) >= height)
        if ok.size:
            s = int(starts[ok[0]])
            return list(range(s, s + height))
        if continuous_only:
            return None
        return np.flatnonzero(free)[:height].tolist()

    def _find_event_allocation_sweep(self, events, width, height, n_nodes, use_cloud, arrival):
        """候補時刻を昇順に sweep し、per-node 重なりカウントを増分維持して最初の feasible を返す。

        候補集合 = {arrival} ∪ {end>=arrival のイベント end}（旧ループが実質訪問する集合と同一・昇順）。
        各候補 start で区間 [start, start+width) に重なるイベント集合を count に反映し、空きノードを選ぶ。
        旧 `_find_nodes_free_for_interval` と同じ占有集合・同じ選択になるので (position, start) はビット一致。
        """
        cand = {int(arrival)}
        for e in events:
            if e["end"] >= arrival:
                cand.add(int(e["end"]))
        cand_sorted = sorted(cand)

        n = len(events)
        order_start = sorted(range(n), key=lambda i: events[i]["start"])
        order_end = sorted(range(n), key=lambda i: events[i]["end"])
        counted = [False] * n          # event i が現在 count に寄与しているか
        count = [0] * n_nodes          # node k を占有する「区間内」イベント数
        free_count = n_nodes
        i_en = 0
        i_ex = 0

        for start in cand_sorted:
            win_end = start + width
            # ENTER: start_e < win_end のイベントを取り込み（end_e > start の生存分のみ計数）。単調前進。
            while i_en < n and events[order_start[i_en]]["start"] < win_end:
                idx = order_start[i_en]
                e = events[idx]
                if e["end"] > start:
                    for k in e["nodes"]:
                        if count[k] == 0:
                            free_count -= 1
                        count[k] += 1
                    counted[idx] = True
                i_en += 1
            # EXIT: end_e <= start のイベントを除去。単調前進。
            while i_ex < n and events[order_end[i_ex]]["end"] <= start:
                idx = order_end[i_ex]
                if counted[idx]:
                    for k in events[idx]["nodes"]:
                        count[k] -= 1
                        if count[k] == 0:
                            free_count += 1
                    counted[idx] = False
                i_ex += 1
            nodes = self._pick_free_from_count(count, free_count, height, n_nodes, use_cloud)
            if nodes is not None:
                if self._is_contiguous(nodes):
                    return (nodes[0], 0), start
                node_cols = [nodes] * width
                return (0, 0, node_cols), start

        # すべての既存イベント終了後なら必ず空く（候補が尽きた場合の保険; 通常は最大 end の候補で成功）
        start = max([arrival] + [int(e["end"]) for e in events])
        nodes = list(range(height))
        return (nodes[0], 0), start

    def _find_event_allocation_sweep_np(self, events, width, height, n_nodes, use_cloud, arrival):
        """_find_event_allocation_sweep の numpy 占有カウント版（ENTER/EXIT/pick から Python ループ除去）。

        占有カウント count を np.int32 配列で持ち、各イベントの固定ノード配列 nodes_np で
        ベクトル加減算する。候補列・訪問順・free run 選択は旧 sweep と同一なのでビット一致。
        """
        cand = {int(arrival)}
        for e in events:
            if e["end"] >= arrival:
                cand.add(int(e["end"]))
        cand_sorted = sorted(cand)

        n = len(events)
        order_start = sorted(range(n), key=lambda i: events[i]["start"])
        order_end = sorted(range(n), key=lambda i: events[i]["end"])
        counted = [False] * n
        count = np.zeros(n_nodes, dtype=np.int32)
        free_count = n_nodes
        i_en = 0
        i_ex = 0

        for start in cand_sorted:
            win_end = start + width
            # ENTER: start_e < win_end かつ生存(end_e > start)のイベントを取り込む。単調前進。
            while i_en < n and events[order_start[i_en]]["start"] < win_end:
                idx = order_start[i_en]
                e = events[idx]
                if e["end"] > start:
                    nd = e["nodes_np"]
                    sub = count[nd]
                    free_count -= int(np.count_nonzero(sub == 0))
                    count[nd] = sub + 1
                    counted[idx] = True
                i_en += 1
            # EXIT: end_e <= start のイベントを除去。単調前進。
            while i_ex < n and events[order_end[i_ex]]["end"] <= start:
                idx = order_end[i_ex]
                if counted[idx]:
                    nd = events[idx]["nodes_np"]
                    sub = count[nd] - 1
                    count[nd] = sub
                    free_count += int(np.count_nonzero(sub == 0))
                    counted[idx] = False
                i_ex += 1
            nodes = self._pick_free_from_count_np(count, free_count, height, n_nodes, use_cloud)
            if nodes is not None:
                if self._is_contiguous(nodes):
                    return (nodes[0], 0), start
                node_cols = [nodes] * width
                return (0, 0, node_cols), start

        # すべての既存イベント終了後なら必ず空く（保険; 通常は最大 end の候補で成功）
        start = max([arrival] + [int(e["end"]) for e in events])
        nodes = list(range(height))
        return (nodes[0], 0), start

    def _find_event_allocation_sweep_c(self, events, width, height, n_nodes, use_cloud, arrival):
        """_find_event_allocation_sweep_np の C 実装版（scheduling_env_core.event_sweep_alloc）。

        Python 側は events から CSR(starts/ends/nodes_flat/node_off)を1度組むだけ。候補列・占有 count・
        pick・フォールバックは全て C 側で行う。占有集合・訪問順・free run 選択は sweep_np と同一なので
        返り値 (position, start) はビット一致。
        """
        n = len(events)
        if _CSR_CACHE:
            # 持ち回りの CSR(追加のたびに末尾へ追記済み)をそのまま使う。行数が食い違ったときだけ
            # 組み直す(安全弁: 表を触る経路が増えても結果が壊れない)。
            _csr = getattr(self, "_event_csr", None)
            if _csr is None:
                _csr = self._event_csr = {False: None, True: None}
            c = _csr[bool(use_cloud)]
            if c is None or c.n != n:
                c = _csr_from_events(events)
                _csr[bool(use_cloud)] = c
            starts = c.starts[:n]
            ends = c.ends[:n]
            node_off = c.node_off[: n + 1]
            nodes_flat = c.nodes_flat[: c.m]
            if _CSR_VERIFY:
                ref = _csr_from_events(events)
                assert np.array_equal(starts, ref.starts[:n]), "CSR starts mismatch"
                assert np.array_equal(ends, ref.ends[:n]), "CSR ends mismatch"
                assert np.array_equal(node_off, ref.node_off[: n + 1]), "CSR node_off mismatch"
                assert np.array_equal(nodes_flat, ref.nodes_flat[: ref.m]), "CSR nodes mismatch"
        else:
            starts = np.fromiter((e["start"] for e in events), dtype=np.int64, count=n)
            ends = np.fromiter((e["end"] for e in events), dtype=np.int64, count=n)
            node_lists = [e["nodes_np"] for e in events]
            node_off = np.empty(n + 1, dtype=np.int32)
            node_off[0] = 0
            if n:
                np.cumsum(
                    np.fromiter((x.shape[0] for x in node_lists), dtype=np.int32, count=n),
                    out=node_off[1:],
                )
                nodes_flat = np.concatenate(node_lists).astype(np.int32, copy=False)
            else:
                nodes_flat = np.empty(0, dtype=np.int32)

        start, is_contig, nodes = _c_event_sweep_alloc(
            starts, ends, nodes_flat, node_off,
            int(width), int(height), int(n_nodes), bool(use_cloud), int(arrival),
        )
        if is_contig:
            return (nodes[0], 0), int(start)
        node_cols = [nodes] * width
        return (0, 0, node_cols), int(start)

    def _is_contiguous(self, nodes: list[int]) -> bool:
        return all(nodes[i] + 1 == nodes[i + 1] for i in range(len(nodes) - 1))

    def _record_event_schedule(self, job: np.ndarray, use_cloud: bool, position, start_time: int) -> None:
        width = int(job[0])
        height = int(job[1])
        job_id = int(job[4])
        end_time = int(start_time + width)
        if len(position) == 3:
            node_cols = position[2]
            nodes = list(node_cols[0])
            start_node = min(nodes)
        else:
            start_node = int(position[0])
            nodes = list(range(start_node, start_node + height))
            node_cols = None

        record = {
            "job_id": job_id,
            "start": int(start_time),
            "end": end_time,
            "width": width,
            "height": height,
            "use_cloud": bool(use_cloud),
            "start_node": int(start_node),
            "node_cols": node_cols,
            "nodes": nodes,
            # sweep(np) が占有カウント更新で参照する固定ノード配列（生成時に1回だけ確保）。
            "nodes_np": np.asarray(nodes, dtype=np.intp),
        }
        self._resource_event_list(use_cloud).append(record)
        self._absolute_schedule_records.append(record)
        # [PCN_EVENT_CSR_CACHE] 持ち回りの CSR にも同じ順序で追記する(未構築なら何もしない)
        if _CSR_CACHE:
            _csr = getattr(self, "_event_csr", None)
            _c = None if _csr is None else _csr[bool(use_cloud)]
            if _c is not None:
                _c.append(int(start_time), int(end_time), record["nodes_np"])

        if self._event_buf is not None:
            self._event_buf.append(
                float(start_time),
                float(end_time),
                float(width),
                float(use_cloud),
                float(start_node),
                float(height),
            )
        else:
            self._scheduled_events.append(
                (start_time, end_time, width, float(use_cloud), start_node, height)
            )

    def _compute_event_cost(self, job: np.ndarray, use_cloud: bool) -> int:
        if not use_cloud:
            return 0
        return int(job[0]) * int(job[1])

    def build_schedule_maps(self) -> Tuple[np.ndarray, np.ndarray]:
        """要求時のみ: スケジュール済みイベントからノード×時刻の job_id 行列を構築。"""
        max_end = 1
        if self._absolute_schedule_records:
            max_end = max(int(r["end"]) for r in self._absolute_schedule_records)
        cols = max(1, max_end)
        onpre = np.zeros((self.n_on_premise_node, cols), dtype=np.int32)
        cloud = np.zeros((self.n_cloud_node, cols), dtype=np.int32)
        for rec in self._absolute_schedule_records:
            mat = cloud if rec["use_cloud"] else onpre
            start, end = int(rec["start"]), int(rec["end"])
            job_id = int(rec["job_id"])
            for node in rec["nodes"]:
                if 0 <= node < mat.shape[0]:
                    mat[node, start:end] = job_id
        self.on_premise_window_history_full = onpre
        self.cloud_window_history_full = cloud
        return onpre, cloud

    def finalize_window_history(self, build_maps: Optional[bool] = None):
        """エピソード終了処理。既定では指標のみ更新し、マップ行列は構築しない。"""
        cost, _, _ = self.calc_objective_values(
            calc_makespan=False, calc_avg_waiting_time=False
        )
        self.cost = cost
        if schedule_maps_enabled(build_maps):
            return self.build_schedule_maps()
        self.on_premise_window_history_full = None
        self.cloud_window_history_full = None
        return None

    def calc_objective_values(self, calc_makespan=True, calc_avg_waiting_time=True):
        makespan = -1
        if calc_makespan:
            if self._absolute_schedule_records:
                makespan = max(int(r["end"]) for r in self._absolute_schedule_records)
            else:
                makespan = 0
        avg_wait = float(np.mean(self.waiting_times)) if (calc_avg_waiting_time and self.waiting_times) else 0.0
        return self.cost, makespan, avg_wait
