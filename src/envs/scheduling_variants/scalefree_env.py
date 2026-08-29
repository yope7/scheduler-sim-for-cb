"""スケールフリー・逐次運用向け スケジューリング env(新規設計・2026-07-11).

狙い: 育てた方策(重み)を、学習と違うジョブ数N・容量C・時間スケールの実運用でも
そのまま使える(逐次に1ジョブずつ捌ける)env。既存 event_native_env の良い設計は
踏襲し、スケール依存・全知(未来を知る)量を排除する。

踏襲(車輪の再発明を避ける):
  - イベント駆動: 絶対時刻上の scheduled events(start,end,height)で資源を管理
  - sweep-line 区間占有配置: オンプレは「幅wの区間[t,t+w)で空きノード数>=h」の最早tに置く
    (オンプレは非連続配置OK=どのノードかは待ちにほぼ無関係=count-only近似で平均待ち0.3-3%)
  - urgency: 現ジョブのオンプレ予測待ちを log 正規化した局所観測
  - 報酬 [-wait, -cost]: PCN の desired_return 正規化と接続

新規(スケールフリー・逐次):
  - 観測は固定次元・全て [0,1]/[-1,1] のスケールフリー座標(N・C・時間に不変)
  - 全知量(全ジョブ占有量ソート順位)を排除 → 負荷寄与率 a1(jobs[i]のみで逐次計算可)
  - 配置は count-based 近似(node identity を無視)。ビット一致は捨てる。prune で実効 O(N)。

ジョブ列(JobGenerator 準拠, shape (N,>=6)):
  raw[0]=arrival(到着), raw[1]=width(実行時間 pt), raw[2]=height(必要ノード数)
  cost(cloud) = width*height  (既存 _compute_event_cost と同じ)
"""
from __future__ import annotations

import numpy as np
import gym


class _SpecStub:
    """env.unwrapped.spec.id 互換の最小 spec。"""
    def __init__(self, id_: str = "ScaleFreeScheduling-v0"):
        self.id = id_


class ScaleFreeSchedulingEnv(gym.Env):
    # 観測ブロック: global(3) + cur_job(4) + queue(K*3) + events(M*5)
    metadata = {"render.modes": []}
    spec = _SpecStub()
    N_GLOBAL = 3
    N_CUR = 4
    Q_FEAT = 3
    E_FEAT = 5

    def __init__(
        self,
        jobs,
        n_on_premise_node: int,
        n_cloud_node: int,
        n_window: int = 100,
        k_queue: int = 5,
        m_events: int = 16,
        urgency_log_k: float = 16.0,
    ):
        self.jobs = np.asarray(jobs, dtype=np.float64)
        self.n_on = int(n_on_premise_node)
        self.n_cl = int(n_cloud_node)
        # 学習パイプライン(distributed_pcn)が読む別名。n_on/n_cl と同一値。
        self.n_on_premise_node = self.n_on
        self.n_cloud_node = self.n_cl
        self.n_window = int(n_window)
        self.k_queue = int(k_queue)
        self.m_events = int(m_events)
        self.urgency_log_k = float(urgency_log_k)
        self.n_jobs = int(len(self.jobs))

        # --- スケールフリー正規化基準(全て学習/運用で同じ座標を作るための固定量) ---
        self._norm_time = max(1.0, float(self.n_window))   # 時間スケール = 窓幅
        self._cap = max(1.0, float(self.n_on))             # 容量 = オンプレノード数
        if self.n_jobs > 1:
            self._T = max(1.0, float(self.jobs[:, 0].max() - self.jobs[:, 0].min()))
        else:
            self._T = 1.0

        self.obs_dim = (
            self.N_GLOBAL + self.N_CUR + self.k_queue * self.Q_FEAT + self.m_events * self.E_FEAT
        )

        # 学習パイプライン互換の gym Space。state_dim は observation_space.shape[0] から自動取得される。
        from gym import spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)          # 0=onprem, 1=cloud
        self.reward_space = spaces.Box(
            low=-np.inf, high=0.0, shape=(2,), dtype=np.float32
        )
        # multi_algorithm 経路(jobs_set[episode])の最小互換。単一ジョブ列を episode0 に。
        self.jobs_set = {0: self.jobs}
        self.episode = 0
        self.multi_algorithm = False
        self.job_is_constant = True

        self.reset()

    def finalize_window_history(self, build_maps=None):
        """スケジュールマップ復元はこの env では未対応(スケールフリー観測が本体)。互換のため no-op。"""
        return None

    def _front_job_leverage(self) -> float:
        """現ジョブの負荷寄与 a1(観測 obs[6] と同じ)。パイプライン互換のため公開。"""
        i = self.index_next_job
        if i >= self.n_jobs:
            return 0.0
        raw = self.jobs[i]
        occ_frac = (float(raw[1]) / self._norm_time) * (float(raw[2]) / self._cap)
        return float(np.clip(np.log1p(occ_frac * 1.0e3) / 4.3, 0.0, 1.0))

    # ------------------------------------------------------------------ core
    def reset(self):
        self.time = 0
        self.index_next_job = 0
        self.cost = 0
        self.waiting_times: list[float] = []
        self._placed: list[tuple[int, int, int]] = []  # (start, end, height) onプレ予約のみ
        self.done_flag = False
        return self._observe()

    def step(self, action):
        """action: 0=onprem, 1=cloud。戻り値 (obs, reward[-wait,-cost], scheduled, wait, done)。"""
        if self.index_next_job >= self.n_jobs:
            self.done_flag = True
            return self._observe(), np.zeros(2, dtype=np.float64), False, 0, True

        raw = self.jobs[self.index_next_job]
        arrival = int(raw[0]); w = int(raw[1]); h = int(raw[2])
        use_cloud = (int(action) == 1)

        if use_cloud:
            wait = 0
            job_cost = w * h
            self.cost += job_cost
        else:
            start = self._place_onprem(w, h, arrival, commit=True)
            wait = max(0, start - arrival)
            job_cost = 0

        self.waiting_times.append(float(wait))
        self.time = max(self.time, arrival)
        self.index_next_job += 1
        done = self.index_next_job >= self.n_jobs
        self.done_flag = done
        reward = np.array([-float(wait), -float(job_cost)], dtype=np.float64)
        return self._observe(), reward, True, wait, done

    # ------------------------------------------------------- 配置(count sweep)
    def _place_onprem(self, w: int, h: int, arrival: int, commit: bool) -> int:
        """幅wの区間[t,t+w)で空きノード数>=h になる最早 t>=arrival を返す(count-only近似)。
        commit=True なら予約を確定(_placed に追加)。prune: 区間に二度と重ならない予約を落とす。"""
        if self._placed:
            # end<=arrival の予約は今後の区間[t>=arrival,...)に重ならない → 落とす(結果不変・O(R)償却)
            self._placed = [e for e in self._placed if e[1] > arrival]
        placed = self._placed
        # 候補時刻 = {arrival} ∪ {arrival以降に終わる予約の end}(空きが増える瞬間)
        cands = {arrival}
        for (s, e, hh) in placed:
            if e >= arrival:
                cands.add(e)
        chosen = None
        for t in sorted(cands):
            end = t + w
            # 区間[t,end)の最大同時占有を差分掃引で求める
            evs = []
            for (s, e, hh) in placed:
                if s < end and e > t:
                    evs.append((max(s, t), hh))
                    evs.append((min(e, end), -hh))
            evs.sort()
            occ = 0; mx = 0
            for (_, d) in evs:
                occ += d
                if occ > mx:
                    mx = occ
            if self.n_on - mx >= h:
                chosen = t
                break
        if chosen is None:
            chosen = max([arrival] + [e for (_, e, _) in placed])
        if commit:
            self._placed.append((chosen, chosen + w, h))
        return chosen

    # ---------------------------------------------------------------- 観測
    def _observe(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        i = self.index_next_job
        now = self.time

        # --- global(3): スケールフリー ---
        live_occ = 0
        for (s, e, hh) in self._placed:
            if s <= now < e:
                live_occ += hh
        obs[0] = min(1.0, live_occ / self._cap)                 # 現在混雑度 ρ_now
        obs[1] = i / max(1, self.n_jobs)                        # 進捗
        if self.waiting_times:
            recent = self.waiting_times[-10:]
            obs[2] = min(1.0, (sum(recent) / len(recent)) / self._norm_time)  # 直近バックログ圧

        # --- 現ジョブ(4) ---
        if i < self.n_jobs:
            raw = self.jobs[i]; w = float(raw[1]); h = float(raw[2]); arr = int(raw[0])
            obs[3] = min(1.0, w / self._norm_time)
            obs[4] = min(1.0, h / self._cap)
            start = self._place_onprem(int(w), int(h), arr, commit=False)  # 副作用なし予測
            obs[5] = float(np.clip(np.log1p(max(0, start - arr)) / self.urgency_log_k, 0.0, 1.0))
            occ_frac = (w / self._norm_time) * (h / self._cap)
            obs[6] = float(np.clip(np.log1p(occ_frac * 1.0e3) / 4.3, 0.0, 1.0))  # 負荷寄与 a1

        # --- job_queue 先頭K件(3K): 未来の近傍だけ(全知でない) ---
        base = self.N_GLOBAL + self.N_CUR
        for k in range(self.k_queue):
            jj = i + k
            if jj < self.n_jobs:
                raw = self.jobs[jj]
                o = base + k * self.Q_FEAT
                obs[o + 0] = min(1.0, float(raw[1]) / self._norm_time)
                obs[o + 1] = min(1.0, float(raw[2]) / self._cap)
                obs[o + 2] = float(np.clip((float(raw[0]) - now) / self._norm_time, -1.0, 1.0))

        # --- イベント窓 直近M件(5M) ---
        base2 = base + self.k_queue * self.Q_FEAT
        ws = max(0, now - self.n_window)
        recent_ev = [ev for ev in self._placed if ev[1] >= ws][-self.m_events:]
        for m, (s, e, hh) in enumerate(recent_ev):
            o = base2 + m * self.E_FEAT
            obs[o + 0] = float(np.clip((s - ws) / self._norm_time, 0.0, 1.0))
            obs[o + 1] = float(np.clip((e - ws) / self._norm_time, 0.0, 1.0))
            obs[o + 2] = float(np.clip((e - s) / self._norm_time, 0.0, 1.0))
            obs[o + 3] = 0.0  # use_cloud(オンプレ予約のみ記録なので0)
            obs[o + 4] = min(1.0, hh / self._cap)
        return obs

    # ---------------------------------------------------------------- objective
    def calc_objective_values(self, calc_makespan=True, calc_avg_waiting_time=True):
        avg_wait = float(np.mean(self.waiting_times)) if self.waiting_times else 0.0
        makespan = max([e for (_, e, _) in self._placed], default=0)
        return self.cost, makespan, avg_wait
