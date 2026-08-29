"""
イベントベース観測のスケジューリング環境

ビットマップ（ウィンドウ占有状態）を観測から撤廃し、
スケジュール済みイベントの開始時間・終了時間・継続時間・配置位置から学習する。

- 内部のスケジューリングロジックは SchedulingEnvCacheOptimized と同一（C実装使用）
- 観測をイベント形式に変更: (start_time, end_time, duration, use_cloud, start_node, job_height) × N + job_queue
- start_node, job_height によりマップの復元が可能
- イベント履歴は ``scheduling_env_core.SchedulingEventBuffer``（C メモリ）に保持し、get_observation で numpy 経由のコピーを避ける
"""
import os

import numpy as np
from gym import spaces

from .bitmap_c_env import SchedulingEnvCacheOptimized

try:
    from scheduling_env_core import (
        SchedulingEventBuffer as _CSchedulingEventBuffer,
        get_observation_event as _c_get_observation_event,
    )
except ImportError:
    _CSchedulingEventBuffer = None
    _c_get_observation_event = None

# 観測に含める最大イベント数（オンプレ+クラウド合計）
N_EVENTS_OBS = 30
# 各イベントの属性数: start_time, end_time, duration, use_cloud, start_node, job_height
EVENT_FEATURES = 6
# ジョブキュー: 5ジョブ × 8属性
JOB_QUEUE_SIZE = 5 * 8

# [SCHEDULER_OBS_EFFICIENCY] 現ジョブの「雲送り効率」3特徴の log 正規化幅。
# 効率(待ち削減÷追加コスト)はジョブ間で5桁開くので log 必須。
EFF_COST_K = 20.0    # log1p(追加コスト = pt*nodes) / K。pt*nodes は 1e0〜1e8 ⇒ log1p 0〜18.4
# ⚠️ [2026-08-30] この「urgency と同一に揃える」較正の結果、log1p(b)/K は urgency と
#    完全に同値になっている。クラウドは弾性で s_cl=到着時刻 が常に成立するため
#    b = s_on - arrival = urgency の中身そのもの(実測 6000step でビット一致)。
#    さらに efficiency 3次元は fast100 の iter020/iter100 で行動を1ステップも変えて
#    いない(6指令×5万step の反実仮想で 0.001-0.011%)。既定運用では OFF にした
#    (scripts/run_j50000_v10.sh)。ここの定数を変えると 224次元の既存 checkpoint が
#    再現できなくなるので、較正そのものは触らない。
EFF_GAIN_K = 16.0    # log1p(待ち削減秒) / K。urgency の正規化幅と同一に揃える
EFF_RATIO_K = 40.0   # (log1p(gain)-log1p(cost))/K + 0.5。log比 ±20 を [0,1] に(中心 0.5)
EFF_EXTRA_DIM = 3

# [SCHEDULER_OBS_BUDGET_RATIO] 現ジョブの「残り予算との比」3特徴の log 正規化幅。
# 効率特徴(待ち削減/追加コストの絶対値)には「残り予算」という基準が無く、方策は
# 「このジョブは高いか安いか」を予算スケールで判断できない(18ジョブ全列挙の診断で
# 既存効率特徴の寄与率2.7%=ほぼ死んでいた)。真の前線は同じ雲送り件数kの中でも
# 「このジョブの値段 vs 残り予算」で選ぶ大物が入れ替わる。ここでは以下だけを使う:
#   a       = pt*nodes(現ジョブの雲送りコスト。効率特徴のaと同一・答え非依存の既知量)
#   budget  = 残り予算(agent側rolloutがset_remaining_budgetで都度反映する
#             desired_return の cost 成分。envはG規約/anti-ration等の意味を知らない)
#   rem     = 残りジョブ(現在位置=自分を含めて末尾まで、この窓のジョブ集合から実行時に
#             計算可能。未来の実測アウトカムは一切含まない)の雲コスト合計
#   BR1 = log1p(a / max(budget,1)) / K1        … このジョブは残り予算の何割を食うか
#   BR2 = log1p(a / max(rem,1)) / K2            … 残りの仕事の中でこのジョブはどれだけ大きいか
#   BR3 = log1p(budget / max(rem,1)) / K3       … 残り予算は残りの仕事をどれだけ賄えるか
# 既定 OFF でビット不変。答え(真PF)を参照する要素は一切含まない。
BUD_A_K = 20.0        # a/budget は budget→1 付近で最大 1e8 級 ⇒ log1p 0-18.4 (EFF_COST_K と同一較正)
BUD_FRAC_K = 0.6931471805599453  # =ln2。a<=rem(自分含む合計)なので比は(0,1]⇒log1pは(0,ln2]
BUD_COVER_K = 20.0    # budget/rem も同様に大きく開き得る(EFF_COST_K と同一較正)
BUDGET_EXTRA_DIM = 3


class SchedulingEnvEventObs(SchedulingEnvCacheOptimized):
    """
    イベントベース観測のスケジューリング環境
    
    観測: スケジュール済みイベント(start, end, duration, use_cloud, start_node, job_height) + ジョブキュー
    ビットマップは使用しない。start_node, job_height によりマップ復元が可能。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._absolute_schedule_records = []
        if _CSchedulingEventBuffer is not None:
            self._event_buf = _CSchedulingEventBuffer(4096)
            self._scheduled_events = None
        else:
            self._event_buf = None
            self._scheduled_events = []
        self._obs_events_size = N_EVENTS_OBS * EVENT_FEATURES
        # オプトイン: 現ジョブのオンプレ予測待ち(緊急度)を1次元追加（front-loading対策）
        self._obs_urgency = os.environ.get("SCHEDULER_OBS_URGENCY", "0") == "1"
        # オプトイン: 現ジョブの占有量(pt*nodes)のインスタンス内パーセンタイル順位を1次元追加。
        # trace は少数の巨大ジョブ(giant)の配置が高レバレッジ＝報酬地形の崖。順位は「これは
        # top-k の巨大ジョブか」を相対的に符号化し、決定的なジョブをノイズから分離できるようにする。
        self._obs_occupancy = os.environ.get("SCHEDULER_OBS_OCCUPANCY", "0") == "1"
        self._occ_sorted = None  # reset 時に self.jobs から占有量ソート列をキャッシュ
        self._occ_jobs_id = None
        # オプトイン: job_queue 観測(先頭5ジョブ×8列)を「スケールフリー座標」に正規化。
        # 現状 job_queue は _to_queue_job(raw)=roll(raw,-1) の生値で、col0=width(pt,絶対時間)/
        # col1=height(nodes,容量依存)/col7=arrival(絶対時刻) がスケール依存 → N・容量が変わると
        # 観測分布が動き、育てた重みが運用(学習≠運用のN/分布)で効かない。明確に絶対値な3列だけを
        # obs_events と同じ座標系(/_norm_time, /_norm_nodes, 現在時刻相対)へ写す。他列は据え置き。
        # 既定 OFF でビット一致。ON で逐次運用に耐える観測になる。
        self._obs_jq_norm = os.environ.get("SCHEDULER_OBS_JQ_NORM", "0") == "1"
        # オプトイン: leverage(占有量)を「全ジョブ順位(=未来を知る全知)」でなく、逐次計算できる
        # 物理的負荷寄与率 (pt*nodes)/(窓*容量) の対数正規化にする(案a1)。運用(未来未知)で再現でき、
        # 分布が変わっても頑健。既定 OFF で従来のインスタンス内順位(ビット一致)。
        self._obs_occ_prior = os.environ.get("SCHEDULER_OBS_OCC_PRIOR", "0") == "1"
        # オプトイン: 報酬の無次元化(PCN_DIMLESS_NORM=1)。各workloadの達成レンジ
        # [max待ち(all-onprem), max cost(all-cloud)] を2点掃引で1回測り(jobs単位キャッシュ)、
        # step の報酬をそれで割る。→ return-to-go/archive/指令/正規化が全て無次元・全レジーム一貫に
        # なり、レジーム混合でも各レジームの指令が full range を張る(=万能税の根本除去)。
        # 既定OFF(_dimless_scale=None)で報酬素通り=ビット一致。
        # オプトイン(診断用オラクル): 各ジョブの「真解雲率」(NSGA非支配解150本のうち雲に置く割合)を
        # npz(vals: (n_jobs,))から読み、現ジョブの値を1次元追加。特徴量不足 vs 学習方式の壁の識別実験用。
        # 既定OFF(空文字)=ビット不変。
        self._obs_oracle_path = os.environ.get("SCHEDULER_ORACLE_NPZ", "")
        self._oracle_vals = None
        # オプトイン: 現ジョブの「雲送り効率」3次元を末尾に追加(SCHEDULER_OBS_EFFICIENCY=1)。
        # 真の前線は「待ち削減÷追加コスト」が大きい順に k 件だけ雲へ送るだけで再現できる
        # (18ジョブ全列挙で 19点中13点が誤差0)。ところが観測にはこの比を作る材料が無く、
        # 方策の P(cloud) と効率の順位相関は +0.26 止まり=「いくら使うか」でなく「どれに
        # 使うか」が欠けている。実行時に計算できる量だけで比を明示的に渡す:
        #   a = pt*nodes                    … 雲送り時の追加コスト(既知)
        #   b = max(0, s_on - s_cl)         … 待ち削減の見込み(オンプレ/クラウド各1回の
        #                                     配置探索クエリ _find_event_allocation の差)
        #   観測 = [log1p(a)/20, log1p(b)/16, (log1p(b)-log1p(a))/40+0.5]
        # 既定 OFF でビット不変。
        self._obs_efficiency = os.environ.get("SCHEDULER_OBS_EFFICIENCY", "0") == "1"
        # オプトイン: 現ジョブの「残り予算との比」3次元を末尾に追加(SCHEDULER_OBS_BUDGET_RATIO=1)。
        # 詳細は BUD_A_K/BUD_FRAC_K/BUD_COVER_K の定義コメント参照。
        self._obs_budget_ratio = os.environ.get("SCHEDULER_OBS_BUDGET_RATIO", "0") == "1"
        self._remaining_budget = 0.0   # set_remaining_budget() で外部(agent側rollout)から都度反映
        self._occ_suffix = None        # 残りジョブ雲コスト合計(suffix sum)キャッシュ(jobs単位)
        self._occ_suffix_jobs_id = None
        self._dimless_norm = os.environ.get("PCN_DIMLESS_NORM", "0") == "1"
        self._dimless_scale = None            # np.array([wait_scale, cost_scale]) or None
        self._dimless_cache = {}              # id(self.jobs) -> scale
        self._computing_dimless = False       # scale計測中はstepで割らない(生報酬でレンジ測定)
        self._obs_total_size = (
            self._obs_events_size
            + JOB_QUEUE_SIZE
            + (1 if self._obs_urgency else 0)
            + (1 if self._obs_occupancy else 0)
            + (1 if self._obs_oracle_path else 0)
            + (EFF_EXTRA_DIM if self._obs_efficiency else 0)
            + (BUDGET_EXTRA_DIM if self._obs_budget_ratio else 0)
        )
        # 正規化用（0-1にスケール）
        self._norm_time = max(1, self.n_window)
        self._norm_nodes = max(1, max(self.n_on_premise_node, self.n_cloud_node))
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self._obs_total_size,),
            dtype=np.float32
        )

    def reset(self):
        """リセット時にイベント履歴をクリア"""
        obs = super().reset()
        self._absolute_schedule_records = []
        if self._event_buf is not None:
            self._event_buf.reset()
        else:
            self._scheduled_events = []
        # [SCHEDULER_OBS_BUDGET_RATIO] 前エピソードの残り予算を持ち越さない(既定は0.0=
        # 「予算情報なし」。agent側rolloutが desired_return 確定後に set_remaining_budget で
        # 上書きする。random/heuristic 種エピソードは呼ばれないため 0.0 のまま=決定的)。
        self._remaining_budget = 0.0
        return self.get_observation()

    def set_remaining_budget(self, v) -> None:
        """[SCHEDULER_OBS_BUDGET_RATIO] agent側rollout(distributed_pcn._run_episode /
        pcn_agent._run_episode)が各ステップ、desired_return の cost 成分から呼ぶ。
        env は desired_return の意味(G規約・PCN_COST_HOLD等の agent側変換)を知らないため、
        「今の残り予算」を外側から都度渡す設計にする(答え・将来情報は一切使わない)。
        負値(オーバーラン)は 0 にクランプする(「残り予算」は非負の概念)。"""
        try:
            self._remaining_budget = max(0.0, float(v))
        except (TypeError, ValueError):
            self._remaining_budget = 0.0

    def _compute_dimless_scale(self):
        """all-onprem(a=0)とall-cloud(a=1)を各1回走らせ [max待ち和, max cost和] を返す(生報酬)。
        報酬無次元化のスケール。self.reset()/self.step()は実leaf(EventNative)側が呼ばれ、
        _computing_dimless=True の間は割らない/再計測しない。"""
        self._computing_dimless = True
        wsum = csum = 0.0
        try:
            n = int(getattr(self, "total_jobs_count", 0) or len(getattr(self, "jobs", [])) or 256)
            for a in (0, 1):
                self.reset()
                done = False; st = 0; tw = tc = 0.0
                while not done and st < n + 5:
                    out = self.step(a)  # _computing_dimless=True → 生報酬
                    r = out[1]; tw += -float(r[0]); tc += -float(r[1]); done = out[-1]; st += 1
                wsum = max(wsum, tw); csum = max(csum, tc)
        finally:
            self._computing_dimless = False
        return np.array([max(wsum, 1.0), max(csum, 1.0)], dtype=np.float64)

    def do_schedule(self, action, job, position):
        """スケジュール実行 + イベント記録（start_node, job_height を含む）"""
        wt = super().do_schedule(action, job, position)
        job_width = int(job[0])
        job_height = int(job[1])
        job_id = int(job[4])
        use_cloud = action[1]
        start_time = self.time
        end_time = start_time + job_width

        # position から start_node を取得（連続割り当て時は (i, a)、分散割り当て時は (i, a, node_allocation)）
        if isinstance(position, tuple) and len(position) >= 2:
            start_node = int(position[0])
            if len(position) == 3:
                # 分散割り当て: node_allocation は [[col0_nodes], [col1_nodes], ...]、最小ノードを取得
                node_alloc = position[2]
                if node_alloc is not None and len(node_alloc) > 0:
                    flat = [n for col in node_alloc for n in (col if isinstance(col, (list, tuple)) else [col])]
                    if flat:
                        start_node = int(min(flat))
        else:
            start_node = 0

        node_cols = None
        if isinstance(position, tuple) and len(position) == 3 and position[2] is not None:
            node_cols = position[2]

        self._absolute_schedule_records.append({
            "job_id": job_id,
            "start": start_time,
            "end": end_time,
            "width": job_width,
            "height": job_height,
            "use_cloud": bool(use_cloud),
            "start_node": start_node,
            "node_cols": node_cols,
        })

        if self._event_buf is not None:
            self._event_buf.append(
                float(start_time),
                float(end_time),
                float(job_width),
                float(use_cloud),
                float(start_node),
                float(job_height),
            )
        else:
            self._scheduled_events.append((
                start_time, end_time, job_width, float(use_cloud),
                start_node, job_height
            ))
        return wt

    def _pack_job_queue_f64(self) -> np.ndarray:
        """観測用ジョブキュー 40 要素（不足はゼロ埋め）"""
        jq = self.job_queue[:5].astype(np.float64)
        if getattr(self, "_obs_jq_norm", False) and jq.ndim == 2 and jq.shape[1] >= 8:
            # スケールフリー化(既定OFFでビット一致)。job_queue[i]=roll(raw,-1):
            # col0=width(pt), col1=height(nodes), col7=arrival(元raw[0])。他列は据え置き。
            jq = jq.copy()
            _nj = max(1, len(self.jobs))
            jq[:, 0] = np.clip(jq[:, 0] / self._norm_time, 0.0, 1.0)   # width → 窓幅で正規化
            jq[:, 1] = np.clip(jq[:, 1] / self._norm_nodes, 0.0, 1.0)  # height → 容量で正規化
            # col4=jobid(0..N-1, 絶対カウンタ) → 進捗率 jobid/N。N非依存な「どこまで来たか」に。
            jq[:, 4] = np.clip(jq[:, 4] / _nj, 0.0, 1.0)
            # arrival は「今からどれだけ先か」= (arrival - now)/窓幅 に。先頭(処理中)は<=0側、未来は>0側。
            jq[:, 7] = np.clip((jq[:, 7] - float(self.time)) / self._norm_time, -1.0, 1.0)
        j = np.ascontiguousarray(jq.ravel(), dtype=np.float64)
        if j.size >= JOB_QUEUE_SIZE:
            return j[:JOB_QUEUE_SIZE]
        out = np.zeros(JOB_QUEUE_SIZE, dtype=np.float64)
        out[: j.size] = j
        return out

    def _get_observation_python(self) -> np.ndarray:
        """従来の Python 実装（C 拡張が無い場合のフォールバック）"""
        if self._event_buf is not None:
            events = self._event_buf.events_numpy_copy()
        elif self._scheduled_events:
            events = np.asarray(self._scheduled_events, dtype=np.float64).reshape(-1, 6)
        else:
            events = np.empty((0, 6), dtype=np.float64)

        obs_events = np.zeros((N_EVENTS_OBS, EVENT_FEATURES), dtype=np.float32)
        if events.shape[0] > 0:
            window_start = max(0, self.time - self.n_window)
            mask = events[:, 1] >= window_start
            relevant = events[mask]
            order = np.argsort(relevant[:, 0], kind="mergesort")
            relevant = relevant[order]
            n = min(relevant.shape[0], N_EVENTS_OBS)
            if n > 0:
                chunk = relevant[-n:]
                obs_events[:n, 0] = np.clip((chunk[:, 0] - window_start) / self._norm_time, 0, 1)
                obs_events[:n, 1] = np.clip((chunk[:, 1] - window_start) / self._norm_time, 0, 1)
                obs_events[:n, 2] = np.clip(chunk[:, 2] / self._norm_time, 0, 1)
                obs_events[:n, 3] = chunk[:, 3]
                obs_events[:n, 4] = np.clip(chunk[:, 4] / self._norm_nodes, 0, 1)
                obs_events[:n, 5] = np.clip(chunk[:, 5] / self._norm_nodes, 0, 1)

        job_queue_f32 = self.job_queue[:5].astype(np.float32).flatten()
        if len(job_queue_f32) < JOB_QUEUE_SIZE:
            pad = np.zeros(JOB_QUEUE_SIZE - len(job_queue_f32), dtype=np.float32)
            job_queue_f32 = np.concatenate([job_queue_f32, pad])

        observation = np.concatenate([
            obs_events.flatten(),
            job_queue_f32
        ]).astype(np.float32)
        return observation

    def _front_job_urgency(self) -> float:
        """現ジョブのオンプレ予測待ち時間を log 正規化した緊急度 [0,1]。
        event-native の _find_event_allocation（副作用なしのクエリ）で予測。非対応 env は 0。
        これが観測に無いと、方策は閾値判断ができず予算駆動の front-loading に退化する。"""
        try:
            if not hasattr(self, "_find_event_allocation"):
                return 0.0
            jobs = getattr(self, "jobs", None)
            j = int(getattr(self, "index_next_job", 0))
            if jobs is None or j >= len(jobs):
                return 0.0
            raw = jobs[j]
            job = self._to_queue_job(raw) if hasattr(self, "_to_queue_job") else raw
            arr = int(raw[0])
            pos, onp = self._find_event_allocation(job, False, arr)
            # 次step が同じジョブを onprem 配置するとき再計算を省くため (position,start) を控える。
            self._front_alloc_cache = (j, arr, pos, int(onp))
            pw = max(0, int(onp) - arr)
            return float(np.clip(np.log1p(pw) / 16.0, 0.0, 1.0))
        except Exception:
            self._front_alloc_cache = None
            return 0.0

    def _front_job_oracle(self) -> float:
        """現ジョブの真解雲率 [0,1] (診断オラクル、SCHEDULER_ORACLE_NPZ 指定時のみ)。"""
        try:
            if self._oracle_vals is None:
                self._oracle_vals = np.load(self._obs_oracle_path)["vals"].astype(np.float32)
            j = int(getattr(self, "index_next_job", 0))
            if j < len(self._oracle_vals):
                return float(self._oracle_vals[j])
            return 0.0
        except Exception:
            return 0.0

    def _front_job_leverage(self) -> float:
        """現ジョブの占有量(pt*nodes)のインスタンス内パーセンタイル順位 [0,1]。
        1.0 に近いほど「全ジョブ中で最大級の巨大ジョブ＝高レバレッジ決定」。非対応 env は 0。"""
        try:
            jobs = getattr(self, "jobs", None)
            j = int(getattr(self, "index_next_job", 0))
            if jobs is None or j >= len(jobs):
                return 0.0
            # 案a1: 逐次計算できる物理的負荷寄与率(全ジョブ順位=全知を使わない)。
            # occ_frac = (pt*nodes)/(窓幅*容量) = このジョブ1件が窓×容量に占める割合(スケール不変)。
            # 対数正規化で [0,1] に(urgency と同じ思想)。K=負荷寄与率の典型対数幅。
            if getattr(self, "_obs_occ_prior", False):
                raw = jobs[j]
                # 負荷寄与率 = (pt/窓幅)*(nodes/オンプレ容量) = このジョブがオンプレ窓×容量に占める割合。
                # スケジューリングはオンプレの話なので容量はオンプレ基準(n_on)で正規化する。
                _cap = max(1.0, float(self.n_on_premise_node))
                occ_frac = (float(raw[1]) / float(self._norm_time)) * (float(raw[2]) / _cap)
                # occ_frac(~1e-3〜5e-2)を log1p(・*S)/K で [0,1] に広く分布させる(S,K固定較正)。
                return float(np.clip(np.log1p(occ_frac * 1.0e3) / 4.3, 0.0, 1.0))
            # 占有量ソート列をインスタンス単位でキャッシュ（job配列が変わったら再計算）
            if self._occ_sorted is None or self._occ_jobs_id != id(jobs):
                arr = np.asarray(jobs, dtype=np.float64)
                occ_all = arr[:, 1] * arr[:, 2]
                self._occ_sorted = np.sort(occ_all)
                self._occ_jobs_id = id(jobs)
            raw = jobs[j]
            occ = float(raw[1]) * float(raw[2])
            n = len(self._occ_sorted)
            if n <= 0:
                return 0.0
            rank = float(np.searchsorted(self._occ_sorted, occ, side="right")) / n
            return float(np.clip(rank, 0.0, 1.0))
        except Exception:
            return 0.0

    def _front_job_efficiency(self):
        """現ジョブの雲送り効率 3特徴 [log追加コスト, log待ち削減, log効率比] ∈ [0,1]^3。

        近似の内容（報告用に明記）:
          - 追加コスト a は近似なしの厳密値 = required_nodes × processing_time
            （env の _compute_event_cost と同一式）。
          - 待ち削減 b は「今この瞬間の配置探索の差」= s_on - s_cl。s_on/s_cl は
            _find_event_allocation（副作用なしのクエリ）をオンプレ/クラウドで1回ずつ
            呼んだ厳密な予測開始時刻。したがって "現ジョブ自身の待ち削減" は厳密で、
            後続ジョブへの波及（オンプレを空けたことで後続が早く始まる分）は含まない。
          - 18ジョブ全列挙の参照効率（全オンプレからjを1件だけ雲へ動かしたときの
            Δ平均待ち÷Δコスト）との順位相関は Spearman 0.973（実測）。
            比較: -log(コスト)単独は 0.845、待ち削減単独は 0.518。

        非対応 env / 例外時は (0,0,0)。ジョブ列を使い切っている場合も (0,0,0)。
        """
        try:
            if not hasattr(self, "_find_event_allocation"):
                return (0.0, 0.0, 0.0)
            jobs = getattr(self, "jobs", None)
            j = int(getattr(self, "index_next_job", 0))
            if jobs is None or j >= len(jobs):
                return (0.0, 0.0, 0.0)
            raw = jobs[j]
            job = self._to_queue_job(raw) if hasattr(self, "_to_queue_job") else raw
            arr = int(raw[0])
            # urgency が直前に同じ (job, arrival) のオンプレ配置を解いていれば再利用
            # （events 不変なので同一結果=追加コスト0）。無ければ自分で解いて同じ形で控える。
            cache = getattr(self, "_front_alloc_cache", None)
            if cache is not None and cache[0] == j and cache[1] == arr:
                s_on = int(cache[3])
            else:
                pos, s_on = self._find_event_allocation(job, False, arr)
                s_on = int(s_on)
                self._front_alloc_cache = (j, arr, pos, s_on)
            _, s_cl = self._find_event_allocation(job, True, arr)
            a = float(raw[1]) * float(raw[2])          # 雲送りの追加コスト = pt*nodes
            b = max(0.0, float(s_on) - float(s_cl))    # 待ち削減の見込み(秒)
            la = float(np.log1p(max(a, 0.0)))
            lb = float(np.log1p(b))
            return (
                float(np.clip(la / EFF_COST_K, 0.0, 1.0)),
                float(np.clip(lb / EFF_GAIN_K, 0.0, 1.0)),
                float(np.clip((lb - la) / EFF_RATIO_K + 0.5, 0.0, 1.0)),
            )
        except Exception:
            return (0.0, 0.0, 0.0)

    def _front_job_budget_ratio(self):
        """現ジョブの「残り予算との比」3特徴 [BR1, BR2, BR3] ∈ [0,1]^3
        (SCHEDULER_OBS_BUDGET_RATIO=1; 定数の意味は BUD_A_K 等の定義コメント参照)。

        rem(残りジョブの雲コスト合計)は現在位置 index_next_job から末尾まで(自分を含む)の
        pt*nodes 総和。self.jobs は reset() 時点でこの窓分すべて確定しているジョブ列(未来の
        「実測アウトカム」ではなく、job_queue が既に5件先読みしているのと同種の既知情報)
        なので、jobs[j:] の集計は「実行時に計算可能」の範囲内。

        budget(残り予算)は self._remaining_budget(set_remaining_budget 経由で agent 側
        rollout が都度反映)をそのまま使う。呼ばれていなければ 0.0(=予算情報なし)。

        非対応/例外/ジョブ使い切り時は (0,0,0)。"""
        try:
            jobs = getattr(self, "jobs", None)
            j = int(getattr(self, "index_next_job", 0))
            if jobs is None or j >= len(jobs):
                return (0.0, 0.0, 0.0)
            if self._occ_suffix is None or self._occ_suffix_jobs_id != id(jobs):
                arr = np.asarray(jobs, dtype=np.float64)
                occ_all = arr[:, 1] * arr[:, 2]
                # suffix[i] = sum(occ_all[i:]) : i(自分含む)以降の雲コスト合計
                self._occ_suffix = np.cumsum(occ_all[::-1])[::-1].copy()
                self._occ_suffix_jobs_id = id(jobs)
            raw = jobs[j]
            a = float(raw[1]) * float(raw[2])
            rem = float(self._occ_suffix[j]) if j < len(self._occ_suffix) else a
            budget = float(self._remaining_budget)
            r1 = a / max(budget, 1.0)
            r2 = a / max(rem, 1.0)
            r3 = budget / max(rem, 1.0)
            return (
                float(np.clip(np.log1p(r1) / BUD_A_K, 0.0, 1.0)),
                float(np.clip(np.log1p(r2) / BUD_FRAC_K, 0.0, 1.0)),
                float(np.clip(np.log1p(r3) / BUD_COVER_K, 0.0, 1.0)),
            )
        except Exception:
            return (0.0, 0.0, 0.0)

    def get_observation(self):
        """イベントベース観測（必要なら緊急度・レバレッジ・効率・予算比を順に末尾へ追加）。"""
        base = self._get_observation_base()
        extras = []
        if getattr(self, "_obs_urgency", False):
            extras.append(self._front_job_urgency())
        if getattr(self, "_obs_occupancy", False):
            extras.append(self._front_job_leverage())
        if getattr(self, "_obs_oracle_path", ""):
            extras.append(self._front_job_oracle())
        if getattr(self, "_obs_efficiency", False):
            extras.extend(self._front_job_efficiency())
        if getattr(self, "_obs_budget_ratio", False):
            extras.extend(self._front_job_budget_ratio())
        if not extras:
            return base
        return np.concatenate(
            [np.asarray(base, dtype=np.float32), np.asarray(extras, dtype=np.float32)]
        )

    def _get_observation_base(self):
        """
        イベントベース観測を返す（ビットマップ不使用）

        観測構成:
        - イベント部分 (N_EVENTS_OBS * 6): 各 (start_norm, end_norm, duration_norm, use_cloud, start_node_norm, job_height_norm)
        - ジョブキュー部分 (40): job_queue[:5].flatten()
        
        既定: scheduling_env_core.get_observation_event（C）で構築。
        イベント履歴は SchedulingEventBuffer を渡し list→ndarray の変換を省略。
        マップ復元: 各イベントの (start_node, start_node+job_height) × (start_time, end_time) が占有領域
        """
        if _c_get_observation_event is not None and self._event_buf is not None:
            jq = self._pack_job_queue_f64()
            return _c_get_observation_event(
                self._event_buf,
                int(self.time),
                int(self.n_window),
                float(self._norm_time),
                float(self._norm_nodes),
                N_EVENTS_OBS,
                EVENT_FEATURES,
                JOB_QUEUE_SIZE,
                jq,
            )
        if _c_get_observation_event is not None:
            if self._scheduled_events:
                ev = np.asarray(self._scheduled_events, dtype=np.float64).reshape(-1, 6)
            else:
                ev = np.empty((0, 6), dtype=np.float64)
            jq = self._pack_job_queue_f64()
            return _c_get_observation_event(
                ev,
                int(self.time),
                int(self.n_window),
                float(self._norm_time),
                float(self._norm_nodes),
                N_EVENTS_OBS,
                EVENT_FEATURES,
                JOB_QUEUE_SIZE,
                jq,
            )
        return self._get_observation_python()
