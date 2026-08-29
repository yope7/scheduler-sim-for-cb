import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from src.utils.job_gen.poason_job import JobSimulator


class JobGenerator:
    jobs_set = {}
    job_id = 0
    nb_steps = None
    nb_episodes = None
    max_t = None
    job_type = None
    n_jobs = None
    lam = None
    # envのパラメータ
    n_window = None
    n_on_premise_node = None
    n_cloud_node = None

    # ファイルに出力するジョブのパラメータ(dataframe)
    df_prmj = None

    def __init__(self, seed, nb_steps, n_window, n_on_premise_node, n_cloud_node, config,n_jobs,lam,nb_episodes):
        self.config = config
        self.nb_steps = nb_steps
        self.n_jobs = n_jobs
        # 最大時間を入力
        self.lam = lam
        self.max_t = self.config['param_job']['max_t']
        self.nb_episodes = nb_episodes
        self.seed = seed


        # envのパラメータ
        self.n_window = n_window
        self.n_on_premise_node = n_on_premise_node
        self.n_cloud_node = n_cloud_node

        # ジョブ設定
        # ジョブをエピソードごとに変えるか，固定にするかを指定
        job_is_constant = self.config['param_job']['job_is_constant']

        # ジョブタイプを指定
        if job_is_constant:
            self.job_type = config['param_job']['job_type']
        else:
            self.job_type = int(input(
                'select job-type \n 1:default 2:input file 3:random 4:random(エピソードごとにジョブ数やジョブサイズの範囲を変える) 5:dqnに有利でヒューリスティックに不利な選択が大事なジョブ環境 \n >'))

    # 最終的にenvに渡すジョブのまとまりを生成
    def generate_jobs_set(self):
        if self.job_type == 1:
            np.random.seed(self.seed)
            # 合成↔trace の連続補間ダイヤル（占有量 power-law 裾）。
            # env(SYNTH_TAIL_*) > config(param_job.synth_tail_*) > 既定OFF の優先順。
            # 既定(level=0)では JobSimulator が裾ブロックを実行せず従来とビット一致。
            pj = self.config.get('param_job', {})

            def _envcfg(env_key, cfg_key, default=None):
                v = os.environ.get(env_key)
                if v is not None and v != "":
                    return v
                return pj.get(cfg_key, default)

            _tl = _envcfg("SYNTH_TAIL_LEVEL", "synth_tail_level", 0.0)
            tail_level = float(_tl) if _tl not in (None, "") else 0.0
            _ta = _envcfg("SYNTH_TAIL_ALPHA", "synth_tail_alpha", None)
            tail_alpha = float(_ta) if _ta not in (None, "") else None
            _tf = _envcfg("SYNTH_TAIL_FRAC", "synth_tail_frac", None)
            tail_frac = float(_tf) if _tf not in (None, "") else None
            _tc = _envcfg("SYNTH_TAIL_PT_CAP", "synth_tail_pt_cap", None)
            tail_pt_cap = int(float(_tc)) if _tc not in (None, "") else None
            _tb = _envcfg("SYNTH_TAIL_BURST", "synth_tail_burst", 0)
            tail_burst = int(float(_tb)) if _tb not in (None, "") else 0

            # 合成ジョブの要求ノード数上限。既定(None)=JobSimulatorの既定256=従来とビット一致。
            # SYNTH_MAX_NODES / config param_job.synth_max_required_nodes で小ジョブ化(現実的な
            # クラスタ負荷率<~1)にできる。実SQUIDのジョブは実質1ノード級=大ジョブU[1,256)は非現実的。
            _mn = _envcfg("SYNTH_MAX_NODES", "synth_max_required_nodes", None)
            synth_max_nodes = int(float(_mn)) if _mn not in (None, "") else 256

            # JobSimulatorのデフォルト設定を使用（過去の実験と一致）
            jobgen = JobSimulator(
                self.seed,
                n_jobs=self.n_jobs,
                n_users=self.n_jobs,
                lam=self.lam,
                max_required_nodes=synth_max_nodes,
                tail_level=tail_level,
                tail_alpha=tail_alpha,
                tail_frac=tail_frac,
                tail_max_nodes=self.n_cloud_node,
                tail_seed=int(self.seed) + 1000003,
                tail_pt_cap=tail_pt_cap,
                tail_burst=tail_burst,
            )
            jobs = jobgen.generate_jobs()
            if tail_level and tail_level > 0:
                _occ = (jobs[:, 1] * jobs[:, 2]).astype(float)
                _srt = np.sort(_occ)[::-1]
                _top10 = float(_srt[:10].sum() / max(_occ.sum(), 1e-9))
                print(
                    f"[JobGenerator] SYNTH_TAIL_LEVEL={tail_level} alpha={tail_alpha} frac={tail_frac} "
                    f"→ occupancy max/median={_occ.max()/max(np.median(_occ),1):.1f}x, top10={_top10:.1%}"
                )
            # 混雑レジーム制御(既定1.0=ビット一致): 到着時刻(col0)を係数倍し到着スパンを伸縮=稼働率ρを変える。
            # ρ=drain/到着スパン なので f倍で ρ→ρ/f。synthを引き伸ばして trace並みに空かせる等に使う。
            _as = float(os.environ.get("SCHEDULER_ARRIVAL_SCALE", "1.0"))
            if _as != 1.0:
                jobs = np.array(jobs, dtype=float); jobs[:, 0] = jobs[:, 0] * _as
                print(f"[JobGenerator] SYNTH arrival scale x{_as} (混雑レジーム制御, ρ→ρ/{_as})")
            for episode in range(self.nb_episodes + 1):
                self.jobs_set[episode] = jobs

        # csvファイルからジョブを読み込む
        elif self.job_type == 2:
            trace_path_raw = self.config["param_job"].get("job_trace_path")
            if trace_path_raw is None or trace_path_raw == "":
                raise ValueError("param_job.job_trace_path が設定されていません")

            repo_root = Path(__file__).resolve().parents[3]
            trace_path = Path(trace_path_raw).expanduser()
            if not trace_path.is_absolute():
                trace_path = repo_root / trace_path
            if not trace_path.exists():
                raise FileNotFoundError(f"指定されたジョブトレースが見つかりません: {trace_path}")

            n_rows = self.config["param_job"].get("job_trace_n_jobs", -1)
            n_rows = int(n_rows) if n_rows is not None else -1
            exclude_outlier = bool(
                self.config["param_job"].get("job_trace_exclude_largest_outlier", False)
            )
            read_n = None if n_rows <= 0 else n_rows
            if exclude_outlier and read_n is not None and read_n > 0:
                read_n = int(read_n) + 1

            jobs_df = pd.read_csv(trace_path, nrows=read_n)
            jobs = jobs_df.values.tolist()

            # 退化ジョブの防御的クランプ: processing_time<=0 はイベント割当
            # (_find_event_allocation) で width=0 となり ValueError を起こす。
            # アカウンティングログ由来の 0 秒ジョブを最小 1 に丸める（件数は維持）。
            _n_clamped_pt = 0
            for row in jobs:
                if float(row[1]) < 1:
                    row[1] = 1
                    _n_clamped_pt += 1
            if _n_clamped_pt:
                print(
                    f"[JobGenerator] clamped processing_time>=1 for {_n_clamped_pt} "
                    f"degenerate (pt<=0) trace job(s)"
                )

            if exclude_outlier and n_rows > 0:
                if len(jobs) <= n_rows:
                    raise ValueError(
                        f"job_trace_exclude_largest_outlier: need at least {n_rows + 1} rows, got {len(jobs)}"
                    )
                scores = [float(row[1]) * float(row[2]) for row in jobs]
                drop_i = int(np.argmax(scores))
                dropped = jobs.pop(drop_i)
                for i, row in enumerate(jobs):
                    row[5] = i
                print(
                    "[JobGenerator] excluded largest outlier: "
                    f"trace job_id={dropped[5]}, pt={dropped[1]}, nodes={dropped[2]}, "
                    f"pt*nodes={scores[drop_i]:.0f}"
                )
                if len(jobs) != n_rows:
                    raise ValueError(
                        f"expected {n_rows} jobs after outlier exclusion, got {len(jobs)}"
                    )

            print(f"[JobGenerator] job trace読み込み: {trace_path} (取得件数={len(jobs)} / nrows={read_n})")
            print("[JobGenerator] 先頭5件サンプル:", jobs[:5])

            # --- 因果分離ノブ(既定OFF=ビット一致): 実traceの「到着↔ジョブサイズ相関/giant塊到着」を破壊する。
            # SCHEDULER_TRACE_SHUFFLE_PAYLOAD=1 で、到着時刻(col0)とjob_id(col5)を固定したまま
            # payload列(pt1/nodes2/cloud3/user4)を行間で一括置換=各到着スロットへ別ジョブの中身を割当。
            # 両周辺分布(到着分布・pt/nodes分布)と到着ソート順は完全保持し「どのジョブがいつ来るか」だけ壊す。
            # trace固有の不安定が到着↔サイズ構造由来かを切り分ける。seedは固定(全runで同一instance)。
            _shuf = os.environ.get("SCHEDULER_TRACE_SHUFFLE_PAYLOAD", "0")
            if _shuf not in ("0", "", "off"):
                _sseed = int(os.environ.get("SCHEDULER_TRACE_SHUFFLE_SEED", "20260715"))
                _rng = np.random.default_rng(_sseed)
                _n = len(jobs)
                _perm = _rng.permutation(_n)
                _orig = [list(r) for r in jobs]
                for _i in range(_n):
                    _src = _orig[_perm[_i]]
                    for _c in (1, 2, 3, 4):
                        jobs[_i][_c] = _src[_c]
                _occ0 = [float(r[1]) * float(r[2]) for r in _orig]
                _occ1 = [float(r[1]) * float(r[2]) for r in jobs]
                _st = [float(r[0]) for r in jobs]
                _c0 = float(np.corrcoef(_st, _occ0)[0, 1]) if np.std(_st) > 0 else 0.0
                _c1 = float(np.corrcoef(_st, _occ1)[0, 1]) if np.std(_st) > 0 else 0.0
                print(f"[JobGenerator] TRACE payload shuffle (到着↔サイズ相関を破壊): seed={_sseed} "
                      f"corr(arrival,occ) {_c0:+.3f}→{_c1:+.3f} (両周辺分布は不変)")

            # --- 因果分離ノブ2(既定OFF=ビット一致): 実traceの「塊/同時到着(バースト的arrival分布)」を破壊。
            # SCHEDULER_TRACE_ARRIVAL_SPREAD=1 で、到着時刻(col0)を元の総スパンに均等再配置し
            # 同時到着・微小間隔クラスタを消す(payloadは順序保持=到着ランク↔サイズ関係は温存)。
            # synth(指数到着でバラける)との差=arrival分布のクラスタ性かを切り分ける。
            _spr = os.environ.get("SCHEDULER_TRACE_ARRIVAL_SPREAD", "0")
            if _spr not in ("0", "", "off"):
                _st0 = [float(r[0]) for r in jobs]
                _span = max(_st0) - min(_st0)
                _lo = min(_st0)
                _n = len(jobs)
                _ndist0 = len(set(_st0))
                if _span <= 0:
                    _span = float(_n)  # 全同時刻の退化ケースの保険
                for _i in range(_n):
                    jobs[_i][0] = _lo + _span * _i / max(1, _n - 1)  # 均等スパン再配置(昇順=元順序保持)
                print(f"[JobGenerator] TRACE arrival spread (塊/同時到着を破壊): "
                      f"distinct時刻 {_ndist0}/{_n}→{_n}/{_n} span={_span:.0f} 均等間隔={_span/max(1,_n-1):.0f} (payload順序不変)")

            # --- 因果分離ノブ3(既定OFF=ビット一致): 実traceの「ジョブ種の縮退(54%が同一ジョブ)」を破壊。
            # SCHEDULER_TRACE_JITTER_PT=J で pt に ×(1±U(0,J)) の微小jitterを掛け、完全重複を全て別値に。
            # (pt,nodes)ペアが全distinct化=各ジョブに固有の観測信号→部分cloud可能に。低多様性が崩壊源かを切り分け。
            # 分布の形はほぼ不変(±J%)。seed固定=全runで同一instance。
            _jit = os.environ.get("SCHEDULER_TRACE_JITTER_PT", "0")
            if _jit not in ("0", "", "off"):
                _J = float(_jit)
                _rng = np.random.default_rng(int(os.environ.get("SCHEDULER_TRACE_JITTER_SEED", "20260716")))
                _n = len(jobs)
                _u0 = len(set((int(r[1]), int(r[2])) for r in jobs))
                for _i in range(_n):
                    _mult = 1.0 + _rng.uniform(-_J, _J)
                    jobs[_i][1] = max(1, int(round(float(jobs[_i][1]) * _mult)))
                _u1 = len(set((int(r[1]), int(r[2])) for r in jobs))
                print(f"[JobGenerator] TRACE pt jitter (ジョブ種の縮退を破壊): J=±{_J:.0%} "
                      f"ユニーク(pt,nodes) {_u0}→{_u1}/{_n}種 (分布の形はほぼ不変)")

            # 混雑レジーム制御(既定1.0=ビット一致): 到着時刻を係数倍しρを変える(trace圧縮でsynth並みに混ませる等)。
            _as = float(os.environ.get("SCHEDULER_ARRIVAL_SCALE", "1.0"))
            if _as != 1.0:
                for _r in jobs:
                    _r[0] = float(_r[0]) * _as
                print(f"[JobGenerator] TRACE arrival scale x{_as} (混雑レジーム制御, ρ→ρ/{_as})")

            for episode in range(self.nb_episodes + 1):
                self.jobs_set[episode] = jobs

        # ランダムの場合，必要な変数を指定
        elif self.job_type == 3:
            # 最大時間を入力
            # max_t = int(input('input max time \n >'))

            # 単位時間あたりのジョブ数を入力
            n_job_per_time = self.config['param_job']['n_job_per_time']
            n_job_per_time_is_random = False
            if n_job_per_time == -1:
                n_job_per_time_is_random = True

            # 最大処理時間を入力
            max_processing_time = self.config['param_job']['max_processing_time']
            max_processing_time_is_random = False
            if max_processing_time == -1:
                max_processing_time_is_random = True

            # 最大要求ノード数を入力
            max_n_required_nodes = self.config['param_job']['max_n_required_nodes']
            max_n_required_nodes_is_random = False
            if max_n_required_nodes == -1:
                max_n_required_nodes_is_random = True

            nb_jobs_per_user = self.config['param_job']['nb_jobs_per_user']

            #ユーザ数を入力
            max_user_id = self.config['param_job']['max_user_id']-1

            user_increment = self.config['param_job']['user_increment']


            #waiting_timeの初期値は-1
            waiting_time = -1 

            prm_job = [self.nb_steps, self.max_t, n_job_per_time, max_processing_time, max_n_required_nodes, max_user_id]
            self.df_prmj = pd.DataFrame(
                [prm_job],
                columns=['n_steps', 'max_time', 'n_job_per_time', 'max_processing_time', 'max_n_required_nodes', 'max_user_id'],
                index=['idx']
            )

            if user_increment:

                for episode in range(self.nb_episodes): 
                    user_id = 1
                    jobs = []
                    self.job_id = 0
                    user_count = 0
                    for i in range(self.max_t + 1):
                        if n_job_per_time_is_random:
                            n_job_per_time = np.random.randint(1, 6)
                        for _ in range(n_job_per_time):
                            submit_time = i

                            # 通常サイズのジョブ(デフォルト仕様)
                            processing_time = np.random.randint(1, max_processing_time + 1)
                            n_required_nodes = np.random.randint(1, max_n_required_nodes + 1)
                            #ユーザIDをランダムに割り振る。
                            user_count += 1


                            can_use_cloud = 1
                            jobs.append(
                                self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud, user_id)
                            )

                            if user_count == n_job_per_time*self.max_t //max_user_id:
                                user_id += 1
                                user_count = 0

                    self.jobs_set[episode] = jobs
                
            else:
                for episode in range(self.nb_episodes): 
                    jobs = []
                    self.job_id = 0
                    user_count = 0
                    for i in range(self.max_t + 1):
                        if n_job_per_time_is_random:
                            n_job_per_time = np.random.randint(1, 6)
                        for _ in range(n_job_per_time):
                            submit_time = i

                            # 通常サイズのジョブ(デフォルト仕様)
                            processing_time = np.random.randint(1, max_processing_time + 1)
                            n_required_nodes = np.random.randint(1, max_n_required_nodes + 1)
                            #ユーザIDをランダムに割り振る。
                            user_id = np.random.randint(1, max_user_id+2)



                            can_use_cloud = 1
                            jobs.append(
                                self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud, user_id)
                            )


                    self.jobs_set[episode] = jobs


        # 途中で性質が切り替わる場合の設定
        elif self.job_type == 4:
            random_job_type = int(input(
                'select random job type \n 1:a few small jobs -> a few big jobs -> a few small jobs(only on-premise many '
                '\n 2:n_job gradually go up) \n 3:小 -> 中 -> 大 \n 4:n_job 2->3->2->3->4 \n 5: njob:2, '
                'maxsize:2×2->3×3->2×2->3×3->4×4 \n 6: njob:2, maxsize:2×2->3×3->4×4 \n 7: njob:2, maxsize:2×2->2×2->3×3 '
                '\n 8: njob:2->2->3 \n 9: njob:1->2->3 \n >'
            ))

            # 小少 -> 大少 -> オンプレミスのみ多
            if random_job_type == 1:
                points_switch_random_job_type = [self.nb_episodes // 3, self.nb_episodes // 3 * 2]
                for episode in range(self.nb_episodes + 1):
                    jobs = []
                    self.job_id = 0

                    # 小
                    if episode < points_switch_random_job_type[0]:
                        n_job_per_time = 3
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # 大
                    elif episode < points_switch_random_job_type[1]:
                        n_job_per_time = 2
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(3, 5)
                                n_required_nodes = np.random.randint(3, 5)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # 小(オンプレミスのみが多め)
                    else:
                        n_job_per_time = 3
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                # 20%の確率でクラウド使用可
                                if np.random.uniform() <= 0.2:
                                    can_use_cloud = 1
                                else:
                                    can_use_cloud = 0
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    self.jobs_set[episode] = jobs

            # ジョブ数：3 -> 4 -> 5
            elif random_job_type == 2:
                # nb_episodesを再設定
                nb_episodes = self.nb_steps // (self.max_t * 4)

                points_switch_random_job_type = [nb_episodes // 3, nb_episodes // 3 * 2]
                for episode in range(nb_episodes + 1):
                    jobs = []
                    self.job_id = 0

                    # ジョブ数=3
                    if episode < points_switch_random_job_type[0]:
                        n_job_per_time = 3
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # ジョブ数=4
                    elif episode < points_switch_random_job_type[1]:
                        n_job_per_time = 4
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # ジョブ数=5
                    else:
                        n_job_per_time = 5
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    self.jobs_set[episode] = jobs

            # 小 -> 中 -> 大
            elif random_job_type == 3:
                points_switch_random_job_type = [self.nb_episodes // 3, self.nb_episodes // 3 * 2]
                for episode in range(self.nb_episodes + 1):
                    jobs = []
                    self.job_id = 0

                    # 小
                    if episode < points_switch_random_job_type[0]:
                        n_job_per_time = 3
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # 中
                    elif episode < points_switch_random_job_type[1]:
                        n_job_per_time = 3
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(2, 4)
                                n_required_nodes = np.random.randint(2, 4)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # 大
                    else:
                        n_job_per_time = 2
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(3, 5)
                                n_required_nodes = np.random.randint(3, 5)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    self.jobs_set[episode] = jobs

                prm_job = [self.nb_steps, self.max_t, points_switch_random_job_type[0], points_switch_random_job_type[1]]
                self.df_prmj = pd.DataFrame(
                    [prm_job],
                    columns=['n_steps', 'max_time', 'point_switch_job_type1', 'point_switch_job_type2'],
                    index=['idx']
                )

            # ジョブ数：2 -> 3 -> 2 -> 3 -> 4
            elif random_job_type == 4:
                # nb_episodesを再設定
                # nb_episodes = nb_steps // (max_t * 3)
                nb_episodes = 50000

                # points_switch_random_job_type = [nb_episodes // 6, nb_episodes // 6 * 2, nb_episodes // 6 * 3, nb_episodes // 6 * 4]
                points_switch_random_job_type = [7500, 15000, 20000, 25000]
                for episode in range(nb_episodes + 1):
                    jobs = []
                    self.job_id = 0

                    # ジョブ数=2(事前学習)
                    if episode < points_switch_random_job_type[0]:
                        n_job_per_time = 2
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # ジョブ数=3(事前学習)
                    elif episode < points_switch_random_job_type[1]:
                        n_job_per_time = 3
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # ジョブ数=2(経験済み)
                    elif episode < points_switch_random_job_type[2]:
                        n_job_per_time = 2
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # ジョブ数=3(経験済み)
                    elif episode < points_switch_random_job_type[3]:
                        n_job_per_time = 3
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # ジョブ数=4(未経験)
                    else:
                        n_job_per_time = 4
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    self.jobs_set[episode] = jobs

                prm_job = [self.nb_steps, self.max_t, points_switch_random_job_type[0], points_switch_random_job_type[1],
                           points_switch_random_job_type[2], points_switch_random_job_type[3]]
                self.df_prmj = pd.DataFrame(
                    [prm_job],
                    columns=['n_steps', 'max_time', 'point_switch_job_type1', 'point_switch_job_type2',
                             'point_switch_job_type3', 'point_switch_job_type4'],
                    index=['idx']
                )

            # njob:2, maxsize:2×2->3×3->2×2->3×3->4×4
            elif random_job_type == 5:
                # nb_episodesを再設定
                # nb_episodes = nb_steps // (max_t * 3)
                nb_episodes = 60000

                # points_switch_random_job_type = [nb_episodes // 6, nb_episodes // 6 * 2, nb_episodes // 6 * 3, nb_episodes // 6 * 4]
                points_switch_random_job_type = [7500, 15000, 20000, 25000]
                for episode in range(nb_episodes + 1):
                    jobs = []
                    self.job_id = 0
                    n_job_per_time = 2  # njob=2で固定

                    # maxsize:2×2(事前学習)
                    if episode < points_switch_random_job_type[0]:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # maxsize:3×3(事前学習)
                    elif episode < points_switch_random_job_type[1]:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 4)
                                n_required_nodes = np.random.randint(1, 4)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # maxsize:2×2(経験済み)
                    elif episode < points_switch_random_job_type[2]:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # maxsize:3×3(経験済み)
                    elif episode < points_switch_random_job_type[3]:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 4)
                                n_required_nodes = np.random.randint(1, 4)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # maxsize:4×4(未経験)
                    else:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 5)
                                n_required_nodes = np.random.randint(1, 5)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    self.jobs_set[episode] = jobs

                prm_job = [self.nb_steps, self.max_t, points_switch_random_job_type[0], points_switch_random_job_type[1],
                           points_switch_random_job_type[2], points_switch_random_job_type[3]]
                self.df_prmj = pd.DataFrame(
                    [prm_job],
                    columns=['n_steps', 'max_time', 'point_switch_job_type1', 'point_switch_job_type2',
                             'point_switch_job_type3', 'point_switch_job_type4'],
                    index=['idx']
                )

            # njob:2, maxsize:2×2->3×3->4×4
            elif random_job_type == 6:
                # nb_episodesを再設定
                # nb_episodes = nb_steps // (max_t * 3)
                nb_episodes = 50000

                # points_switch_random_job_type = [nb_episodes // 6, nb_episodes // 6 * 2, nb_episodes // 6 * 3, nb_episodes // 6 * 4]
                points_switch_random_job_type = [7500, 15000]
                for episode in range(nb_episodes + 1):
                    jobs = []
                    self.job_id = 0
                    n_job_per_time = 2  # njob=2で固定

                    # maxsize:2×2
                    if episode < points_switch_random_job_type[0]:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # maxsize:3×3
                    elif episode < points_switch_random_job_type[1]:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 4)
                                n_required_nodes = np.random.randint(1, 4)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # maxsize:4×4
                    else:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 5)
                                n_required_nodes = np.random.randint(1, 5)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    self.jobs_set[episode] = jobs

                prm_job = [self.nb_steps, self.max_t, points_switch_random_job_type[0], points_switch_random_job_type[1]]
                self.df_prmj = pd.DataFrame(
                    [prm_job],
                    columns=['n_steps', 'max_time', 'point_switch_job_type1', 'point_switch_job_type2'],
                    index=['idx']
                )

            # njob:2, maxsize:2×2->2×2->3×3
            elif random_job_type == 7:
                n_job_per_time = 2  # njob=2で固定
                # nb_episodesを再設定
                # nb_episodes = nb_steps // (max_t * n_job_per_time)
                nb_episodes = 70000

                # points_switch_random_job_type = [nb_episodes // 6, nb_episodes // 6 * 2, nb_episodes // 6 * 3, nb_episodes // 6 * 4]
                points_switch_random_job_type = [10000, 20000]
                for episode in range(nb_episodes + 1):
                    jobs = []
                    self.job_id = 0

                    # maxsize:2×2
                    if episode < points_switch_random_job_type[0]:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # maxsize:2×2
                    elif episode < points_switch_random_job_type[1]:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # maxsize:3×3
                    else:
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 4)
                                n_required_nodes = np.random.randint(1, 4)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    self.jobs_set[episode] = jobs

                prm_job = [self.nb_steps, self.max_t, points_switch_random_job_type[0], points_switch_random_job_type[1]]
                self.df_prmj = pd.DataFrame(
                    [prm_job],
                    columns=['n_steps', 'max_time', 'point_switch_job_type1', 'point_switch_job_type2'],
                    index=['idx']
                )

            # njob:2->2->3
            elif random_job_type == 8:
                # nb_episodesを再設定
                # nb_episodes = nb_steps // (max_t * n_job_per_time)
                nb_episodes = 70000

                # points_switch_random_job_type = [nb_episodes // 6, nb_episodes // 6 * 2, nb_episodes // 6 * 3, nb_episodes // 6 * 4]
                points_switch_random_job_type = [10000, 20000]
                for episode in range(nb_episodes + 1):
                    jobs = []
                    self.job_id = 0

                    # njob=2
                    if episode < points_switch_random_job_type[0]:
                        n_job_per_time = 2
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # njob=2
                    elif episode < points_switch_random_job_type[1]:
                        n_job_per_time = 2
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # njob=3
                    else:
                        n_job_per_time = 3
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    self.jobs_set[episode] = jobs

                prm_job = [self.nb_steps, self.max_t, points_switch_random_job_type[0], points_switch_random_job_type[1]]
                self.df_prmj = pd.DataFrame(
                    [prm_job],
                    columns=['n_steps', 'max_time', 'point_switch_job_type1', 'point_switch_job_type2'],
                    index=['idx']
                )

            # njob:1->2->3
            elif random_job_type == 9:
                # nb_episodesを再設定
                # nb_episodes = nb_steps // (max_t * n_job_per_time)
                nb_episodes = 70000

                # points_switch_random_job_type = [nb_episodes // 6, nb_episodes // 6 * 2, nb_episodes // 6 * 3, nb_episodes // 6 * 4]
                points_switch_random_job_type = [10000, 20000]
                for episode in range(nb_episodes + 1):
                    jobs = []
                    self.job_id = 0

                    # njob=1
                    if episode < points_switch_random_job_type[0]:
                        n_job_per_time = 1
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # njob=2
                    elif episode < points_switch_random_job_type[1]:
                        n_job_per_time = 2
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    # njob=3
                    else:
                        n_job_per_time = 3
                        for i in range(self.max_t + 1):
                            for _ in range(n_job_per_time):
                                submit_time = i
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = np.random.randint(0, 2)
                                jobs.append(
                                    self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                                )

                    self.jobs_set[episode] = jobs

                prm_job = [self.nb_steps, self.max_t, points_switch_random_job_type[0], points_switch_random_job_type[1]]
                self.df_prmj = pd.DataFrame(
                    [prm_job],
                    columns=['n_steps', 'max_time', 'point_switch_job_type1', 'point_switch_job_type2'],
                    index=['idx']
                )


        # 処理するジョブを選択できるdqnに有利でヒューリスティックに不利なジョブを生成
        elif self.job_type == 5:
            # 最大時間を入力
            #         max_t = int(input('input max time \n >'))
            # 単位時間あたりのジョブ数を入力
            while True:
                n_job_per_time = int(input('input number of jobs per time (-1:random) \n >'))
                n_job_per_time_is_random = False
                if 1 <= n_job_per_time <= 10:
                    break
                elif n_job_per_time == -1:
                    n_job_per_time_is_random = True
                    break
                else:
                    print('input again')

            for episode in range(self.nb_episodes + 1):
                jobs = []
                self.job_id = 0
                for i in range(self.max_t + 1):
                    if n_job_per_time_is_random:
                        n_job_per_time = np.random.randint(1, 6)
                    if np.random.uniform() < 0.3:  # たまに選択が大事なジョブ群がくる
                        for j in range(n_job_per_time):
                            submit_time = i
                            if j == 0:  # 一つ目は
                                # オンプレミスをほとんど埋めるジョブ
                                processing_time = self.n_window - 1
                                n_required_nodes = self.n_on_premise_node - 1
                                can_use_cloud = np.random.randint(0, 2)
                            else:
                                processing_time = np.random.randint(1, 3)
                                n_required_nodes = np.random.randint(1, 3)
                                can_use_cloud = 0
                            jobs.append(
                                self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                            )

                    else:
                        for j in range(n_job_per_time):
                            submit_time = i
                            processing_time = np.random.randint(1, 3)
                            n_required_nodes = np.random.randint(1, 3)
                            can_use_cloud = np.random.randint(0, 2)
                            jobs.append(
                                self.generate_job(submit_time, processing_time, n_required_nodes, can_use_cloud)
                            )

                self.jobs_set[episode] = jobs

        return self.jobs_set

    # ジョブ単体を生成
    def generate_job(self, submit_time, processing_time, n_required_nodes, can_use_cloud, user_id,  waiting_time=-1):
        job = [submit_time, processing_time, n_required_nodes, can_use_cloud, user_id, self.job_id, waiting_time,0]
        self.job_id += 1

        return job

    # def save_jobs_set(self, path):
    #     with open(path, 'w') as f:
    #         json.dump(self.jobs_set, f, indent=5)

    # if __name__ == '__main__':
    #     #jobを生成
    #     job_generator = JobGenerator()
    #     job_generator.generate_jobs_set()