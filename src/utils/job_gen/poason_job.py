import numpy as np

class JobSimulator:
    def __init__(self, seed, n_jobs=50, n_users=50, lam=0.3, mean_processing_time=50, std_processing_time=50, min_processing_time=1, max_processing_time=100, min_required_nodes=1, max_required_nodes=256,
                 tail_level=0.0, tail_alpha=None, tail_frac=None, tail_max_nodes=None, tail_seed=None, tail_pt_cap=None,
                 tail_burst=0):
        self.n_jobs = n_jobs
        self.n_users = n_users
        self.lam = lam
        self.seed = seed
        self.mean_processing_time = mean_processing_time
        self.std_processing_time = std_processing_time
        self.min_required_nodes = min_required_nodes
        self.max_required_nodes = max_required_nodes
        self.min_processing_time = min_processing_time
        self.max_processing_time = max_processing_time
        # --- 占有量(=pt*nodes)の power-law 裾を n_required_nodes に注入する制御ダイヤル ---
        # tail_level L∈[0,1]: 0=OFF(既定, 完全に従来挙動・ビット一致)。L↑で裾が重く・giant件数↑。
        # 裾は専用 np.random.default_rng でのみ引き、既存の np.random 系列は一切乱さない。
        self.tail_level = float(tail_level) if tail_level else 0.0
        self.tail_alpha = tail_alpha
        self.tail_frac = tail_frac
        self.tail_max_nodes = tail_max_nodes
        self.tail_seed = tail_seed
        self.tail_pt_cap = tail_pt_cap
        # tail_burst=k(>0): giant を「同一ジョブの k 連続到着クラスタ」として注入（trace の
        # バースト到着構造の再現。trace では同じ巨大ジョブが連番で並ぶ）。0=従来の散在注入。
        self.tail_burst = int(tail_burst) if tail_burst else 0

    def generate_jobs(self,seed=None):
        # ランダムシードを固定
        if seed is not None:
            np.random.seed(seed)
        #到着時間は指数分布に基づく整数値
        inter_arrivals = np.random.exponential(self.lam, self.n_jobs)
        arrival_times = np.ceil(np.cumsum(inter_arrivals))
        processing_times = np.random.randint(self.min_processing_time, self.max_processing_time, self.n_jobs)
        required_nodes = np.random.randint(self.min_required_nodes, self.max_required_nodes, self.n_jobs)
        # --- gated power-law 裾注入（既定OFF: tail_level=0 ならこのブロックは非実行＝ビット一致） ---
        # 占有量(=pt*nodes)の重い裾を processing_time に注入し、合成→trace を連続補間する。
        # trace の裾は主に pt 駆動（pt max/med≈30x, nodes≈6x, 積で≈190x）かつ「多数の中〜大 giant」
        # （>10x median 48個 / >100x 10個）。これを再現するため frac のジョブに Pareto 乗数を広く掛ける。
        # 較正: FRAC_MAX=0.8, alpha∈[0.75,2.0], cap=432006(≈trace max pt) で L=1≈trace の broad tail。
        # trace 実測点はこのダイヤル上 L≈0.85-1.0 にブラケットされる。
        if self.tail_level and self.tail_level > 0:
            L = float(self.tail_level)
            FRAC_MAX, ALPHA_MIN, ALPHA_MAX = 0.8, 0.75, 2.0
            frac = float(self.tail_frac) if self.tail_frac is not None else min(1.0, FRAC_MAX * L)
            # L↑で alpha↓＝裾が重く（Pareto inverse-CDF, 小 alpha ほど巨大乗数が出やすい）
            alpha = float(self.tail_alpha) if self.tail_alpha is not None else (ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * (1.0 - L))
            pt_cap = int(self.tail_pt_cap) if getattr(self, "tail_pt_cap", None) is not None else 432006
            n_tail = min(self.n_jobs, max(1, int(round(frac * self.n_jobs))))
            tseed = int(self.tail_seed) if self.tail_seed is not None else (int(self.seed) + 1000003)
            rng = np.random.default_rng(tseed)  # 専用ストリーム: np.random を汚さない＝OFF/base はビット一致
            processing_times = processing_times.astype(np.int64)
            if self.tail_burst > 0:
                # バースト注入: giant を「同一 pt の k 連続ジョブ」クラスタで配置（到着は cumsum 順
                # = index 順なので、連続 index = 連続到着）。trace の同一巨大ジョブ連番を模す。
                k = self.tail_burst
                n_cl = max(1, int(round(n_tail / k)))
                starts = rng.choice(max(1, self.n_jobs - k), size=n_cl, replace=False)
                U = rng.random(n_cl)
                mult = (1.0 - U) ** (-1.0 / max(alpha, 0.3))
                for s, m in zip(starts, mult):
                    val = int(np.clip(round(processing_times[s] * m), 1, pt_cap))
                    processing_times[s:s + k] = val  # 同一ジョブの k 連続到着
            else:
                idx = rng.choice(self.n_jobs, size=n_tail, replace=False)
                U = rng.random(n_tail)
                mult = (1.0 - U) ** (-1.0 / max(alpha, 0.3))  # Pareto(xmin=1) 乗数(>=1), 重い右裾
                processing_times[idx] = np.clip(
                    np.round(processing_times[idx] * mult), 1, pt_cap
                ).astype(np.int64)
        can_use_cloud = np.full(self.n_jobs, 1)
        user_ids = np.full(self.n_jobs, 0)
        job_ids = np.arange(self.n_jobs)
        waiting_time = -1
        init_value = -1
        jobs = np.column_stack((
            arrival_times,
            processing_times,
            required_nodes,
            can_use_cloud,
            user_ids,
            job_ids,
            np.full(self.n_jobs, waiting_time),
            np.full(self.n_jobs, init_value)
        ))
        return jobs

# 使用例
if __name__ == "__main__":
    seed = 42
    simulator = JobSimulator(seed, n_jobs=30, n_users=50, lam=1)
    #lamは大きいほどスカスカ
    jobs = simulator.generate_jobs()
    print("Simulated Jobs:")
    for job in jobs:
        print(job)
