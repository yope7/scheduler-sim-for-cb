import numpy as np

class JobSimulator:
    def __init__(self, seed, n_jobs=50, n_users=50, lam=0.5, mean_processing_time=10, std_processing_time=2, min_processing_time=1, max_processing_time=19, min_required_nodes=1, max_required_nodes=9):
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

    def generate_jobs(self):
        # ランダムシードを固定
        np.random.seed(self.seed)
        #到着時間は指数分布に基づく整数値
        inter_arrivals = np.random.exponential(self.lam, self.n_jobs)
        arrival_times = np.ceil(np.cumsum(inter_arrivals))
        processing_times = np.clip(np.random.normal(self.mean_processing_time, self.std_processing_time, self.n_jobs), self.min_processing_time , self.max_processing_time).astype(int)  
        required_nodes = np.random.randint(self.min_required_nodes, self.max_required_nodes, self.n_jobs)
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
