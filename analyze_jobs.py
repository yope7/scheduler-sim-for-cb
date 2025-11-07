#!/usr/bin/env python3
"""ジョブの統計情報を分析"""

import sys
import os
import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.job_gen.job_generator import JobGenerator

def load_config():
    """設定ファイルを読み込み"""
    with open('config/config.yml', 'r') as yml:
        config = yaml.safe_load(yml)
    return config

def analyze_jobs(nb_jobs: int, seed: int = 0):
    """ジョブの統計情報を分析"""
    config = load_config()
    
    # JobGeneratorでジョブセットを作成
    lam = config['param_job'].get('lam', 0.2)
    
    job_generator = JobGenerator(
        seed, 1,
        config['param_env']['n_window'],
        config['param_env']['n_on_premise_node'],
        config['param_env']['n_cloud_node'],
        config, nb_jobs, lam, 0
    )
    
    jobs_set = job_generator.generate_jobs_set()
    
    # エピソード0のジョブを取得
    jobs = jobs_set[0]
    
    # ジョブの統計情報を計算
    processing_times = jobs[:, 1]  # 処理時間
    required_nodes = jobs[:, 2]   # ノード数
    arrival_times = jobs[:, 0]    # 到着時刻
    
    print(f"ジョブ数: {len(jobs)}")
    print(f"\n処理時間の統計:")
    print(f"  平均: {np.mean(processing_times):.2f}")
    print(f"  標準偏差: {np.std(processing_times):.2f}")
    print(f"  最小値: {np.min(processing_times)}")
    print(f"  最大値: {np.max(processing_times)}")
    print(f"  中央値: {np.median(processing_times):.2f}")
    
    print(f"\nノード数の統計:")
    print(f"  平均: {np.mean(required_nodes):.2f}")
    print(f"  標準偏差: {np.std(required_nodes):.2f}")
    print(f"  最小値: {np.min(required_nodes)}")
    print(f"  最大値: {np.max(required_nodes)}")
    print(f"  中央値: {np.median(required_nodes):.2f}")
    
    print(f"\n到着時刻の統計:")
    print(f"  最初の到着: {np.min(arrival_times)}")
    print(f"  最後の到着: {np.max(arrival_times)}")
    print(f"  平均到着間隔: {np.mean(np.diff(arrival_times)):.2f}")
    
    # ジョブのサイズ分布（処理時間 × ノード数）
    job_sizes = processing_times * required_nodes
    print(f"\nジョブサイズ（処理時間 × ノード数）の統計:")
    print(f"  平均: {np.mean(job_sizes):.2f}")
    print(f"  標準偏差: {np.std(job_sizes):.2f}")
    print(f"  最小値: {np.min(job_sizes)}")
    print(f"  最大値: {np.max(job_sizes)}")
    print(f"  中央値: {np.median(job_sizes):.2f}")
    
    # 大規模ジョブの割合
    large_jobs = np.sum(job_sizes > 100)
    print(f"\n大規模ジョブ（サイズ>100）の数: {large_jobs} ({large_jobs/len(jobs)*100:.1f}%)")
    
    # 超大型ジョブの割合
    very_large_jobs = np.sum(job_sizes > 500)
    print(f"超大型ジョブ（サイズ>500）の数: {very_large_jobs} ({very_large_jobs/len(jobs)*100:.1f}%)")

if __name__ == "__main__":
    analyze_jobs(128, seed=0)




