#!/usr/bin/env python3
import sys
import os
import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.job_gen.job_generator import JobGenerator

def load_config():
    with open('config/config.yml', 'r') as yml:
        config = yaml.safe_load(yml)
    return config

config = load_config()
job_generator = JobGenerator(
    0, 1,
    config['param_env']['n_window'],
    config['param_env']['n_on_premise_node'],
    config['param_env']['n_cloud_node'],
    config, 128, 0.2, 0
)
jobs_set = job_generator.generate_jobs_set()
jobs = jobs_set[0]

print("ジョブの詳細:")
print(f"ジョブ数: {len(jobs)}")
print("\n最初の5ジョブ:")
for i in range(min(5, len(jobs))):
    print(f"ジョブ{i}: {jobs[i]}")
    print(f"  到着時刻: {jobs[i][0]}, 処理時間: {jobs[i][1]}, ノード数: {jobs[i][2]}")

print("\n統計:")
print(f"到着時刻範囲: {np.min(jobs[:, 0])} - {np.max(jobs[:, 0])}")
print(f"処理時間範囲: {np.min(jobs[:, 1])} - {np.max(jobs[:, 1])}")
print(f"ノード数範囲: {np.min(jobs[:, 2])} - {np.max(jobs[:, 2])}")





