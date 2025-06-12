# 計算コストと計算時間の関係を可視化する
# 外側からall や pcnを呼び出し，その計算時間を計測する
# その際パラメータを変えて複数実行する．

import subprocess
import matplotlib.pyplot as plt
import re
import argparse

def parse_time_output(time_str):
    # timeコマンドの出力から実時間を抽出
    # 形式1: real    0m0.125s
    match = re.search(r"real\s+(\d+)m(\d+\.\d+)s", time_str)
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    
    # 形式2: 0:09.43elapsed
    match = re.search(r"(\d+):(\d+\.\d+)elapsed", time_str)
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    
    return None

def parse_time_file(file_path):
    """時間比較ファイルからデータを解析する"""
    all_times = {}
    distributed_times = {}
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    current_mode = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line == "all mode:":
            current_mode = "all"
            continue
        elif line == "all_distributed mode:":
            current_mode = "distributed"
            continue
            
        if current_mode:
            # ジョブ数と時間を抽出
            match = re.search(r"ジョブ数 (\d+): (?:計算時間 )?(\d+\.\d+)秒", line)
            if match:
                job_num = int(match.group(1))
                time = float(match.group(2))
                
                if current_mode == "all":
                    all_times[job_num] = time
                else:
                    distributed_times[job_num] = time
    
    # 共通のジョブ数を取得
    common_job_numbers = sorted(set(all_times.keys()) & set(distributed_times.keys()))
    
    # 共通のジョブ数に対する時間を取得
    all_times_list = [all_times[job_num] for job_num in common_job_numbers]
    distributed_times_list = [distributed_times[job_num] for job_num in common_job_numbers]
    
    return common_job_numbers, all_times_list, distributed_times_list

def plot_from_file(file_path):
    """ファイルから直接グラフを作成する"""
    job_numbers, all_times, distributed_times = parse_time_file(file_path)
    
    plt.figure(figsize=(12, 8))
    
    # allモードの結果をプロット
    plt.plot(job_numbers, all_times, marker='o', label='all mode', color='blue')
    
    # all_distributedモードの結果をプロット
    plt.plot(job_numbers, distributed_times, marker='s', label='all_distributed mode', color='red')
    
    plt.xlabel('Number of Jobs')
    plt.ylabel('Computation Time (seconds)')
    plt.title('Comparison of Computation Time between all and all_distributed modes')
    plt.grid(True)
    plt.legend()
    
    # グラフの保存
    plt.savefig('plot/computation_time_comparison.png')
    plt.close()
    
    print("グラフを生成しました: plot/computation_time_comparison.png")

def measure_computation_time():
    # ジョブ数の範囲を設定
    job_numbers = range(16, 20)  # 5から15まで
    computation_times_all = []
    computation_times_distributed = []

    # allモードの実験
    print("\n=== allモードの実験を開始 ===")
    for n_jobs in job_numbers:
        try:
            # timeコマンドで実行時間を計測
            cmd = f"time python3 -m scripts.main --mode all --nb_jobs {n_jobs}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            # timeコマンドの出力から実時間を抽出
            time_output = result.stderr
            real_time = parse_time_output(time_output)
            
            if real_time is not None:
                computation_times_all.append(real_time)
                print(f"ジョブ数 {n_jobs}:  {real_time:.2f}秒")
            else:
                print(f"ジョブ数 {n_jobs}: 時間の計測に失敗しました")
                print(f"出力: {time_output}")
        except Exception as e:
            print(f"ジョブ数 {n_jobs}: エラーが発生しました - {str(e)}")

    # all_distributedモードの実験
    print("\n=== all_distributedモードの実験を開始 ===")
    for n_jobs in job_numbers:
        try:
            # timeコマンドで実行時間を計測
            cmd = f"time python3 -m scripts.main --mode all_distributed --nb_jobs {n_jobs}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            # timeコマンドの出力から実時間を抽出
            time_output = result.stderr
            real_time = parse_time_output(time_output)
            
            if real_time is not None:
                computation_times_distributed.append(real_time)
                print(f"ジョブ数 {n_jobs}: {real_time:.2f}秒")
            else:
                print(f"ジョブ数 {n_jobs}: 時間の計測に失敗しました")
                print(f"出力: {time_output}")
        except Exception as e:
            print(f"ジョブ数 {n_jobs}: エラーが発生しました - {str(e)}")

    # グラフの作成
    plt.figure(figsize=(12, 8))
    
    # allモードの結果をプロット
    plt.plot(job_numbers, computation_times_all, marker='o', label='all mode', color='blue')
    
    # all_distributedモードの結果をプロット
    plt.plot(job_numbers, computation_times_distributed, marker='s', label='all_distributed mode', color='red')
    
    plt.xlabel('Number of Jobs')
    plt.ylabel('Computation Time (seconds)')
    plt.title('Comparison of Computation Time between all and all_distributed modes')
    plt.grid(True)
    plt.legend()
    
    # グラフの保存
    plt.savefig('plot/computation_time_comparison.png')
    plt.close()

    # 結果の要約を表示
    print("\n=== 実験結果の要約 ===")
    print("all mode:")
    for n_jobs, time in zip(job_numbers, computation_times_all):
        print(f"ジョブ数 {n_jobs}: {time:.2f}秒")
    
    print("\nall_distributed mode:")
    for n_jobs, time in zip(job_numbers, computation_times_distributed):
        print(f"ジョブ数 {n_jobs}: {time:.2f}秒")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='計算時間の計測と可視化')
    parser.add_argument('--file', type=str, help='時間比較ファイルのパス')
    args = parser.parse_args()
    
    if args.file:
        plot_from_file(args.file)
    else:
        measure_computation_time()

