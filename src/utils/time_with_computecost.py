# 計算コストと計算時間の関係を可視化する
# 外側からall や pcnを呼び出し，その計算時間を計測する
# その際パラメータを変えて複数実行する．

import subprocess
import matplotlib.pyplot as plt
import re

def parse_time_output(time_str):
    # timeコマンドの出力から実時間を抽出
    match = re.search(r"real\s+(\d+)m(\d+\.\d+)s", time_str)
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    return None

def measure_computation_time():
    # ジョブ数の範囲を設定
    job_numbers = range(5, 15)  # 5から15まで
    computation_times = []

    for n_jobs in job_numbers:
        # timeコマンドで実行時間を計測
        cmd = f"time python3 -m scripts.main --mode all --nb_jobs {n_jobs}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # timeコマンドの出力から実時間を抽出
        time_output = result.stderr
        real_time = parse_time_output(time_output)
        
        if real_time is not None:
            computation_times.append(real_time)
            print(f"ジョブ数 {n_jobs}: 計算時間 {real_time:.2f}秒")
        else:
            print(f"ジョブ数 {n_jobs}: 時間の計測に失敗しました")

    # グラフの作成
    plt.figure(figsize=(10, 6))
    plt.plot(job_numbers, computation_times, marker='o')
    plt.xlabel('ジョブ数')
    plt.ylabel('計算時間 (秒)')
    plt.title('ジョブ数と計算時間の関係')
    plt.grid(True)
    
    # グラフの保存
    plt.savefig('plot/computation_time_vs_jobs.png')
    plt.close()

if __name__ == "__main__":
    measure_computation_time()

