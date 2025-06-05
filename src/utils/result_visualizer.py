import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def parse_data(input_text: str) -> List[Tuple[str, np.ndarray]]:
    """
    入力テキストを解析して、データセットのリストを返す
    
    Args:
        input_text (str): 入力テキスト（データ名とデータのペア）
    
    Returns:
        List[Tuple[str, np.ndarray]]: データセットのリスト（データ名とデータのペア）
    """
    lines = input_text.strip().split('\n')
    datasets = []
    current_name = None
    current_data = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('[') and line.endswith(']'):
            # データ行の場合
            data = np.array(eval(line))
            current_data.append(data)
        else:
            # データ名の場合
            if current_name and current_data:
                datasets.append((current_name, np.array(current_data)))
            current_name = line
            current_data = []
    
    # 最後のデータセットを追加
    if current_name and current_data:
        datasets.append((current_name, np.array(current_data)))
    
    return datasets

def visualize_results(input_text: str, title: str = "Results Visualization"):
    """
    入力テキストを解析してグラフを作成する
    
    Args:
        input_text (str): 入力テキスト
        title (str): グラフのタイトル
    """
    datasets = parse_data(input_text)
    
    plt.figure(figsize=(10, 6))
    
    for name, data in datasets:
        x = data[:, 0]  # 最初の列をx軸として使用
        y = data[:, 1]  # 2番目の列をy軸として使用
        plt.plot(x, y, marker='o', label=name)
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.show()

def interactive_input():
    """
    対話的にデータを入力する
    """
    print("データの可視化を開始します。")
    print("データセットを入力してください。")
    print("データ名を入力した後、データを入力し、空行で区切ってください。")
    print("終了するには 'q' を入力してください。")
    print("-" * 50)
    
    all_input = []
    while True:
        # データ名の入力
        name = input("\nデータ名を入力してください（終了: q）: ").strip()
        if name.lower() == 'q':
            break
            
        print(f"\n{name}のデータを入力してください（空行で終了）:")
        data_lines = []
        while True:
            line = input().strip()
            if not line:
                break
            data_lines.append(line)
        
        if data_lines:
            all_input.append(name)
            all_input.extend(data_lines)
            all_input.append("")  # データセット間の区切り
    
    if all_input:
        input_text = "\n".join(all_input)
        title = input("\nグラフのタイトルを入力してください（デフォルト: Results Visualization）: ").strip()
        if not title:
            title = "Results Visualization"
        visualize_results(input_text, title)
    else:
        print("データが入力されませんでした。")

if __name__ == "__main__":
    interactive_input()
