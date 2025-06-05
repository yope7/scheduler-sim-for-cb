import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import os
from datetime import datetime

def parse_data(input_text: str) -> List[Tuple[str, np.ndarray]]:
    """
    入力テキストを解析して、データセットのリストを返す
    
    Args:
        input_text (str): 入力テキスト（データ名とデータのペア）
    
    Returns:
        List[Tuple[str, np.ndarray]]: データセットのリスト（データ名とデータのペア）
    """
    # 入力テキストを行ごとに分割
    lines = input_text.strip().split('\n')
    datasets = []
    current_name = None
    current_data = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 配列形式の行を処理
        if line.startswith('[') and line.endswith(']'):
            try:
                # 数値を抽出して配列に変換
                values = line.strip('[]').split()
                if len(values) == 2:  # x, y の2つの値がある場合
                    x, y = map(float, values)
                    current_data.append([x, y])
            except Exception as e:
                print(f"警告: データ行の解析に失敗しました: {line}")
                continue
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

def visualize_results(input_text: str, title: str = "Results Visualization", save_path: str = None) -> bool:
    """
    入力テキストを解析してグラフを作成する
    
    Args:
        input_text (str): 入力テキスト
        title (str): グラフのタイトル
        save_path (str): 保存先のパス（Noneの場合は保存しない）
    
    Returns:
        bool: ファイルが保存された場合はTrue、それ以外はFalse
    """
    datasets = parse_data(input_text)
    
    if not datasets:
        print("警告: データが見つかりませんでした。")
        return False
    
    plt.figure(figsize=(10, 6))
    
    for name, data in datasets:
        x = data[:, 0]  # 最初の列をx軸として使用
        y = data[:, 1]  # 2番目の列をy軸として使用
        plt.scatter(x, y, label=name)
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        # plotディレクトリの存在チェックと作成
        plot_dir = os.path.dirname(save_path)
        if plot_dir and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(save_path)
        print(f"グラフを保存しました: {save_path}")
        plt.close()  # グラフを閉じる
        return True
    else:
        plt.show()
        return False

def read_data_file(file_path: str) -> str:
    """
    データファイルを読み込む
    
    Args:
        file_path (str): データファイルのパス
    
    Returns:
        str: ファイルの内容
    """
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"エラー: ファイルの読み込みに失敗しました: {e}")
        return ""

def interactive_input():
    """
    対話的にデータを入力する
    """
    print("データの可視化を開始します。")
    print("入力方法を選択してください:")
    print("1. ファイルから読み込む")
    print("2. 手動で入力する")
    print("3. 終了")
    
    while True:
        choice = input("\n選択してください (1-3): ").strip()
        
        if choice == '1':
            # ファイルから読み込む
            input_dir = 'plot/input_data'
            if not os.path.exists(input_dir):
                print(f"エラー: ディレクトリ {input_dir} が存在しません。")
                continue
                
            files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
            if not files:
                print(f"エラー: {input_dir} に.txtファイルが見つかりません。")
                continue
                
            print("\n利用可能なファイル:")
            for i, file in enumerate(files, 1):
                print(f"{i}. {file}")
            
            try:
                file_choice = int(input("\nファイル番号を選択してください: ").strip())
                if 1 <= file_choice <= len(files):
                    file_path = os.path.join(input_dir, files[file_choice - 1])
                    input_text = read_data_file(file_path)
                    if input_text:
                        title = input("\nグラフのタイトルを入力してください（デフォルト: Results Visualization）: ").strip()
                        if not title:
                            title = "Results Visualization"
                            
                        save_option = input("\nグラフを保存しますか？ (y/n, デフォルト: n): ").strip().lower()
                        save_path = None
                        if save_option == 'y':
                            filename = input("保存するファイル名を入力してください（例: result.png）: ").strip()
                            if filename:
                                save_path = os.path.join('plot', filename)
                            else:
                                # ファイル名が入力されなかった場合は、入力ファイル名から自動生成
                                base_name = os.path.splitext(files[file_choice - 1])[0]
                                save_path = os.path.join('plot', f"{base_name}.png")
                        
                        if visualize_results(input_text, title, save_path):
                            print("プログラムを終了します。")
                            return
                else:
                    print("エラー: 無効なファイル番号です。")
            except ValueError:
                print("エラー: 有効な数字を入力してください。")
                
        elif choice == '2':
            # 手動入力
            print("\nデータの入力方法:")
            print("1. 最初にデータセットの名前を入力")
            print("2. 次に、NumPy配列形式でデータを入力（例: [110 39]）")
            print("3. 新しいデータセットを始めるには、新しい名前を入力")
            print("4. 終了するには 'q' を入力")
            print("-" * 50)
            
            all_input = []
            while True:
                # データ名の入力
                name = input("\nデータセット名を入力してください（終了: q）: ").strip()
                if name.lower() == 'q':
                    break
                    
                print(f"\n{name}のデータを入力してください（NumPy配列形式、空行で終了）:")
                all_input.append(name)
                
                while True:
                    line = input().strip()
                    if not line:
                        break
                    all_input.append(line)
            
            if all_input:
                input_text = "\n".join(all_input)
                title = input("\nグラフのタイトルを入力してください（デフォルト: Results Visualization）: ").strip()
                if not title:
                    title = "Results Visualization"
                    
                save_option = input("\nグラフを保存しますか？ (y/n, デフォルト: n): ").strip().lower()
                save_path = None
                if save_option == 'y':
                    filename = input("保存するファイル名を入力してください（例: result.png）: ").strip()
                    if filename:
                        save_path = os.path.join('plot', filename)
                    else:
                        # ファイル名が入力されなかった場合は、タイムスタンプを使用
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = os.path.join('plot', f"plot_{timestamp}.png")
                
                if visualize_results(input_text, title, save_path):
                    print("プログラムを終了します。")
                    return
            else:
                print("データが入力されませんでした。")
                
        elif choice == '3':
            print("プログラムを終了します。")
            break
            
        else:
            print("エラー: 1-3の数字を入力してください。")

if __name__ == "__main__":
    interactive_input()
