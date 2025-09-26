import os
import re
import numpy as np
from typing import Dict, List, Tuple

def extract_cost_waiting_time(directory_path: str) -> Dict[int, Tuple[List[float], List[float]]]:
    """
    指定されたディレクトリから各世代のコストと待ち時間を抽出する
    
    Args:
        directory_path (str): 世代ファイルが格納されているディレクトリのパス
        
    Returns:
        Dict[int, Tuple[List[float], List[float]]]: 
            世代番号をキーとし、(コストのリスト, 待ち時間のリスト)を値とする辞書
    """
    
    # 結果を格納する辞書
    generation_data = {}
    
    # ディレクトリ内のファイルを取得
    try:
        files = os.listdir(directory_path)
    except FileNotFoundError:
        print(f"エラー: ディレクトリ '{directory_path}' が見つかりません")
        return {}
    
    # solutions_generation_XXX.txt 形式のファイルをフィルタリング
    generation_files = [f for f in files if f.startswith('solutions_generation_') and f.endswith('.txt')]
    
    if not generation_files:
        print(f"エラー: ディレクトリ '{directory_path}' に世代ファイルが見つかりません")
        return {}
    
    # 各世代ファイルを処理
    for filename in sorted(generation_files):
        # 世代番号を抽出
        match = re.search(r'solutions_generation_(\d+)\.txt', filename)
        if not match:
            continue
            
        generation_num = int(match.group(1))
        file_path = os.path.join(directory_path, filename)
        
        print(f"処理中: {filename} (世代 {generation_num})")
        
        costs = []
        waiting_times = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # ヘッダー行をスキップしてデータ行を処理
            for line in lines:
                # タブで区切られた行を処理
                if '\t' in line:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        try:
                            # コストと待ち時間を抽出
                            cost = float(parts[1])
                            waiting_time = float(parts[2])
                            
                            costs.append(cost)
                            waiting_times.append(waiting_time)
                        except ValueError:
                            # 数値変換できない行はスキップ
                            continue
                            
        except Exception as e:
            print(f"警告: ファイル '{filename}' の処理中にエラーが発生しました: {e}")
            continue
        
        # 結果を辞書に格納
        generation_data[generation_num] = (costs, waiting_times)
        print(f"  抽出完了: コスト {len(costs)}件, 待ち時間 {len(waiting_times)}件")
    
    return generation_data

def print_summary(generation_data: Dict[int, Tuple[List[float], List[float]]]):
    """
    抽出されたデータのサマリーを表示する
    
    Args:
        generation_data (Dict[int, Tuple[List[float], List[float]]]): 抽出されたデータ
    """
    print("\n" + "="*60)
    print("抽出結果サマリー")
    print("="*60)
    
    for generation in sorted(generation_data.keys()):
        costs, waiting_times = generation_data[generation]
        
        if costs and waiting_times:
            print(f"世代 {generation:3d}: コスト {len(costs):3d}件, 待ち時間 {len(waiting_times):3d}件")
            print(f"          コスト範囲: {min(costs):8.2f} - {max(costs):8.2f}")
            print(f"          待ち時間範囲: {min(waiting_times):6.2f} - {max(waiting_times):6.2f}")
        else:
            print(f"世代 {generation:3d}: データなし")

def save_to_numpy(directory_path: str, output_file: str = "extracted_data.npz"):
    """
    抽出されたデータをNumPy形式で保存する
    
    Args:
        directory_path (str): 世代ファイルが格納されているディレクトリのパス
        output_file (str): 出力ファイル名
    """
    generation_data = extract_cost_waiting_time(directory_path)
    
    if not generation_data:
        print("保存するデータがありません")
        return
    
    # データをNumPy配列に変換
    generations = sorted(generation_data.keys())
    max_solutions = max(len(generation_data[gen][0]) for gen in generations)
    
    # コストと待ち時間の配列を作成（不足分はNaNで埋める）
    costs_array = np.full((len(generations), max_solutions), np.nan)
    waiting_times_array = np.full((len(generations), max_solutions), np.nan)
    
    for i, generation in enumerate(generations):
        costs, waiting_times = generation_data[generation]
        costs_array[i, :len(costs)] = costs
        waiting_times_array[i, :len(waiting_times)] = waiting_times
    
    # 保存
    np.savez(output_file, 
              generations=np.array(generations),
              costs=costs_array,
              waiting_times=waiting_times_array)
    
    print(f"\nデータを '{output_file}' に保存しました")
    print(f"形状: コスト {costs_array.shape}, 待ち時間 {waiting_times_array.shape}")

if __name__ == "__main__":
    # 対象ディレクトリのパス
    target_directory = "execution_nsga_32-256"
    
    print(f"ディレクトリ '{target_directory}' からデータを抽出中...")
    
    # データを抽出
    data = extract_cost_waiting_time(target_directory)
    
    if data:
        # サマリーを表示
        print_summary(data)
        
        # NumPy形式で保存
        save_to_numpy(target_directory)
        
        # 個別の世代データにアクセスする例
        print("\n" + "="*60)
        print("データアクセス例")
        print("="*60)
        
        for generation in sorted(data.keys())[:3]:  # 最初の3世代のみ表示
            costs, waiting_times = data[generation]
            print(f"世代 {generation}:")
            print(f"  コスト配列: {costs[:5]}...")  # 最初の5件のみ表示
            print(f"  待ち時間配列: {waiting_times[:5]}...")
            print()
    else:
        print("データの抽出に失敗しました") 