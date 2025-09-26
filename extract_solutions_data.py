#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solutionsファイルからコストと待ち時間を抽出し、世代ごとのデータを整理するスクリプト
"""

import os
import re
import json
from pathlib import Path

def extract_solutions_data(directory_path):
    """
    指定されたディレクトリ内のsolutionsファイルからデータを抽出
    
    Args:
        directory_path (str): solutionsファイルが格納されているディレクトリのパス
    
    Returns:
        dict: 世代ごとの解のリスト（コスト、待ち時間）
    """
    solutions_data = {}
    
    # ディレクトリ内のファイルを取得
    directory = Path(directory_path)
    solution_files = [f for f in directory.glob("solutions_generation_*.txt")]
    
    for file_path in solution_files:
        # ファイル名から世代番号を抽出
        match = re.search(r'solutions_generation_(\d+)\.txt', file_path.name)
        if match:
            generation = int(match.group(1))
            
            # ファイルの内容を読み込み
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # コストと待ち時間を抽出
            solutions = []
            lines = content.split('\n')
            
            for line in lines:
                # 個体の行を検索（タブ区切りで個体ID、コスト、待ち時間、染色体の順）
                if re.match(r'^\d+\t', line):
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        try:
                            cost = float(parts[1])
                            waiting_time = float(parts[2])
                            solutions.append({
                                'cost': cost,
                                'waiting_time': waiting_time
                            })
                        except ValueError:
                            continue
            
            if solutions:
                solutions_data[generation] = solutions
                print(f"世代 {generation}: {len(solutions)} 個の解を抽出")
    
    return solutions_data

def save_extracted_data(data, output_file):
    """
    抽出したデータをJSONファイルに保存
    
    Args:
        data (dict): 抽出したデータ
        output_file (str): 出力ファイル名
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"データを {output_file} に保存しました")

if __name__ == "__main__":
    # ディレクトリパスを指定
    directory_path = "execution_20250818_020958"
    
    # データを抽出
    print("solutionsファイルからデータを抽出中...")
    solutions_data = extract_solutions_data(directory_path)
    
    # 世代順にソート
    sorted_generations = sorted(solutions_data.keys())
    print(f"\n抽出完了: {len(sorted_generations)} 世代のデータ")
    print(f"世代: {sorted_generations}")
    
    # データを保存
    output_file = "extracted_solutions_data.json"
    save_extracted_data(solutions_data, output_file)
    
    # 各世代の解の数を表示
    print("\n各世代の解の数:")
    for gen in sorted_generations:
        print(f"世代 {gen}: {len(solutions_data[gen])} 個") 