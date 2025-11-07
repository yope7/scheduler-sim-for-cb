#!/usr/bin/env python3
"""
プロファイルファイルをSnakeVizで使用可能にするため、破損した参照を削除
"""
import marshal
import struct
import sys

def fix_profile(input_file, output_file):
    """プロファイルファイルを読み込んで、無効な参照を修正"""
    with open(input_file, 'rb') as f:
        # cProfileファイルのヘッダーを読み込む
        version = struct.unpack('=i', f.read(4))[0]
        print(f"プロファイルバージョン: {version}")
        
        # 統計データを読み込む
        stats = marshal.load(f)
        
        # プロジェクト関連の関数名を収集
        valid_keys = set()
        for func_name in stats.keys():
            filename = func_name[0] if isinstance(func_name, tuple) and len(func_name) > 0 else ''
            if any(kw in filename for kw in ['scheduling_env', 'heuristic_agent', 'job_generator', 'test_large_scale_timing']):
                valid_keys.add(func_name)
        
        print(f"保持する関数数: {len(valid_keys)}")
        
        # 無効な参照を削除した新しい統計を作成
        new_stats = {}
        for func_name, func_data in stats.items():
            if func_name in valid_keys:
                cc, nc, tt, ct, callers = func_data
                # 呼び出し元もフィルタリング
                filtered_callers = {k: v for k, v in callers.items() if k in valid_keys}
                new_stats[func_name] = (cc, nc, tt, ct, filtered_callers)
        
        print(f"フィルタリング後の関数数: {len(new_stats)}")
        
        # 新しいファイルに書き込む
        with open(output_file, 'wb') as out:
            out.write(struct.pack('=i', version))
            marshal.dump(new_stats, out)
        
        print(f"修正済みプロファイルを保存しました: {output_file}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使用法: python fix_profile_for_snakeviz.py <入力プロファイル> [出力プロファイル]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.prof', '_fixed.prof')
    
    try:
        fix_profile(input_file, output_file)
        print(f"\n可視化するには: snakeviz {output_file}")
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()










