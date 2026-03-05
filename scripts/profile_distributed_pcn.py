#!/usr/bin/env python3
"""
分散PCNのプロファイリングスクリプト

使用方法:
  # 短時間プロファイリング（推奨: イテレーション3回、Actor 4）
  python scripts/profile_distributed_pcn.py --quick

  # py-spyによるサンプリングプロファイリング（全プロセス）
  python scripts/profile_distributed_pcn.py --py-spy --duration 60

  # cProfileによる deterministic プロファイリング
  python scripts/profile_distributed_pcn.py --cprofile --output profile_distpcn.prof

  # 詳細タイミングのみ（イテレーション内の各処理時間）
  python scripts/profile_distributed_pcn.py --timing

本番の高速化:
  DISTRIBUTED_PCN_FAST=1 python -m src.distributed.distributed_pcn
  # N_UPDATESを5→3に削減（約40%のLearner時間短縮）
"""
import argparse
import os
import sys
import subprocess
import time

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)


def run_with_timing():
    """詳細タイミング計測付きで分散PCNを実行"""
    # 環境変数でプロファイリングモードを有効化
    os.environ['DISTRIBUTED_PCN_PROFILE'] = '1'
    os.environ['DISTRIBUTED_PCN_QUICK'] = '1'  # イテレーション3, Actor 4
    from src.distributed.distributed_pcn import main
    main()


def run_with_cprofile(output_path: str):
    """cProfileで分散PCNを実行"""
    import cProfile
    import pstats
    
    os.environ['DISTRIBUTED_PCN_QUICK'] = '1'
    
    prof = cProfile.Profile()
    prof.enable()
    try:
        from src.distributed.distributed_pcn import main
        main()
    finally:
        prof.disable()
        prof.dump_stats(output_path)
        print(f"\nプロファイリング結果を {output_path} に保存しました")
        print(f"可視化: snakeviz {output_path}")
        
        # 簡易サマリーを表示
        s = __import__('io').StringIO()
        stats = pstats.Stats(prof, stream=s)
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats(30)
        print("\n--- Top 30 (累積時間) ---")
        print(s.getvalue())


def run_with_pyspy(duration: int, output_path: str):
    """py-spyでサンプリングプロファイリング（子プロセス含む）"""
    try:
        import py_spy
    except ImportError:
        print("py-spyがインストールされていません。uv add py-spy でインストールしてください")
        sys.exit(1)
    
    os.environ['DISTRIBUTED_PCN_QUICK'] = '1'
    
    cmd = [sys.executable, '-m', 'src.distributed.distributed_pcn']
    output_file = output_path or f'distpcn_pyspy_{int(time.time())}.speedscope.json'
    
    print(f"py-spyで{duration}秒間プロファイリングを開始...")
    print(f"出力: {output_file}")
    
    # py-spyをサブプロセスとして実行（--subprocesses でRayワーカーも含む）
    result = subprocess.run([
        'py-spy', 'record',
        '--format', 'speedscope',
        '--output', output_file,
        '--duration', str(duration),
        '--subprocesses',
        '--', sys.executable, '-m', 'src.distributed.distributed_pcn'
    ], cwd=project_root)
    
    if result.returncode == 0:
        print(f"\nプロファイリング完了。結果: {output_file}")
        print("可視化: https://www.speedscope.app/ で開く")
    else:
        print("py-spyが失敗しました。sudoで実行する必要がある場合: sudo py-spy record ...")


def main():
    parser = argparse.ArgumentParser(description='分散PCNプロファイリング')
    parser.add_argument('--quick', action='store_true', help='短時間プロファイリング（イテレーション3, Actor 4）')
    parser.add_argument('--timing', action='store_true', help='詳細タイミング計測を有効化')
    parser.add_argument('--cprofile', action='store_true', help='cProfileでプロファイリング')
    parser.add_argument('--py-spy', action='store_true', help='py-spyでサンプリングプロファイリング')
    parser.add_argument('--duration', type=int, default=60, help='py-spyの計測時間（秒）')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルパス')
    
    args = parser.parse_args()
    
    if args.py_spy:
        run_with_pyspy(args.duration, args.output)
    elif args.cprofile:
        output = args.output or 'profile_distributed_pcn.prof'
        run_with_cprofile(output)
    elif args.timing or args.quick:
        run_with_timing()
    else:
        print("オプションを指定してください: --quick, --timing, --cprofile, --py-spy")
        parser.print_help()


if __name__ == '__main__':
    main()
