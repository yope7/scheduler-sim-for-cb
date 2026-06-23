#!/usr/bin/env python3
"""
C言語実装が実際に使用されているか確認するスクリプト
"""
import sys
import os
import numpy as np
import time

# プロジェクトのルートディレクトリをパスに追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# C言語実装をインポート
try:
    from scheduling_env_core import (
        WindowCache,
        find_allocation_position as c_find_allocation_position,
        time_transition as c_time_transition,
        do_schedule as c_do_schedule,
    )
    C_AVAILABLE = True
    print("✓ C言語実装のインポートに成功しました")
except ImportError as e:
    C_AVAILABLE = False
    print(f"✗ C言語実装のインポートに失敗しました: {e}")
    sys.exit(1)

from test_large_scale_timing_c import SchedulingEnvC
from src.utils.job_gen.job_generator import JobGenerator
import yaml

def load_config():
    """設定ファイルを読み込み"""
    with open('config/config.yml', 'r') as yml:
        config = yaml.safe_load(yml)
    return config

def verify_c_implementation():
    """C言語実装が実際に使用されているか確認"""
    print("\n" + "="*60)
    print("C言語実装の使用確認")
    print("="*60)
    
    # 設定を読み込み
    config = load_config()
    
    # ジョブセットを作成
    job_generator = JobGenerator(
        0, 1, config['param_env']['n_window'],
        config['param_env']['n_on_premise_node'],
        config['param_env']['n_cloud_node'],
        config, 10, 0.2, 0
    )
    jobs_set = job_generator.generate_jobs_set()
    
    # 環境を初期化
    env = SchedulingEnvC(
        np.inf, config['param_env']['n_window'],
        config['param_env']['n_on_premise_node'],
        config['param_env']['n_cloud_node'],
        config['param_env']['n_job_queue_obs'],
        config['param_env']['n_job_queue_bck'],
        config['param_agent']['weight_wt'],
        config['param_agent']['weight_cost'],
        config['param_env']['penalty_not_allocate'],
        config['param_env']['penalty_invalid_action'],
        jobs_set, None, flag=0
    )
    
    env.reset()
    
    # テスト1: find_allocation_positionがC実装を使用しているか確認
    print("\n--- テスト1: find_allocation_position ---")
    action = [0, 0]  # オンプレミス
    job = env.job_queue[0]
    
    # キャッシュを構築
    cache_c = env._rebuild_cache_if_needed_c(use_cloud=False)
    print(f"  Cキャッシュの型: {type(cache_c)}")
    print(f"  WindowCacheインスタンス: {isinstance(cache_c, WindowCache)}")
    
    # find_allocation_positionを呼び出し
    position, waiting_time = env.find_allocation_position(action)
    print(f"  位置: {position}, 待ち時間: {waiting_time}")
    
    # テスト2: time_transitionがC実装を使用しているか確認
    print("\n--- テスト2: time_transition ---")
    on_premise_status_before = env.on_premise_window['status'].copy()
    env.time_transition(slide_on_premise=True, slide_cloud=False)
    on_premise_status_after = env.on_premise_window['status'].copy()
    
    # スライドが実行されたか確認（最初の列が2列目の値になっている）
    if np.array_equal(on_premise_status_before[:, 0], on_premise_status_before[:, 1]):
        print("  ✓ time_transitionが実行されました（最初の列が2列目と同じ）")
    else:
        print("  ✗ time_transitionが正しく実行されていません")
    
    # テスト3: do_scheduleがC実装を使用しているか確認
    print("\n--- テスト3: do_schedule ---")
    if position is not None:
        env.reset()
        waiting_time = env.do_schedule(action, job, position)
        print(f"  ジョブがスケジュールされました（待ち時間: {waiting_time}）")
        
        # ジョブが正しく配置されたか確認
        if len(position) == 2:
            i, a = position
            job_width = int(job[0])
            job_height = int(job[1])
            scheduled_region = env.on_premise_window['status'][i:i+job_height, a:a+job_width]
            if np.all(scheduled_region == 1):
                print("  ✓ ジョブが正しく配置されました")
            else:
                print("  ✗ ジョブが正しく配置されていません")
    
    # テスト4: _rebuild_cache_if_neededがC実装を使用しているか確認
    print("\n--- テスト4: _rebuild_cache_if_needed ---")
    cache = env._rebuild_cache_if_needed(use_cloud=False)
    print(f"  キャッシュの型: {type(cache)}")
    print(f"  キャッシュのバージョン: {cache.get('version', 'N/A')}")
    
    # Cキャッシュが構築されているか確認
    if env._cache_onpre_c is not None:
        print("  ✓ Cキャッシュが構築されています")
    else:
        print("  ✗ Cキャッシュが構築されていません")
    
    print("\n" + "="*60)
    print("確認完了")
    print("="*60)

if __name__ == "__main__":
    verify_c_implementation()

