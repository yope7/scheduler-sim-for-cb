#!/usr/bin/env python3
"""
do_scheduleの完全な互換性テスト
"""
import numpy as np
import sys
import os
import yaml

# パスを追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

try:
    from scheduling_env_core import do_schedule as c_do_schedule
    C_AVAILABLE = True
except ImportError:
    C_AVAILABLE = False
    print("警告: C言語実装が利用できません。")
    sys.exit(1)

from src.envs.scheduling_env import SchedulingEnv
from src.utils.job_gen.job_generator import JobGenerator


def load_config():
    """設定ファイルを読み込み"""
    config_path = os.path.join(project_root, 'config', 'config.yml')
    with open(config_path, 'r') as yml:
        config = yaml.safe_load(yml)
    return config


def test_do_schedule_exact():
    """do_scheduleの完全な互換性テスト"""
    print("\n=== do_schedule完全互換性テスト ===")
    
    # 設定を読み込み
    config = load_config()
    
    # ジョブセットを作成
    job_generator = JobGenerator(
        42, 1, config['param_env']['n_window'],
        config['param_env']['n_on_premise_node'],
        config['param_env']['n_cloud_node'],
        config, 100, 0.2, 0
    )
    jobs_set = job_generator.generate_jobs_set()
    
    # 環境を初期化
    env = SchedulingEnv(
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
    
    # テストケース: 連続割り当て
    action = [0, 0]  # オンプレミス
    job = env.job_queue[0]
    position = (2, 5)  # 連続割り当て
    
    # 現在の状態を保存
    onpre_status_before = env.on_premise_window['status'].copy()
    onpre_job_id_before = env.on_premise_window['job_id'].copy()
    
    # Python実装でdo_scheduleを実行
    py_waiting_time = env.do_schedule(action, job, position)
    
    onpre_status_py = env.on_premise_window['status'].copy()
    onpre_job_id_py = env.on_premise_window['job_id'].copy()
    
    # 環境を再初期化して同じ状態に戻す
    env.reset()
    
    # C言語実装でdo_scheduleを実行
    onpre_status_c = np.ascontiguousarray(
        env.on_premise_window['status'], dtype=np.int32
    )
    onpre_job_id_c = np.ascontiguousarray(
        env.on_premise_window['job_id'], dtype=np.int32
    )
    
    job_width = int(job[0])
    job_height = int(job[1])
    job_id = int(job[4])
    
    c_do_schedule(
        onpre_status_c, onpre_job_id_c,
        env.n_on_premise_node, env.n_window,
        job_width, job_height, job_id,
        position
    )
    
    # 結果を比較
    status_match = np.array_equal(onpre_status_py, onpre_status_c)
    job_id_match = np.array_equal(onpre_job_id_py, onpre_job_id_c)
    
    if status_match and job_id_match:
        print("✓ do_schedule (連続割り当て): 完全に同じ結果")
        print(f"  待ち時間: {py_waiting_time}")
    else:
        print("✗ do_schedule (連続割り当て): 結果が異なります")
        if not status_match:
            print(f"  status不一致:")
            print(f"    Python: {onpre_status_py[2:2+job_height, 5:5+job_width]}")
            print(f"    C:      {onpre_status_c[2:2+job_height, 5:5+job_width]}")
        if not job_id_match:
            print(f"  job_id不一致:")
            print(f"    Python: {onpre_job_id_py[2:2+job_height, 5:5+job_width]}")
            print(f"    C:      {onpre_job_id_c[2:2+job_height, 5:5+job_width]}")
        return False
    
    return True


def main():
    """メイン関数"""
    print("=" * 60)
    print("do_scheduleの完全な互換性テスト")
    print("=" * 60)
    
    try:
        test_do_schedule_exact()
        
        print("\n" + "=" * 60)
        print("✓ すべての互換性テストが完了しました")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

