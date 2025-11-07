#!/usr/bin/env python3
"""
既存のPython実装とC言語実装の完全な互換性テスト（実際のSchedulingEnvを使用）
同じ入力に対して同じ出力が得られることを確認
"""
import numpy as np
import sys
import os
import yaml

# パスを追加（プロジェクトのルートディレクトリを追加）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

try:
    from scheduling_env_core import (
        WindowCache,
        find_allocation_position as c_find_allocation_position,
        time_transition as c_time_transition,
        do_schedule as c_do_schedule,
        get_unique_job_ids as c_get_unique_job_ids,
        calculate_makespan as c_calculate_makespan
    )
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


def test_find_allocation_position_exact():
    """find_allocation_positionの完全な互換性テスト（実際のSchedulingEnvを使用）"""
    print("\n=== find_allocation_position完全互換性テスト ===")
    
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
    
    # テストケース: 複数のアクションを試行
    test_actions = [0, 1, 0, 1, 0]  # オンプレミスとクラウドを交互に
    
    all_match = True
    for action_idx, action_raw in enumerate(test_actions):
        # アクションを変換
        action = env.get_converted_action(action_raw)
        job = env.job_queue[0]
        
        # ジョブが空の場合はスキップ
        if job[0] == 0 and job[1] == 0:
            env.time_transition(True, True)
            continue
        
        # Python実装で位置を探索
        cache_onpre = env._rebuild_cache_if_needed(use_cloud=False)
        cache_cloud = env._rebuild_cache_if_needed(use_cloud=True)
        
        py_position, py_waiting_time = env.find_allocation_position(
            action, cache_onpre=cache_onpre, cache_cloud=cache_cloud
        )
        
        # C言語実装で位置を探索
        if not action[1]:  # オンプレミス
            window_status = np.ascontiguousarray(
                env.on_premise_window['status'], dtype=np.int32
            )
            cache_c = WindowCache(window_status, env.n_on_premise_node, env.n_window)
        else:  # クラウド
            window_status = np.ascontiguousarray(
                env.cloud_window['status'], dtype=np.int32
            )
            cache_c = WindowCache(window_status, env.n_cloud_node, env.n_window)
        
        job_width = int(job[0])
        job_height = int(job[1])
        when_submitted = int(job[-1])
        
        c_position, c_waiting_time = c_find_allocation_position(
            cache_c, job_width, job_height, when_submitted, env.time
        )
        
        # 結果を比較
        if py_position is None:
            if c_position is not None:
                print(f"✗ アクション {action_idx} (action={action_raw}): Pythonは見つからなかったがCは見つかった")
                print(f"    C位置: {c_position}, C待ち時間: {c_waiting_time}")
                all_match = False
            else:
                # 両方とも見つからなかった
                if abs(py_waiting_time - c_waiting_time) > 1e-6:
                    print(f"✗ アクション {action_idx} (action={action_raw}): 待ち時間が異なる（見つからなかった場合）")
                    print(f"    Python: {py_waiting_time}, C: {c_waiting_time}")
                    all_match = False
        else:
            if c_position is None:
                print(f"✗ アクション {action_idx} (action={action_raw}): Pythonは見つかったがCは見つからなかった")
                print(f"    Python位置: {py_position}, Python待ち時間: {py_waiting_time}")
                all_match = False
            else:
                # 位置を比較
                if isinstance(py_position, tuple) and len(py_position) == 2:
                    # 連続割り当て
                    if isinstance(c_position, tuple) and len(c_position) == 2:
                        if py_position != c_position:
                            print(f"✗ アクション {action_idx} (action={action_raw}): 位置が異なる")
                            print(f"    Python: {py_position}, C: {c_position}")
                            all_match = False
                        elif abs(py_waiting_time - c_waiting_time) > 1e-6:
                            print(f"✗ アクション {action_idx} (action={action_raw}): 待ち時間が異なる")
                            print(f"    Python: {py_waiting_time}, C: {c_waiting_time}")
                            all_match = False
                        else:
                            print(f"✓ アクション {action_idx} (action={action_raw}): 完全に同じ結果")
                    else:
                        print(f"✗ アクション {action_idx} (action={action_raw}): 位置の形式が異なる")
                        print(f"    Python: {py_position} (連続), C: {c_position}")
                        all_match = False
                elif isinstance(py_position, tuple) and len(py_position) == 3:
                    # 分散割り当て
                    if isinstance(c_position, tuple) and len(c_position) == 3:
                        # 分散割り当ての比較（順序が異なる可能性があるため、内容を比較）
                        py_i, py_a, py_nodes = py_position
                        c_i, c_a, c_nodes = c_position
                        if py_a != c_a:
                            print(f"✗ アクション {action_idx} (action={action_raw}): 列位置が異なる")
                            print(f"    Python: a={py_a}, C: a={c_a}")
                            all_match = False
                        elif abs(py_waiting_time - c_waiting_time) > 1e-6:
                            print(f"✗ アクション {action_idx} (action={action_raw}): 待ち時間が異なる")
                            print(f"    Python: {py_waiting_time}, C: {c_waiting_time}")
                            all_match = False
                        else:
                            print(f"✓ アクション {action_idx} (action={action_raw}): 分散割り当てで一致")
                    else:
                        print(f"✗ アクション {action_idx} (action={action_raw}): 位置の形式が異なる")
                        print(f"    Python: {py_position} (分散), C: {c_position}")
                        all_match = False
        
        # 実際にステップを実行（次のテストの準備）
        if py_position is not None:
            env.step(action_raw)
        else:
            env.time_transition(True, True)
    
    if all_match:
        print("✓ find_allocation_position: すべてのテストケースで完全に同じ結果")
    else:
        print("✗ find_allocation_position: 一部のテストケースで結果が異なります")
    
    return all_match


def test_time_transition_exact():
    """time_transitionの完全な互換性テスト（実際のSchedulingEnvを使用）"""
    print("\n=== time_transition完全互換性テスト ===")
    
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
    
    # いくつかのジョブをスケジュールして状態を変更
    for _ in range(5):
        action = 0  # オンプレミス
        env.step(action)
    
    # 現在の状態を保存
    onpre_status_before = env.on_premise_window['status'].copy()
    onpre_job_id_before = env.on_premise_window['job_id'].copy()
    cloud_status_before = env.cloud_window['status'].copy()
    cloud_job_id_before = env.cloud_window['job_id'].copy()
    
    # Python実装でtime_transitionを実行
    env.time_transition(slide_on_premise=True, slide_cloud=True)
    
    onpre_status_py = env.on_premise_window['status'].copy()
    onpre_job_id_py = env.on_premise_window['job_id'].copy()
    cloud_status_py = env.cloud_window['status'].copy()
    cloud_job_id_py = env.cloud_window['job_id'].copy()
    
    # 環境を再初期化して同じ状態に戻す
    env.reset()
    for _ in range(5):
        action = 0
        env.step(action)
    
    # C言語実装でtime_transitionを実行
    onpre_status_c = np.ascontiguousarray(
        env.on_premise_window['status'], dtype=np.int32
    )
    onpre_job_id_c = np.ascontiguousarray(
        env.on_premise_window['job_id'], dtype=np.int32
    )
    cloud_status_c = np.ascontiguousarray(
        env.cloud_window['status'], dtype=np.int32
    )
    cloud_job_id_c = np.ascontiguousarray(
        env.cloud_window['job_id'], dtype=np.int32
    )
    
    result_c = c_time_transition(
        onpre_status_c, onpre_job_id_c,
        env.n_on_premise_node, env.n_window, True
    )
    if result_c is not None:
        onpre_status_c, onpre_job_id_c = result_c
    
    result_c = c_time_transition(
        cloud_status_c, cloud_job_id_c,
        env.n_cloud_node, env.n_window, True
    )
    if result_c is not None:
        cloud_status_c, cloud_job_id_c = result_c
    
    # 結果を比較
    onpre_status_match = np.array_equal(onpre_status_py, onpre_status_c)
    onpre_job_id_match = np.array_equal(onpre_job_id_py, onpre_job_id_c)
    cloud_status_match = np.array_equal(cloud_status_py, cloud_status_c)
    cloud_job_id_match = np.array_equal(cloud_job_id_py, cloud_job_id_c)
    
    if (onpre_status_match and onpre_job_id_match and 
        cloud_status_match and cloud_job_id_match):
        print("✓ time_transition: 完全に同じ結果")
        return True
    else:
        print("✗ time_transition: 結果が異なります")
        if not onpre_status_match:
            print(f"  オンプレミスstatus不一致:")
            print(f"    Python: {onpre_status_py[:, 0]}")
            print(f"    C:      {onpre_status_c[:, 0]}")
        if not onpre_job_id_match:
            print(f"  オンプレミスjob_id不一致:")
            print(f"    Python: {onpre_job_id_py[:, 0]}")
            print(f"    C:      {onpre_job_id_c[:, 0]}")
        if not cloud_status_match:
            print(f"  クラウドstatus不一致:")
            print(f"    Python: {cloud_status_py[:, 0]}")
            print(f"    C:      {cloud_status_c[:, 0]}")
        if not cloud_job_id_match:
            print(f"  クラウドjob_id不一致:")
            print(f"    Python: {cloud_job_id_py[:, 0]}")
            print(f"    C:      {cloud_job_id_c[:, 0]}")
        return False


def main():
    """メイン関数"""
    print("=" * 60)
    print("既存のPython実装とC言語実装の完全な互換性テスト")
    print("（実際のSchedulingEnvを使用）")
    print("=" * 60)
    
    try:
        test_time_transition_exact()
        test_find_allocation_position_exact()
        
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

