"""ワークロード推定に基づく distributed PCN 学習プロファイル（合成・トレース共通）。"""
from __future__ import annotations

import os
from typing import Any, Dict

from src.utils.workload_calibration import (
    apply_calibration_env,
    calibrate_from_config,
    format_calibration_log,
)


def apply_workload_adaptive_training_env(config: dict) -> Dict[str, Any]:
    """ジョブセットからスケールを推定し、中域 PF + Eval ギャップ FB を有効化。"""
    os.environ["PCN_LEFT_TAIL_PROFILE"] = "0"
    # [ablation] 強制ONを setdefault 化（env で上書き可能に。未指定なら従来通りON＝ビット一致）。
    os.environ.setdefault("PCN_EVAL_GAP_FEEDBACK", "1")
    os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
    os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_NATIVE", "1")
    os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
    os.environ.setdefault("DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP", "1")
    os.environ.setdefault("DISTRIBUTED_PCN_LIVE_UNIFORM_PF", "1")
    os.environ.setdefault("DISTRIBUTED_PCN_LIVE_UNIFORM_PF_GRID", "20")
    os.environ.setdefault("DISTRIBUTED_PCN_LIVE_UNIFORM_PF_LOW_TAIL_FRAC", "0.20")
    os.environ.setdefault("DISTRIBUTED_PCN_LIVE_UNIFORM_PF_LOW_TAIL_EXTRA", "20")
    os.environ.setdefault("DISTRIBUTED_PCN_LIVE_UNIFORM_PF_LABEL", "workload_adapt")
    os.environ.setdefault("DISTRIBUTED_PCN_EVAL_INTERVAL", "5")
    os.environ.setdefault("DISTRIBUTED_PCN_EVAL_SAMPLES", "64")
    # Phase2 教師あり学習は「1遷移ごとの (obs,desired_return)→action」を直接模倣する＝
    # 地平長に不変＝スケール非依存に conditioning の土台を作る。これを強制 0 にすると
    # conditioning 確立が Phase3（長地平のクレジット割当）任せになり、大規模(512/1024)で
    # 指令無視・全クラウド崩壊する。recipe/ユーザ指定を尊重するため setdefault に変更。
    os.environ.setdefault("DISTRIBUTED_PCN_SUPERVISED_EPOCHS", "0")
    os.environ.setdefault("DISTRIBUTED_PCN_ENABLE_VISUALIZATION", "0")
    os.environ.setdefault("PCN_VALUE_REPRO_WEIGHT", "0")
    os.environ.setdefault("PCN_ADAPTIVE_RETURN_NORMALIZATION", "1")
    os.environ.setdefault("PCN_COMMAND_BALANCE", "1")
    os.environ.setdefault("PCN_CONDITIONING_SENS_WEIGHT", "0.04")
    os.environ.setdefault("PCN_CONDITIONING_KL_MARGIN", "0.08")
    os.environ.setdefault("PCN_CONDITIONING_SENS_WAIT_DR_THRESH", "5e-4")
    os.environ.setdefault("PCN_COND_ADD_SCALE", "0.25")
    os.environ.setdefault("PCN_S_EMB_DROPOUT", "0.08")
    os.environ.setdefault("PCN_TRAIN_COST_ENDPOINT_WEIGHT", "8")
    os.environ.setdefault("PCN_TRAIN_MID_PF_WEIGHT", "16")
    os.environ.setdefault("PCN_TRAIN_MID_STEP_WEIGHT", "8")
    os.environ.setdefault("PCN_TRAIN_EVALIKE_STEP_WEIGHT", "3")
    os.environ.setdefault("PCN_TRAIN_EVALIKE_STEP_FRAC", "0.12")
    os.environ.setdefault("PCN_TRAIN_KNEE_PF_WEIGHT", "8")
    os.environ.setdefault("PCN_TRAIN_LOW_SLOPE_PF_WEIGHT", "6")
    os.environ.setdefault("PCN_TRAIN_LOW_WAIT_PF_WEIGHT", "8")
    os.environ.setdefault("PCN_TRAIN_LOW_WAIT_FRAC", "0.30")
    os.environ.setdefault("PCN_TRAIN_KNEE_STEP_WEIGHT", "5")
    os.environ.setdefault("PCN_TRAIN_LOW_SLOPE_STEP_WEIGHT", "0")

    cal = calibrate_from_config(config)
    env_updates = apply_calibration_env(cal)
    env_updates["PCN_EVAL_GAP_BOOST_MAX"] = "5.0"
    env_updates["PCN_EVAL_GAP_REF_FRAC"] = "0.06"
    os.environ["PCN_EVAL_GAP_BOOST_MAX"] = "5.0"
    os.environ["PCN_EVAL_GAP_REF_FRAC"] = "0.06"
    os.environ.setdefault("PCN_CHOOSE_COMMANDS_MODE", "pf_mixed")
    os.environ.setdefault("PCN_PF_COMMAND_LOW_WAIT_FRAC", "0.30")
    os.environ.setdefault("PCN_PF_COMMAND_LOW_WAIT_QUOTA", "8")
    os.environ.setdefault("PCN_PF_COMMAND_INCLUDE_EXTREMES", "1")
    os.environ.setdefault("PCN_EVAL_COMMAND_MODE", "pf_ref")
    # [2026-08-06 バグ修正] PCN_REF_PF_NPZ の既定セット(trace24 前線)を廃止=オプトインのみ。
    # 旧挙動: trace24(24ジョブ)の桁違いに小さい前線が全ワークロードの command pool へ
    # 既定で混入し、非支配比較で自分の達成点を押し出して注文の大半が「よそのワーク
    # ロードの点」になっていた(18ジョブ統制実験で実測)。参照前線を使いたい場合のみ
    # PCN_REF_PF_NPZ を明示指定する(使用時は pcn_agent が [REF_PF] ログ+スケール
    # 不整合ガードを適用する)。
    print(format_calibration_log(cal, env_updates))
    return {"calibration": cal, "env": env_updates}
