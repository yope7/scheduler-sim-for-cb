"""分散PCN CLI（環境変数へ反映）。distributed_pcn 本体の import 前に呼ぶ。"""
import argparse
import os
import sys
from typing import List, Optional, Sequence


def apply_distributed_pcn_cli_env(
    *,
    event_obs: Optional[bool] = None,
    event_native: Optional[bool] = None,
    build_schedule_maps: Optional[bool] = None,
    learner_bitmap: Optional[bool] = None,
    enable_viz: Optional[bool] = None,
    conditioning: Optional[bool] = None,
    mid_core: Optional[bool] = None,
) -> None:
    os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
    os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_NATIVE", "1")
    os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
    os.environ.setdefault("DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP", "1")
    os.environ.setdefault("DISTRIBUTED_PCN_PHASE3_GPU_CACHE", "1")
    os.environ.setdefault("DISTRIBUTED_PCN_REPLAY_ZERO_COPY", "1")
    os.environ.setdefault("DISTRIBUTED_PCN_ACTOR_RAY_PUT", "1")
    os.environ.setdefault("DISTRIBUTED_PCN_DISTRIBUTED_EVAL", "1")
    os.environ.setdefault("DISTRIBUTED_PCN_UNIFORM_PF_DISTRIBUTED", "1")
    os.environ.setdefault("DISTRIBUTED_PCN_EVAL_QUIET", "1")
    os.environ.setdefault("DISTRIBUTED_PCN_LOG_RAY_TRANSFER", "0")
    # 学習結果に影響しない診断のみ（left_tail 本番の壁時計短縮）
    os.environ.setdefault("DISTRIBUTED_PCN_PHASE2_IMPORTANCE", "0")

    if event_obs is not None:
        os.environ["DISTRIBUTED_PCN_USE_EVENT_OBS"] = "1" if event_obs else "0"
        if not event_obs:
            os.environ["DISTRIBUTED_PCN_USE_EVENT_NATIVE"] = "0"
    if event_native is not None:
        os.environ["DISTRIBUTED_PCN_USE_EVENT_NATIVE"] = "1" if event_native else "0"
    if build_schedule_maps is not None:
        os.environ["SCHEDULER_BUILD_SCHEDULE_MAPS"] = "1" if build_schedule_maps else "0"
    if learner_bitmap is not None:
        os.environ["SCHEDULER_LEARNER_BITMAP"] = "1" if learner_bitmap else "0"
    if enable_viz is not None:
        os.environ["DISTRIBUTED_PCN_ENABLE_VISUALIZATION"] = "1" if enable_viz else "0"
    if conditioning:
        os.environ.setdefault("PCN_CONDITIONING_SENS_WEIGHT", "0.03")
        os.environ.setdefault("PCN_CONDITIONING_KL_MARGIN", "0.08")
        os.environ.setdefault("PCN_COND_ADD_SCALE", "0.25")
        os.environ.setdefault("PCN_S_EMB_DROPOUT", "0.08")
        os.environ.setdefault("PCN_TRAIN_COST_ENDPOINT_WEIGHT", "8")
        if not mid_core:
            os.environ.setdefault("PCN_TRAIN_MID_PF_WEIGHT", "0")
        os.environ.setdefault("PCN_CONDITIONING_SENS_WAIT_DR_THRESH", "0")
        os.environ.setdefault("DISTRIBUTED_PCN_EVAL_DIAG", "1")
        # value_repro(補助 value head 損失)は Phase3 後半で共有埋め込みを奪い、
        # 方策の command 応答を潰して eval(argmax)を片隅へ崩壊させることが分かったため既定 OFF。
        # （根本修正=正規化 + hinge-KL + cond_add + dropout で安定して全域 PF を獲得できる）
        os.environ.setdefault("PCN_VALUE_REPRO_WEIGHT", "0")
        os.environ.setdefault("PCN_VALUE_COST_SCALE", "100000.0")
        os.environ.setdefault("PCN_VALUE_WAIT_SCALE", "500.0")
    if mid_core:
        # 中域 PF: ステップ replay + Archive 中域 wait 条件付け + command バランス
        os.environ.setdefault("PCN_COMMAND_BALANCE", "1")
        os.environ.setdefault("PCN_TRAIN_MID_STEP_WEIGHT", "6")
        os.environ.setdefault("PCN_TRAIN_EVALIKE_STEP_WEIGHT", "4")
        os.environ.setdefault("PCN_TRAIN_EVALIKE_STEP_FRAC", "0.15")
        os.environ.setdefault("PCN_TRAIN_MID_PF_WEIGHT", "4")
        os.environ.setdefault("PCN_MID_BAND_COND_WEIGHT", "0.06")
        os.environ.setdefault("PCN_MID_BAND_COND_WAIT_LEVELS", "5")
        os.environ.setdefault("PCN_MID_BAND_COND_COST_LEVELS", "4")
        os.environ.setdefault("PCN_CONDITIONING_SENS_WAIT_DR_THRESH", "0.002")


def apply_left_tail_training_env() -> None:
    """左上先端 PF 向け本番プロファイル（dual12 + EvalギャップFB、0 から学習でも同設定）。

    ``--left-tail`` または config ``distributed_pcn.left_tail: true`` で有効化。
    ``--conditioning`` / ``--mid-core`` より後に呼ぶと dual12 用の重みで上書きする。
    """
    os.environ["PCN_LEFT_TAIL_PROFILE"] = "1"

    # 分散 PCN 本番（scratch100 / dual12 と同系）
    os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
    os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_NATIVE", "1")
    os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
    os.environ.setdefault("DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP", "1")
    os.environ.setdefault("DISTRIBUTED_PCN_EVAL_SAMPLES", "50")
    os.environ.setdefault("DISTRIBUTED_PCN_EVAL_INTERVAL", "10")
    os.environ.setdefault("DISTRIBUTED_PCN_SKIP_FINAL_EVAL", "1")
    os.environ.setdefault("DISTRIBUTED_PCN_SUPERVISED_EPOCHS", "0")
    os.environ.setdefault("DISTRIBUTED_PCN_ENABLE_VISUALIZATION", "0")
    os.environ.setdefault("DISTRIBUTED_PCN_LEARNING_RATE", "0.001")
    # 学習中の均等 command PF 図（Eval 間隔ごと → execution_dir/uniform_cmd_pf_iter_XXX.png）
    os.environ.setdefault("DISTRIBUTED_PCN_LIVE_UNIFORM_PF", "1")
    os.environ.setdefault("DISTRIBUTED_PCN_LIVE_UNIFORM_PF_GRID", "12")
    os.environ.setdefault("DISTRIBUTED_PCN_LIVE_UNIFORM_PF_LOW_TAIL_FRAC", "0.18")
    os.environ.setdefault("DISTRIBUTED_PCN_LIVE_UNIFORM_PF_LOW_TAIL_EXTRA", "16")
    os.environ.setdefault("DISTRIBUTED_PCN_LIVE_UNIFORM_PF_LABEL", "left_tail")
    os.environ.setdefault("DISTRIBUTED_PCN_CMD_OUTCOMES", "1")

    # conditioning（value_repro OFF）
    os.environ.setdefault("PCN_VALUE_REPRO_WEIGHT", "0")
    os.environ.setdefault("PCN_COMMAND_BALANCE", "1")
    os.environ.setdefault("PCN_ADAPTIVE_RETURN_NORMALIZATION", "1")
    os.environ.setdefault("PCN_CONDITIONING_SENS_WEIGHT", "0.025")
    os.environ.setdefault("PCN_CONDITIONING_SENS_WAIT_DR_THRESH", "0.001")
    os.environ.setdefault("PCN_COND_ADD_SCALE", "0.22")
    os.environ.setdefault("PCN_S_EMB_DROPOUT", "0.06")
    os.environ.setdefault("PCN_VALUE_COST_SCALE", "100000.0")
    os.environ.setdefault("PCN_VALUE_WAIT_SCALE", "500.0")

    # 膝・中域（dual12 / quest ベスト。--mid-core 既定の 6/4 より控えめ）
    os.environ.setdefault("PCN_TRAIN_MID_STEP_WEIGHT", "3")
    os.environ.setdefault("PCN_TRAIN_MID_PF_WEIGHT", "2")
    os.environ.setdefault("PCN_MID_BAND_COND_WEIGHT", "0.02")
    os.environ.setdefault("PCN_TRAIN_KNEE_PF_WEIGHT", "3")
    os.environ.setdefault("PCN_TRAIN_KNEE_STEP_WEIGHT", "3")

    # 低コスト帯 conditioning（dual）
    os.environ.setdefault("PCN_TRAIN_LOW_SLOPE_PF_WEIGHT", "0")
    os.environ.setdefault("PCN_TRAIN_LOW_SLOPE_STEP_WEIGHT", "0")
    os.environ.setdefault("PCN_TRAIN_LOW_SLOPE_COST_MAX_FRAC", "0.18")
    os.environ.setdefault("PCN_TRAIN_LOW_WAIT_PF_WEIGHT", "6")
    os.environ.setdefault("PCN_TRAIN_LOW_WAIT_FRAC", "0.30")
    os.environ.setdefault("PCN_LOW_BAND_COND_MODE", "dual")
    os.environ.setdefault("PCN_LOW_BAND_COND_WEIGHT", "0.07")
    os.environ.setdefault("PCN_LOW_BAND_COND_COST_LEVELS", "12")
    os.environ.setdefault("PCN_LOW_BAND_COND_WAIT_LEVELS", "10")
    os.environ.setdefault("PCN_LOW_BAND_COND_KL_MARGIN", "0.07")
    os.environ.setdefault("PCN_LOW_BAND_DUAL_R1_FRAC", "0.65")
    os.environ.setdefault("PCN_PF_COMMAND_LOW_WAIT_FRAC", "0.30")
    os.environ.setdefault("PCN_PF_COMMAND_LOW_WAIT_QUOTA", "8")
    os.environ.setdefault("PCN_PF_COMMAND_INCLUDE_EXTREMES", "1")

    # Eval 弱点帯域 → replay（低域のみ。全域ブーストは膝悪化しやすい）
    os.environ.setdefault("PCN_EVAL_GAP_FEEDBACK", "1")
    os.environ.setdefault("PCN_EVAL_GAP_BAND_NAMES", "low,low_slope")
    os.environ.setdefault("PCN_EVAL_GAP_FEEDBACK_GRID", "12")
    os.environ.setdefault("PCN_EVAL_GAP_REF_GAP", "1200")
    os.environ.setdefault("PCN_EVAL_GAP_BOOST_MAX", "2.0")
    # Phase2 後の重要度図は教師データに影響しない（壁時計のみ）
    os.environ.setdefault("DISTRIBUTED_PCN_PHASE2_IMPORTANCE", "0")


def build_distributed_pcn_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--event-obs", action="store_true")
    p.add_argument("--bitmap-obs", action="store_true")
    p.add_argument(
        "--event-native",
        action="store_true",
        help="イベントネイティブ step（既定 ON。--bitmap-step で旧 time_transition ループ）",
    )
    p.add_argument(
        "--bitmap-step",
        action="store_true",
        help="観測はイベントのまま step のみ SchedulingEnvEventObs（time_transition）",
    )
    p.add_argument(
        "--build-schedule-maps",
        action="store_true",
        help="エピソード終了時にノード×時刻のスケジュールマップ行列を構築（重い・可視化用）",
    )
    p.add_argument(
        "--no-build-schedule-maps",
        action="store_true",
        help="スケジュールマップ構築を明示 OFF",
    )
    p.add_argument("--no-learner-bitmap", action="store_true")
    p.add_argument("--learner-bitmap", action="store_true")
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--enable-visualization", action="store_true")
    p.add_argument("--no-enable-visualization", action="store_true")
    p.add_argument("--conditioning", action="store_true")
    p.add_argument("--mid-core", action="store_true",
                   help="中域 PF 向け replay/conditioning 設計（ステップ重み+Archive wait KL）")
    p.add_argument(
        "--left-tail",
        action="store_true",
        help="左上先端 PF 本番プロファイル（dual12 + 低域のみ EvalギャップFB）。0 から学習でも同設定",
    )
    p.add_argument("--no-left-tail", action="store_true", help="--left-tail を無効化")
    p.add_argument(
        "--live-pf",
        action="store_true",
        help="Eval 間隔ごとに均等 command PF 図を execution_dir へ保存",
    )
    p.add_argument("--no-live-pf", action="store_true", help="LIVE_PF を無効化")
    p.add_argument("--profile", action="store_true")
    return p


def apply_distributed_pcn_cli(argv: Optional[Sequence[str]] = None) -> List[str]:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_distributed_pcn_arg_parser()
    args, remaining = parser.parse_known_args(list(argv))

    event_obs: Optional[bool] = None
    if args.bitmap_obs:
        event_obs = False
    elif args.event_obs:
        event_obs = True

    event_native: Optional[bool] = None
    if args.bitmap_step:
        event_native = False
    elif args.event_native:
        event_native = True

    build_schedule_maps: Optional[bool] = None
    if args.build_schedule_maps:
        build_schedule_maps = True
    elif args.no_build_schedule_maps:
        build_schedule_maps = False

    learner_bitmap: Optional[bool] = None
    if args.learner_bitmap:
        learner_bitmap = True
    elif args.no_learner_bitmap:
        learner_bitmap = False

    enable_viz: Optional[bool] = None
    if args.no_viz or args.no_enable_visualization:
        enable_viz = False
    elif args.enable_visualization:
        enable_viz = True

    apply_distributed_pcn_cli_env(
        event_obs=event_obs,
        event_native=event_native,
        build_schedule_maps=build_schedule_maps,
        learner_bitmap=learner_bitmap,
        enable_viz=enable_viz,
        conditioning=args.conditioning,
        mid_core=args.mid_core,
    )
    if args.left_tail and not args.no_left_tail:
        apply_left_tail_training_env()
        print(
            "[LEFT_TAIL] dual12 + 低域 EvalギャップFB + LIVE_PF（eval_interval=10, "
            "bands=low,low_slope）"
        )
    elif args.no_left_tail:
        os.environ["PCN_LEFT_TAIL_PROFILE"] = "0"
        os.environ["PCN_EVAL_GAP_FEEDBACK"] = "0"
    if args.live_pf:
        os.environ["DISTRIBUTED_PCN_LIVE_UNIFORM_PF"] = "1"
    elif args.no_live_pf:
        os.environ["DISTRIBUTED_PCN_LIVE_UNIFORM_PF"] = "0"
    if args.profile:
        os.environ["DISTRIBUTED_PCN_PROFILE"] = "1"
    return remaining


def main_cli_entry() -> None:
    apply_distributed_pcn_cli(sys.argv[1:])
