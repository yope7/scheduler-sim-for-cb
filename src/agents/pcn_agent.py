"""Pareto Conditioned Network. Code adapted from https://github.com/mathieu-reymond/pareto-conditioned-networks ."""
import heapq
import os
import re
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union
from matplotlib.animation import FuncAnimation
import signal
import time
import traceback
import warnings

# CUDAが利用できない場合の警告を抑制
warnings.filterwarnings('ignore', message="Can't initialize NVML")
warnings.filterwarnings('ignore', message="torch.cuda.amp.GradScaler is enabled, but CUDA is not available")

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np

from src.utils.schedule_map_options import schedule_maps_enabled
from src.utils.anchor_residual import get_anchor_set
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import wandb
# wandb.init(project="temp")

# desired_return クリップ（既定は十分大きく 1e6、0以下で無効化）
# NOTE: 1024ジョブ等では command (cost/-wait*nj) が ~1e9 桁になる。クリップ幅が
# command スケール未満だと全 command が同値に潰れ conditioning が消滅し loss=log(2)
# で停滞する。正規化(scale≈到達レンジ)が桁を吸収するので、クリップは Inf 保護用に広く取る。
_DESIRED_RETURN_CLIP = float(os.environ.get("PCN_DESIRED_RETURN_CLIP", "1e12"))
# desired_return 入力スケーリング（数値安定化）
_DESIRED_RETURN_SCALE = float(os.environ.get("PCN_DESIRED_RETURN_SCALE", "10000"))

# --- PCN_LABEL_G: 教師ラベル二重累積バグの修正フラグ（既定 OFF=現行ビット不変） ---
# _add_episode() は格納時に reward を in-place で累積化する（episode[t].reward = G_t =
# Σ_{j>=t} γ^{j-t} r_j）。ところが教師ラベル作成の2経路（get_training_batch 非キャッシュ /
# _encode_episode_training_block キャッシュ）は格納値を「生報酬」と誤認して再度割引累積し、
# ラベル = D_t = Σ_{j>=t} γ^{j-t} G_j という別物になっていた（元論文実装は格納済み G_t を
# そのまま使う）。実行時（evaluate/rollout の desired_return 更新）は G 規約なので、
# 学習と実行で指令の意味が ~9-100 倍ズレる。PCN_LABEL_G=1 で再累積をやめ、格納値 G_t を
# そのままラベルに使う。
# NOTE: このフラグはモジュール import 時に固定されるプロセス定数。_pcn_training_block の
# エピソード単位メモ化はフラグ値をブロックに記録し、値が一致するときだけヒットさせる
# （同一プロセス内でフラグが変わることはないが、混入防止の防壁として）。
_LABEL_G = os.environ.get("PCN_LABEL_G", "0") == "1"
if _LABEL_G:
    print("[PCN_LABEL_G] 有効: 教師ラベル=格納済みG_t（二重累積の再計算を廃止）", flush=True)

# --- C2: 条件付け wait 次元の外れ値ロバスト圧縮（崩壊=loss爆発 対策） ---
# PCN の損失は行動交差エントロピーで、巨大 wait は「条件付け入力 desired_return」経由でのみ
# 学習に効く。trace の巨大ジョブが作る極端な wait リターンが条件付け入力のダイナミックレンジを
# 広げ FiLM 等を不安定化 → loss 発散。これを soft-log で「小値は線形・大外れ値だけ圧縮」する。
# forward / predict_archive_value の両方（=train と eval が同じ変換を通る）に適用するので
# 自己整合（コマンド空間のミスキャリブレ無し）。reward/achieved-PF/rp baseline は生のまま不変。
# 既定 off で完全な no-op（ビット一致）。wait 次元 index=0（reward=[-waiting_time,-cost]）。
_COND_WAIT_ROBUST = os.environ.get("PCN_COND_WAIT_ROBUST", "off")  # off | softlog | logexpand
_COND_WAIT_K = float(os.environ.get("PCN_COND_WAIT_K", "10.0"))     # |x|<<K は線形, |x|>>K で圧縮
# [logexpand] 正規化後z空間で低wait帯を対数で「広げる」(softlogの逆向きの用途)。
# 背景: 5万jobsでwait正規化scaleが全オンプレ外れ値基準になり、勝負どころの0-30秒が
# z<0.013 に潰れて網に読めない(v9 wait不感・armA2の<30s全クラウド縮退の一因)。
# y = sign(z)·log1p(|z|/z0)/log1p(1/z0): 単調・符号保存・z=1で1。z0=1e-3で30秒(z=0.013)→0.38。
# lockstep_nn は同一モデルクラスを共有するため、評価側も同じ env を立てれば自動で一致。
_COND_WAIT_Z0 = float(os.environ.get("PCN_COND_WAIT_Z0", "1e-3"))

# --- 崩壊(複利ドリフト)対策: CE飽和ロックインの構造的封印 ---
# 崩壊runの CE 爆発 0.7→69 は log p(archive行動)≈-69 ＝ softmax 飽和(p≈1e-30)。一度飽和すると
# 当該サンプル不在時は勾配ゼロで戻る力が無く、ドリフトが不可逆化する。
# ②ラベル平滑化: 最適 logit ギャップを有界化し、p→0 の飽和 lock-in を原理的に不可能にする(既定OFF)。
_LABEL_SMOOTH = float(os.environ.get("PCN_LABEL_SMOOTH", "0"))
# 学習内部の細粒度プロファイル(batch/forward/loss/backward/opt)。DISTRIBUTED_PCN_PROFILE=1 で有効。
# self._prof_acc に累積し update_many の末尾で出力・リセット。計測のみで数式・結果に影響しない。
_PROFILE = os.environ.get("DISTRIBUTED_PCN_PROFILE", "0") == "1"
# _act(1ステップ推論)の forward を TorchScript trace 経由にする(Python層の除去)。
# freeze しない trace はモデルの Parameter/Buffer を参照のまま保持するため load_state_dict の
# 重み更新に自動追従し(検証済 equal=True)、出力は素の forward と完全ビット一致(3000サンプル
# mismatch=0)。単段 forward で 0.30→0.20ms/step(1.5×)。rollout/eval のエピソード実行が速くなる。
# dropout(_S_EMB_DROPOUT>0)有効時は乱数分岐が焼き込まれるため自動無効。既定OFF、1で有効。
_JIT_ACT = os.environ.get("PCN_JIT_ACT", "0") == "1"
# 注: バッチ準備のスレッドプリフェッチ(ダブルバッファ)は実装・ビット一致検証まで行ったが撤回した
# (2026-07-03)。理由=バッチ生成のGPU部分(index_select)はメインのforward/backwardと同一CUDAストリーム
# で直列化され、CPU部分はGILで直列化されるため、別スレッドに移しても時間はどこにも隠れない
# (bench実測: batch計時は0%になるが総時間20.4→20.6ms/updateで不変)。隠すには別streamと
# event同期が必要だが、CUDA実行時のbatchは18%しかなく上限1.2×で労力に見合わない。
# ①anchor-KL: 各 learn() 境界で方策スナップショットを凍結し、update 毎に
# KL(anchor‖online) を罰する=イテレーション内 proximal(政策チャーン抑制, CHAIN流)。
# anchor は毎イテレーション追従するので長期の学習進行は妨げない(既定OFF)。
_ANCHOR_KL_WEIGHT = float(os.environ.get("PCN_ANCHOR_KL_WEIGHT", "0"))


def _robust_cond_wait(desired_return):
    """条件付け入力の wait 次元(index 0)だけ soft-log 圧縮して返す（gate off で原値）。
    y = sign(x)*K*log1p(|x|/K): |x|<<K で≈x（順序・小値を保存）, |x|>>K で K*log に圧縮。"""
    if _COND_WAIT_ROBUST in ("off", "", "0") or desired_return.shape[-1] < 1:
        return desired_return
    dr = desired_return.clone()
    w = dr[..., 0]
    if _COND_WAIT_ROBUST == "logexpand":
        z0 = _COND_WAIT_Z0
        denom = float(np.log1p(1.0 / z0))
        dr[..., 0] = th.sign(w) * th.log1p(th.abs(w) / z0) / denom
    else:
        K = _COND_WAIT_K
        dr[..., 0] = th.sign(w) * K * th.log1p(th.abs(w) / K)
    return dr
# choose_commands の改善量スケール（1.0=原著実装相当、既定は大規模報酬向けに控えめ）
_COMMAND_ALPHA = float(os.environ.get("PCN_COMMAND_ALPHA", "0.2"))
# [PCN_CMD_LOCAL_STEP] 注文の改善幅を「前線の隣接点間隔」基準の局所ステップにする(オプトイン)。
# 論文式 U(0, 0.2σ)(σ=非支配前線全体の標準偏差=前線の幅)は、点が多く間隔が不均等な前線では
# 一歩が密な区間で数百点ぶん飛ぶ(実測: 間隔中央値の159-461倍)。ON時の作り方:
#   土台=前線から一様に1点 → 改善方向の隣接点 p_{i±1} へ α~U(0.5,1.5) で両軸内挿/外挿。
#   端点は確率 PCN_CMD_LOCAL_EDGE_FRAC(既定0.2) で内側隣接間隔と同じ幅だけ外側へ(前線を伸ばす役割)。
#   クランプ=達成済みレンジ±端の隣接間隔。レジーム標識(episode_regime_scale)があればレジーム別前線。
# 既定OFF=従来とビット一致。
_CMD_LOCAL_STEP = os.environ.get("PCN_CMD_LOCAL_STEP", "0") == "1"
_CMD_LOCAL_EDGE_FRAC = float(os.environ.get("PCN_CMD_LOCAL_EDGE_FRAC", "0.2"))
# [PCN_CMD_REACH_CLAMP] 局所ステップの端外挿を「到達済み端 × 係数」で頭打ちにする(既定OFF=0)。
# 背景: 既存クランプ上限は c_max + d_hi(端の隣接間隔)。実行可能集合が巨大ジョブで島状に分断
# されている([[attainable-set-islands]])と端の隣接間隔そのものが巨大になり、上限が達成済み端の
# 2倍近くまで開く。実測(weekA4096)では注文が真PF上限の1.7-1.9倍まで飛び、そこは実現不能なので
# 達成が頭打ち→注文と達成のズレ(L2)だけが増える。値 f>1 を指定すると cost 上端を c_max*f、
# wait 上端を w_max*f、wait 下端を w_min/f に制限する(到達幅は残しつつ外挿だけ抑える)。
# 0 または未設定なら従来と完全にビット一致。
_CMD_REACH_CLAMP = float(os.environ.get("PCN_CMD_REACH_CLAMP", "0") or 0.0)
# Anchored command pool: archive PF が崩壊しても command 分布が cost 全域を張るよう、
# workload calibration の (全OP, 全CL) 端点間を線形補間した固定 anchor を毎回 pool に混ぜる。
# 0 で無効（既定）。崩壊フィードバックループ（archive収縮→command収縮→探索収縮）を断ち切る。
_PF_COMMAND_ANCHORS = int(os.environ.get("PCN_PF_COMMAND_ANCHORS", "0"))
# MPFT型 端→内側掃引の command 生成 (survey_stream_pf.html: MPFT 2025 の PCN 翻訳)。
# 「まず端の方策を杭として固定し、端から内側へ PF をなぞる」。paper モードの
# 中間ナッジ(生成の96%が中間へ集中=端消失の機序)を端起点に反転する。0 で無効（既定・ビット一致）。
_MPFT_SWEEP = os.environ.get("PCN_MPFT_SWEEP", "0") == "1"
_MPFT_START_FRAC = float(os.environ.get("PCN_MPFT_START_FRAC", "0.15"))   # 掃引の初期到達率(各端から)
_MPFT_FULL_ITER = float(os.environ.get("PCN_MPFT_FULL_ITER", "40"))       # この iteration で全域到達(r=0.5)
_MPFT_ENDPOINT_QUOTA = float(os.environ.get("PCN_MPFT_ENDPOINT_QUOTA", "0.25"))  # 両端の杭に固定する命令割合
_MPFT_IMPROVE = float(os.environ.get("PCN_MPFT_IMPROVE", "0.02"))         # 掃引命令の Pareto 方向ナッジ(wait↓)
# 達成ゲート型 MPFT（単調に良くなる構造）。reach を時計(it/FULL_ITER)でなく「今の前線帯を
# 一発再現できたら1段広げる」達成ベースで進める。マスターするまで離れない=獲得点を忘れない=単調。
_MPFT_GATED = os.environ.get("PCN_MPFT_GATED", "0") == "1"
_MPFT_GATE_EPS = float(os.environ.get("PCN_MPFT_GATE_EPS", "0.15"))   # 前線帯の正規化再現gapがこれ未満=マスター
_MPFT_GATE_STEP = float(os.environ.get("PCN_MPFT_GATE_STEP", "0.05")) # マスター毎に reach を広げる幅
_MPFT_GATE_PATIENCE = int(os.environ.get("PCN_MPFT_GATE_PATIENCE", "1"))  # 連続何回マスターで前進
# 改善が起きている時に学習量(n_updates)を増やす適応制御。「改善する行動を促して改善が
# 起こった時、もっと多く学習を回す」= productive な局面に計算を厚く張る。ゲートが改善を
# 検知して倍率を上げ(learn() が読む)、伸び悩んだら1.0へ減衰。既定OFF=倍率1.0でビット一致。
_MPFT_VOL_ADAPT = os.environ.get("PCN_MPFT_VOL_ADAPT", "0") == "1"
_MPFT_VOL_RAMP = float(os.environ.get("PCN_MPFT_VOL_RAMP", "1.5"))    # 改善時に倍率を掛ける係数
_MPFT_VOL_MAX = float(os.environ.get("PCN_MPFT_VOL_MAX", "4.0"))      # 倍率上限
_MPFT_VOL_DECAY = float(os.environ.get("PCN_MPFT_VOL_DECAY", "0.7"))  # 非改善時に倍率を戻す係数
_MPFT_VOL_IMPROVE_EPS = float(os.environ.get("PCN_MPFT_VOL_IMPROVE_EPS", "0.02"))  # gap低下がこれ以上で改善判定
# 観測の符号付き log 圧縮。1024ジョブ等で時刻特徴が ~1e6 桁になり s_emb(Sigmoid)を
# 飽和させ conditioning を埋もれさせる対策。0 で無効（既定、24J 等の小規模は不要）。
_OBS_LOG_COMPRESS = os.environ.get("PCN_OBS_LOG", "0") == "1"
# replay学習時にArchive PF/端点を少し多めにサンプルする（1.0で無効化）
_TRAIN_PF_WEIGHT = float(os.environ.get("PCN_TRAIN_PF_WEIGHT", "4.0"))
_TRAIN_ENDPOINT_WEIGHT = float(os.environ.get("PCN_TRAIN_ENDPOINT_WEIGHT", "12.0"))
# Phase3で直近にAchievedした経験を、巨大archive内で埋もれにくくする
_TRAIN_RECENT_WEIGHT = float(os.environ.get("PCN_TRAIN_RECENT_WEIGHT", "6.0"))
# Cost端は構造的に単純な方策で出やすいはずなので、実際にarchiveへ入ったCost端だけ強めに模倣する
_TRAIN_COST_ENDPOINT_WEIGHT = float(os.environ.get("PCN_TRAIN_COST_ENDPOINT_WEIGHT", "32.0"))
# 中域 cost（膨らみ帯）のエピソードを replay で多めに学習（0=無効）
_TRAIN_MID_PF_WEIGHT = float(os.environ.get("PCN_TRAIN_MID_PF_WEIGHT", "0"))
_TRAIN_MID_COST_MIN_FRAC = float(os.environ.get("PCN_TRAIN_MID_COST_MIN_FRAC", "0.06"))
_TRAIN_MID_COST_MAX_FRAC = float(os.environ.get("PCN_TRAIN_MID_COST_MAX_FRAC", "0.38"))
# cost≈0.5e6 付近（low/mid 境界・膝前）を replay / 条件付けで厚く（0=無効）
_TRAIN_KNEE_PF_WEIGHT = float(os.environ.get("PCN_TRAIN_KNEE_PF_WEIGHT", "0"))
_TRAIN_KNEE_COST_MIN_FRAC = float(os.environ.get("PCN_TRAIN_KNEE_COST_MIN_FRAC", "0.04"))
_TRAIN_KNEE_COST_MAX_FRAC = float(os.environ.get("PCN_TRAIN_KNEE_COST_MAX_FRAC", "0.12"))
_TRAIN_KNEE_STEP_WEIGHT = float(os.environ.get("PCN_TRAIN_KNEE_STEP_WEIGHT", "0"))
# 左上先端（低 cost・高 wait からの滑らかな下降）: cost 0〜max_frac を replay / 条件付けで厚く
_TRAIN_LOW_SLOPE_PF_WEIGHT = float(os.environ.get("PCN_TRAIN_LOW_SLOPE_PF_WEIGHT", "0"))
_TRAIN_LOW_SLOPE_COST_MIN_FRAC = float(os.environ.get("PCN_TRAIN_LOW_SLOPE_COST_MIN_FRAC", "0.0"))
_TRAIN_LOW_SLOPE_COST_MAX_FRAC = float(os.environ.get("PCN_TRAIN_LOW_SLOPE_COST_MAX_FRAC", "0.18"))
_TRAIN_LOW_SLOPE_STEP_WEIGHT = float(os.environ.get("PCN_TRAIN_LOW_SLOPE_STEP_WEIGHT", "0"))
# 低 wait（多くは高 cost 側）の達成 episode を teacher replay で厚くする
_TRAIN_LOW_WAIT_PF_WEIGHT = float(os.environ.get("PCN_TRAIN_LOW_WAIT_PF_WEIGHT", "0"))
_TRAIN_LOW_WAIT_MAX = float(os.environ.get("PCN_TRAIN_LOW_WAIT_MAX", "0"))
_TRAIN_LOW_WAIT_FRAC = float(os.environ.get("PCN_TRAIN_LOW_WAIT_FRAC", "0"))


def _pf_region_balanced_weight(weight: float, count: int) -> float:
    """件数不変な領域重み。PCN_TRAIN_PF_BALANCE_REF=REF>0 のとき weight*REF/count を返し、
    各PF領域(cheap=Cost端 / expensive=LowWait / mid 等)の「総重量(=件数×重み)」を件数に依らず
    ≈一定にする。cheap/expensive のエピソード件数はスケールで激しく偏る(例: 1024で cheap 1個 vs
    expensive 352個)ため、固定 per-episode 重みだと総量バランスが崩れ、大規模で expensive 一色→
    安い角を学べない。件数正規化すると balance がスケール不変になる。0(既定)で従来挙動。"""
    ref = float(os.environ.get("PCN_TRAIN_PF_BALANCE_REF", "0"))
    if ref > 0.0 and count > 0:
        # α=1: 完全件数正規化(総量一定)。α<1: 部分正規化(emphasis をスケールで緩やかに逓減)＝
        # 小規模では領域重視を残し(256 の span を保ち)、大規模では支配を崩す(1024 を修復)。
        alpha = float(os.environ.get("PCN_TRAIN_PF_BALANCE_ALPHA", "1.0"))
        return weight * (ref / float(count)) ** alpha
    return weight


# ビンレス PF 密度逆数重み: PF点を cost-wait 正規化空間へ置き、各点の k 番目最近傍距離 r_k を
# 「局所密度の逆指標」として weight ∝ (r_k/mean)^ALPHA を与える。PF上で混んでいる(密)点は小さく、
# スカスカな点は大きく → 領域ごとの件数を手動帯なしで自動均等化(件数逆比サンプリングの連続版)。
# 手動帯(cost端/mid/low_wait など)の固定%マジックナンバーを置換する狙い。0(既定)で無効。
_TRAIN_PF_DENSITY_WEIGHT = float(os.environ.get("PCN_TRAIN_PF_DENSITY_WEIGHT", "0"))
_TRAIN_PF_DENSITY_K = int(os.environ.get("PCN_TRAIN_PF_DENSITY_K", "2"))
_TRAIN_PF_DENSITY_ALPHA = float(os.environ.get("PCN_TRAIN_PF_DENSITY_ALPHA", "1.0"))
# Archive 低 cost PF: r1 固定・r0(wait) を振ったとき方策が分岐（左上プラトー対策）
_LOW_BAND_COND_WEIGHT = float(os.environ.get("PCN_LOW_BAND_COND_WEIGHT", "0"))
_LOW_BAND_COND_WAIT_LEVELS = int(os.environ.get("PCN_LOW_BAND_COND_WAIT_LEVELS", "8"))
_LOW_BAND_COND_COST_LEVELS = int(os.environ.get("PCN_LOW_BAND_COND_COST_LEVELS", "8"))
_LOW_BAND_COND_MAX_SAMPLES = int(os.environ.get("PCN_LOW_BAND_COND_MAX_SAMPLES", "48"))
_LOW_BAND_COND_KL_MARGIN = float(os.environ.get("PCN_LOW_BAND_COND_KL_MARGIN", "0.10"))
# arc=Archive PF 上の連続 command ペア, r1_sweep=r0 固定で r1 を振る, r0_sweep=r1 固定で r0 を振る
_LOW_BAND_COND_MODE = os.environ.get("PCN_LOW_BAND_COND_MODE", "arc").strip().lower()
_LOW_BAND_COND_MIN_R1_SEP_FRAC = float(os.environ.get("PCN_LOW_BAND_COND_MIN_R1_SEP_FRAC", "0.002"))
# dual モード: r1_sweep と arc の損失配分（既定 0.5 / 0.5）
_LOW_BAND_DUAL_R1_FRAC = float(os.environ.get("PCN_LOW_BAND_DUAL_R1_FRAC", "0.5"))
# 巨大ジョブ決定ステップ重み(本来の筋・裾濃度への直接処方): 各ステップの cost 報酬の大きさ
# (=desired_returnsのr1階差)で「巨大ジョブをどこに置くか」の高レバレッジ決定を特定し学習を集中させる。
# 裾の重いワークロード(256)では少数の巨大ジョブの二択がPFを支配するのに、何百の小ジョブに埋もれて
# 学習信号が希薄化する。それをエピソード内 top-frac の cost-delta ステップへ重みで集中する。配管不要。
_TRAIN_GIANT_STEP_WEIGHT = float(os.environ.get("PCN_TRAIN_GIANT_STEP_WEIGHT", "0"))
_TRAIN_GIANT_FRAC = float(os.environ.get("PCN_TRAIN_GIANT_FRAC", "0.06"))  # 上位~6%=~15/256ジョブ
# ステップ単位: 中域 cost command（残り return の r1）と eval 相当の序盤ステップを多めに学習
_TRAIN_MID_STEP_WEIGHT = float(os.environ.get("PCN_TRAIN_MID_STEP_WEIGHT", "0"))
_TRAIN_EVALIKE_STEP_WEIGHT = float(os.environ.get("PCN_TRAIN_EVALIKE_STEP_WEIGHT", "0"))
_TRAIN_EVALIKE_STEP_FRAC = float(os.environ.get("PCN_TRAIN_EVALIKE_STEP_FRAC", "0.12"))
# [PCN_TRAIN_HEAD_STEP_WEIGHT] 学習は各エピソードのステップを一様に引くため、条件(残りリターン)は
# 「途中の小さい値」に偏る(18J実測: 52%が先頭値の半分以下)。一方、注文・評価は必ず先頭条件 G_0 を使う。
# 先頭条件が学習で見られる割合は 1/T(18Jで5.6%, 4096Jでは0.02%)しかなく、これが「注文どおりに出せない」
# 分布ずれの主因候補。ON時は各エピソード先頭 FRAC 区間のステップ重みを ×WEIGHT する。既定1.0=OFFで不変。
_TRAIN_HEAD_STEP_WEIGHT = float(os.environ.get("PCN_TRAIN_HEAD_STEP_WEIGHT", "1.0"))
_TRAIN_HEAD_STEP_FRAC = float(os.environ.get("PCN_TRAIN_HEAD_STEP_FRAC", "0.1"))
# Archive 中域 PF 上で r1 固定・r0(wait) を振ったとき方策が分岐するよう促す（評価グリッドの弱点）
_MID_BAND_COND_WEIGHT = float(os.environ.get("PCN_MID_BAND_COND_WEIGHT", "0"))
_MID_BAND_COND_WAIT_LEVELS = int(os.environ.get("PCN_MID_BAND_COND_WAIT_LEVELS", "5"))
_MID_BAND_COND_COST_LEVELS = int(os.environ.get("PCN_MID_BAND_COND_COST_LEVELS", "4"))
_MID_BAND_COND_MAX_SAMPLES = int(os.environ.get("PCN_MID_BAND_COND_MAX_SAMPLES", "48"))
_MID_BAND_COND_KL_MARGIN = float(os.environ.get("PCN_MID_BAND_COND_KL_MARGIN", "0.12"))
# wait-KL の Archive コスト錨を狭帯域に絞る（0=中域 frac 全体、>0 は中心 frac）
_MID_BAND_COND_FOCUS_FRAC = float(os.environ.get("PCN_MID_BAND_COND_FOCUS_FRAC", "0"))
_MID_BAND_COND_FOCUS_HALF_WIDTH_FRAC = float(
    os.environ.get("PCN_MID_BAND_COND_FOCUS_HALF_WIDTH_FRAC", "0.04")
)
# 正規化後の command 各次元が c_emb へ同等に効くようバランス（cost 一方通行を緩和）
_COMMAND_BALANCE = os.environ.get("PCN_COMMAND_BALANCE", "0") == "1"
# 命令balance の強度: 適用値 = command_balance ** power。0=無効([1,1])、1=full、0.5=中間。
# full balance は速い端を出すが左/中央を過補正で犠牲にするため、部分balanceで両立を狙う。
_COMMAND_BALANCE_POWER = float(os.environ.get("PCN_COMMAND_BALANCE_POWER", "1.0"))
# balance power を front の左右非対称に適応して自己調整(既定OFF)。手動 power を消す。
# 到達PFの「左半分(安)の点割合」が target を超えたら power↑(速い命令を強め右へ引く)、下回れば power↓。
_COMMAND_BALANCE_ADAPT = os.environ.get("PCN_COMMAND_BALANCE_ADAPT", "0") == "1"
_COMMAND_BALANCE_STEP = float(os.environ.get("PCN_COMMAND_BALANCE_STEP", "0.08"))      # 1回の power 調整幅
_COMMAND_BALANCE_PMAX = float(os.environ.get("PCN_COMMAND_BALANCE_PMAX", "1.0"))       # power 上限
# frozen-PF cloning: best-ever 非支配エピソードを凍結保持し、phase-3 教師に常時含める。
# 自己強化崩壊（劣化した自身の rollout を模倣→loss上昇→command無視）を断ち、
# 「劣化しない良いフロント」を behavior-clone してEvalで再現させる。
_FROZEN_PF_CLONE = os.environ.get("PCN_FROZEN_PF_CLONE", "0") == "1"
_FROZEN_PF_MAX = int(os.environ.get("PCN_FROZEN_PF_MAX", "256"))
# [PCN_TEACH_FRONT_ONLY] 教師データを凍結アーカイブ(非支配フロント)だけに絞る統制実験用フラグ。
# PCN_FROZEN_PF_CLONE=1 が前提。既定(0)は従来どおり replay 全件 + frozen の和集合(ビット不変)。
# 1 のときは replay 全件を捨て、_frozen_pf_entries（非支配点のみ・PCN_FROZEN_PF_MAX で間引き量を
# 制御）だけを教師にする＝「データの組成」（希釈データ由来のダウンか）を単離するための片翼。
_TEACH_FRONT_ONLY = os.environ.get("PCN_TEACH_FRONT_ONLY", "0") == "1"
# モデル重みEMA(Polyak averaging): 続学習が効率方策を壊すのを eval/save 重みで平滑化する。
# rollout(探索) は online 重みのまま、eval と save のときだけ EMA 重みへ swap する。0=OFF。
_EMA_DECAY = float(os.environ.get("PCN_EMA_DECAY", "0"))  # 0.99(時定数~100step)/0.999(~1000step)推奨
_EMA_ENABLED = _EMA_DECAY > 0.0
_ADAPTIVE_RETURN_NORMALIZATION = os.environ.get("PCN_ADAPTIVE_RETURN_NORMALIZATION", "1") == "1"
_RETURN_NORM_EMA = float(os.environ.get("PCN_RETURN_NORM_EMA", "0.05"))
_RETURN_NORM_MIN_SCALE = float(os.environ.get("PCN_RETURN_NORM_MIN_SCALE", "1e-6"))
# Phase3: 新規エピソードのみ GPU 教師 cache へ追記（全件再構築を避ける）
_TRAINING_CACHE_INCREMENTAL = os.environ.get("PCN_TRAINING_CACHE_INCREMENTAL", "1") == "1"
# Phase1 action-sweep エピソードの学習重み（1.0=無効、0.2=5倍薄める）
_PHASE1_SWEEP_TRAIN_WEIGHT = float(os.environ.get("PCN_PHASE1_SWEEP_TRAIN_WEIGHT", "1.0"))
# NSGA種まきエピソードの学習サンプリング優先倍率(頻度戦争対策)。既定1=完全従来。
_SEED_EPISODE_WEIGHT = float(os.environ.get("PCN_SEED_EPISODE_WEIGHT", "1"))
# [PCN_DEDUP_TRAIN_WEIGHT] 同じ達成点(cost, 平均待ち)を持つエピソードが重複していると、
# 一様抽選では「その1点」に学習機会が集中する(18Jで最頻点=全オンプレ1通りが2128本中709本=33%を占有し、
# 前線1点あたりの機会1.1%に対し30倍の偏り)。ON時は同一達成点グループの重みを 1/本数 にして
# 「点ごとに均等」にする(データは捨てない)。既定OFF=従来どおりビット不変。
_DEDUP_TRAIN_WEIGHT = os.environ.get("PCN_DEDUP_TRAIN_WEIGHT", "0") == "1"
_DEDUP_TRAIN_DECIMALS = int(os.environ.get("PCN_DEDUP_TRAIN_DECIMALS", "2"))
# [PCN_ADV_WEIGHT] 帯内相対成績による重み(良し悪しの勾配)。
# 目的: 一様模倣は「多数派の平均的な振る舞い」に丸まり、前線に載る鋭い選択が薄まる
# (18J実測: 前線に載る試行が54%→12%へ低下)。同じコスト帯の仲間と比べて待ちが短い経験ほど
# 強く写すことで、外から「良い基準」を注入せずデータ内の相対比較だけで鋭さを保つ。
#   帯: コスト軸を BANDS 分位で分割(帯ごとに比較=前線が1点に潰れるのを防ぐ)
#   重み: exp( (帯中央値 - 自分の待ち) / (帯の待ちのばらつき) / TEMP ) を [1/CLIP, CLIP] にクリップ
# 既定0=OFFでビット不変。
_ADV_WEIGHT = float(os.environ.get("PCN_ADV_WEIGHT", "0"))
_ADV_BANDS = int(os.environ.get("PCN_ADV_BANDS", "8"))
_ADV_CLIP = float(os.environ.get("PCN_ADV_CLIP", "8"))
# [PCN_REPLAY_REGIME_FAIR] replay淘汰のレジーム公平化(オプトイン)。PCN_MIX_REGIMES学習で
# PF-crowding淘汰がレジーム盲目に働くと、(cost,wait)座標で見かけ優秀な最空きレジームが
# バッファを占拠し、評価レジームのNSGA種リプレイ(真PF水準の見本)が全滅する(4環診断)。
# 1= ①淘汰をレジーム別クォータ制(最超過レジームから追い出し) ②_nlargestのヒープ優先度を
# レジーム内ND/crowdingで計算 ③種エピソード(_pcn_seed_episode)は同レジームに非種が残る限り保護。
# 既定0=完全従来(ビット不変)。コマンド生成用の選抜(_nlargest返り値)は両モードで不変。
_REPLAY_REGIME_FAIR = os.environ.get("PCN_REPLAY_REGIME_FAIR", "0") == "1"
# 学習時のみ s_emb へ dropout（条件 c を使わざるを得なくする）
_S_EMB_DROPOUT = float(os.environ.get("PCN_S_EMB_DROPOUT", "0"))
# 同じ obs で desired_return を変えたときの方策差を KL で促す（0=無効）
_CONDITIONING_SENS_WEIGHT = float(os.environ.get("PCN_CONDITIONING_SENS_WEIGHT", "0"))
_CONDITIONING_KL_MARGIN = float(os.environ.get("PCN_CONDITIONING_KL_MARGIN", "0.08"))
_CONDITIONING_SENS_MAX_PAIRS = int(os.environ.get("PCN_CONDITIONING_SENS_MAX_PAIRS", "64"))
_CONDITIONING_SENS_OBS_THRESH = float(os.environ.get("PCN_CONDITIONING_SENS_OBS_THRESH", "1e-3"))
_CONDITIONING_SENS_DR_THRESH = float(os.environ.get("PCN_CONDITIONING_SENS_DR_THRESH", "1e-4"))
# r0(total_wait command) 差がこの閾値以上のペアを hinge-KL 対象に（wait 追従強化）
_CONDITIONING_SENS_WAIT_DR_THRESH = float(
    os.environ.get("PCN_CONDITIONING_SENS_WAIT_DR_THRESH", "0")
)
_COND_ADD_SCALE = float(os.environ.get("PCN_COND_ADD_SCALE", "0"))
# FiLM 条件付け（既定OFF）: 乗算ゲート fc(s*c) を fc(s*γ(c)+β(c)) に置換。γ≈1/β≈0 で初期化し
# 始めは fc(s) 同等（安定）、学習で条件変調を獲得。加算β(c)が乗算ゲートで消える極値の
# 条件retrieval（cost-0角など）を補い、hinge-KL圧では届かなかった追従を狙う。
_FILM = os.environ.get("PCN_FILM", "0") == "1"
# b2: Fourier command encoding。条件 c（正規化済み, 各次元~O(1)）を sin/cos の高周波特徴に
# 展開してから条件線形層（FiLM/c_emb）へ入れる。近い指令を高次元で分離し、条件付けの
# 「分解能（追従性）」を上げる。生 c も連結するのでフォールバック可（学習で使う/捨てるを選べる）。
# 周波数は 2^[0..L-1]（既定 L=4 → [1,2,4,8]）。z-score間隔~0.15 を分離できる控えめな範囲。
_FOURIER_CMD = os.environ.get("PCN_FOURIER_CMD", "0") == "1"
_FOURIER_BANDS = int(os.environ.get("PCN_FOURIER_BANDS", "4"))
# [per-channel bands] cost指令チャネル(指令ベクトル第2成分=index1; c=[wait*nj, cost, horizon])の
# バンド数を個別指定。診断: 共通バンドの最高周波数成分が cost 写像に「さざ波」を作り、密掃引の
# 凹み位置(正規化0.205/0.41)が最高周波の半周期/全周期と一致(wait側は健全)。cost側だけ高周波を
# 落として写像を平滑化する。未設定(空)なら従来=PCN_FOURIER_BANDS と同一でビット不変。
_FOURIER_BANDS_COST_RAW = os.environ.get("PCN_FOURIER_BANDS_COST", "")
_FOURIER_BANDS_COST = int(_FOURIER_BANDS_COST_RAW) if _FOURIER_BANDS_COST_RAW != "" else None
_FOURIER_COST_CH = 1  # cost チャネル index（objectives_to_command: dr=[-wait*nj, -cost]）
# Fourier 周波数モード（生 c は常に連結＝NNが必要な周波数を学習で選ぶ）:
#   geometric(既定): 2^[0..L-1]=[1,2,4,8] NeRF式。高周波が急増→小規模で過剰=崩壊しやすい。
#   linear         : (k+1)*base=[1,2,3,4]*base。2π,4π,6π… の古典フーリエ倍音。高周波が急増せず小規模で滑らか。
#   gaussian       : |N(0,1)|*scale の random Fourier features(Tancik2020)。scale が性能を支配・seed固定で再現。
_FOURIER_MODE = os.environ.get("PCN_FOURIER_MODE", "geometric")
_FOURIER_BASE = float(os.environ.get("PCN_FOURIER_BASE", "1.0"))    # linear: freq=(k+1)*base（2π指定なら ~6.283）
_FOURIER_SCALE = float(os.environ.get("PCN_FOURIER_SCALE", "1.0"))  # gaussian: freq~|N(0,1)|*scale
_FOURIER_SEED = int(os.environ.get("PCN_FOURIER_SEED", "0"))        # gaussian 周波数の固定seed（全モデルで同一）
_VALUE_REPRO_WEIGHT = float(os.environ.get("PCN_VALUE_REPRO_WEIGHT", "0"))
_VALUE_COST_SCALE = float(os.environ.get("PCN_VALUE_COST_SCALE", "100000.0"))
# [案1: 指令追従loss] 指令costに「達成が届かない側だけ」を二乗罰する片側MSE（cost成分）。
# value_head proxy で達成を予測し、s_emb を detach して方策の状態表現を汚さない（既存 value_repro が
# Phase3後半崩壊した機序を断つ）。c_emb は共有し追従勾配を方策へ流す。正規化空間・fp32・片側ヒンジ。
# 既定0でビット一致。qd256 は既に届くので relu=0 で競合せず、dens3v2(達成cost過大)だけ底上げ。
# ANCHOR=v̂を実達成に固定する回帰の相対重み（v̂が嘘で track を騙すのを防ぐ・必須）。
_CMD_TRACK_WEIGHT = float(os.environ.get("PCN_CMD_TRACK_WEIGHT", "0"))
_CMD_TRACK_ANCHOR_WEIGHT = float(os.environ.get("PCN_CMD_TRACK_ANCHOR_WEIGHT", "1.0"))
_CMD_TRACK_MAX_EPISODES = int(os.environ.get("PCN_CMD_TRACK_MAX_EPISODES", "48"))
# [v10: 両側の距離罰] wait側の片側罰 relu(v̂_wait − 指令wait)²。cost側(v9で追従形成の実績)の対称版。
# 実効重みが PCN_CMD_TRACK_WAIT_WEIGHT になるよう、関数内では (WAIT_W/COST_W) 倍で track に合算する
# (外側で COST_W が掛かるため)。既定0=ビット一致。PCN_CMD_TRACK_WEIGHT>0 が前提(0だと関数ごと不発)。
_CMD_TRACK_WAIT_WEIGHT = float(os.environ.get("PCN_CMD_TRACK_WAIT_WEIGHT", "0"))
# [critic対応: サーキットブレーカ] cmd_track損失がこの閾値を超えたら「その更新では加算しない」
# (armA3でcold+凍結教材時に総損失~100へ暴走した前科。夜間無人運転の安全弁)。0で無効。
_CMD_TRACK_BREAKER = float(os.environ.get("PCN_CMD_TRACK_BREAKER", "10"))
# waitヒンジの対数空間z0。入力側(_COND_WAIT_Z0)とは空間が違う(こちらはbalance前のavg-wait正規化、
# v_wait_n=avg/2286s級)ので独立の定数。1e-3で10秒→0.24, 30秒→0.38(狙い帯に勾配が立つ)。
_CMD_TRACK_WAIT_Z0 = float(os.environ.get("PCN_CMD_TRACK_WAIT_Z0", "1e-3"))
# [anti-ration] cost成分の desired_return を decrement せず初期目標で一定保持。既定0=従来(ビット一致)。
# trace の高cost端飽和の真因 = return-to-go rationing(残予算が減ると終盤の巨大ジョブをcloudに出せず飽和,
# corr(残予算,P_cloud)=+0.78)。cost目標を一定に保つと巨大cloudを継続でき expensive端へ届く(eval検証 ×2.46)。
# train/eval 両方に適用が必須(条件一致)。distributed_pcn.py:1390 actor探索側にも同名gate。
_COST_HOLD = os.environ.get("PCN_COST_HOLD", "0") == "1"
# weight decay (L2正則化)。既定0=従来(正則化なし=学習で重み膨張→logit飽和→条件付け死)。
# 学習を続けても条件付けを維持するための根本手当て候補。Adam の weight_decay に渡す。
_WEIGHT_DECAY = float(os.environ.get("PCN_WEIGHT_DECAY", "0"))
# 重みノルム天井(max-norm制約)。壁1(重み膨張2.5-4×→logit飽和→命令無視→一点collapse)を、L2の
# 一律縮小(反証済)でなく「Phase3で各重み行列のノルムが基準比 FACTOR 倍を超えたら物理的に縮める」で
# 直接止める。基準=Phase3最初の更新時ノルム。既定0=無効(ビット一致)。例1.5=1.5倍で頭打ち。
_WEIGHT_MAXNORM_FACTOR = float(os.environ.get("PCN_WEIGHT_MAXNORM_FACTOR", "0"))
# 非有限な勾配のときに optimizer.step() をスキップする（既定 ON）。
# 損失が有限でも勾配が NaN/Inf になり得る（巨大 logit の log_softmax 等）。clip_grad_norm_ は
# total_norm=NaN を coef=NaN にしてしまい全重みを NaN 化→nan_to_num=0→constant出力→command無視で
# 方策が「永久崩壊」する。step をスキップして NaN を重みに焼き付けない。0 で旧挙動（崩壊）。
_NAN_SKIP_STEP = os.environ.get("PCN_NAN_SKIP_STEP", "1") == "1"
_nan_skip_count = 0
_NAN_SKIP_WARN_LIMIT = int(os.environ.get("PCN_NAN_SKIP_WARN_LIMIT", "20"))
# 条件付けKL項のnan-safe化(既定ON, PCN_KL_NANSAFE=0で旧挙動)。logit飽和(p=厳密0)で
# xlogy backwardがnan→全step skip→凍結runになるのを発生源で断つ。床=log(1e-12)は健全域で実質不変。
_KL_NANSAFE = os.environ.get("PCN_KL_NANSAFE", "1") == "1"
_KL_LOG_FLOOR = float(np.log(1e-12))
# 損失スパイク検出→step skip(NaN skipの拡張): 損失が移動平均のRATIO倍を超えたら、その発散勾配で
# 重みを更新しない。後半の膝集中で一部runが損失爆発(0.7→69)するのを直接制動し前半の良い方策を守る。
_LOSS_SPIKE_SKIP = os.environ.get("PCN_LOSS_SPIKE_SKIP", "0") == "1"
_LOSS_SPIKE_RATIO = float(os.environ.get("PCN_LOSS_SPIKE_RATIO", "3.0"))

# update_many の 1 回（= 1 イテレーションの N 回勾配更新）の間は experience_replay が
# 凍結されるため、毎更新で再計算していた「Archive PF→帯域 command 抽出」「破棄される診断
# metrics の .item() 同期」「per-update の loss.item() 同期」を 1 回にまとめる。数式・乱数・
# 更新回数は不変（結果はビット一致）。0 で旧挙動。
_FAST_UPDATE = os.environ.get("PCN_FAST_UPDATE", "1") == "1"


def refresh_train_env_weights() -> None:
    """プロファイル適用後に import 時定数を環境変数から再読込。"""
    global _TRAIN_MID_PF_WEIGHT, _TRAIN_MID_COST_MIN_FRAC, _TRAIN_MID_COST_MAX_FRAC
    global _TRAIN_KNEE_PF_WEIGHT, _TRAIN_KNEE_COST_MIN_FRAC, _TRAIN_KNEE_COST_MAX_FRAC
    global _TRAIN_LOW_SLOPE_PF_WEIGHT, _TRAIN_LOW_SLOPE_COST_MIN_FRAC, _TRAIN_LOW_SLOPE_COST_MAX_FRAC
    global _TRAIN_LOW_WAIT_PF_WEIGHT, _TRAIN_LOW_WAIT_MAX, _TRAIN_LOW_WAIT_FRAC
    global _TRAIN_MID_STEP_WEIGHT, _TRAIN_EVALIKE_STEP_WEIGHT, _TRAIN_EVALIKE_STEP_FRAC
    global _TRAIN_COST_ENDPOINT_WEIGHT, _TRAIN_KNEE_STEP_WEIGHT, _TRAIN_LOW_SLOPE_STEP_WEIGHT
    global _TRAIN_GIANT_STEP_WEIGHT, _TRAIN_GIANT_FRAC
    _TRAIN_MID_PF_WEIGHT = float(os.environ.get("PCN_TRAIN_MID_PF_WEIGHT", "0"))
    _TRAIN_MID_COST_MIN_FRAC = float(os.environ.get("PCN_TRAIN_MID_COST_MIN_FRAC", "0.06"))
    _TRAIN_MID_COST_MAX_FRAC = float(os.environ.get("PCN_TRAIN_MID_COST_MAX_FRAC", "0.38"))
    _TRAIN_KNEE_PF_WEIGHT = float(os.environ.get("PCN_TRAIN_KNEE_PF_WEIGHT", "0"))
    _TRAIN_KNEE_COST_MIN_FRAC = float(os.environ.get("PCN_TRAIN_KNEE_COST_MIN_FRAC", "0.04"))
    _TRAIN_KNEE_COST_MAX_FRAC = float(os.environ.get("PCN_TRAIN_KNEE_COST_MAX_FRAC", "0.12"))
    _TRAIN_LOW_SLOPE_PF_WEIGHT = float(os.environ.get("PCN_TRAIN_LOW_SLOPE_PF_WEIGHT", "0"))
    _TRAIN_LOW_SLOPE_COST_MIN_FRAC = float(os.environ.get("PCN_TRAIN_LOW_SLOPE_COST_MIN_FRAC", "0.0"))
    _TRAIN_LOW_SLOPE_COST_MAX_FRAC = float(os.environ.get("PCN_TRAIN_LOW_SLOPE_COST_MAX_FRAC", "0.18"))
    _TRAIN_LOW_WAIT_PF_WEIGHT = float(os.environ.get("PCN_TRAIN_LOW_WAIT_PF_WEIGHT", "0"))
    _TRAIN_LOW_WAIT_MAX = float(os.environ.get("PCN_TRAIN_LOW_WAIT_MAX", "0"))
    _TRAIN_LOW_WAIT_FRAC = float(os.environ.get("PCN_TRAIN_LOW_WAIT_FRAC", "0"))
    _TRAIN_MID_STEP_WEIGHT = float(os.environ.get("PCN_TRAIN_MID_STEP_WEIGHT", "0"))
    _TRAIN_EVALIKE_STEP_WEIGHT = float(os.environ.get("PCN_TRAIN_EVALIKE_STEP_WEIGHT", "0"))
    _TRAIN_EVALIKE_STEP_FRAC = float(os.environ.get("PCN_TRAIN_EVALIKE_STEP_FRAC", "0.12"))
    _TRAIN_COST_ENDPOINT_WEIGHT = float(os.environ.get("PCN_TRAIN_COST_ENDPOINT_WEIGHT", "32.0"))
    _TRAIN_KNEE_STEP_WEIGHT = float(os.environ.get("PCN_TRAIN_KNEE_STEP_WEIGHT", "0"))
    _TRAIN_LOW_SLOPE_STEP_WEIGHT = float(os.environ.get("PCN_TRAIN_LOW_SLOPE_STEP_WEIGHT", "0"))
    _TRAIN_GIANT_STEP_WEIGHT = float(os.environ.get("PCN_TRAIN_GIANT_STEP_WEIGHT", "0"))
    _TRAIN_GIANT_FRAC = float(os.environ.get("PCN_TRAIN_GIANT_FRAC", "0.06"))


_VALUE_WAIT_SCALE = float(os.environ.get("PCN_VALUE_WAIT_SCALE", "500.0"))
_VALUE_REPRO_MAX_EPISODES = int(os.environ.get("PCN_VALUE_REPRO_MAX_EPISODES", "48"))
_MODEL_NAN_WARN_LIMIT = int(os.environ.get("PCN_MODEL_NAN_WARN_LIMIT", "5"))
_model_nan_warn_count = 0
# forward の NaN/Inf 検査を旧来の .any()+print 経路にする（既定 OFF=高速パス）。
# 高速パスは th.nan_to_num を無条件適用する: NaN/Inf が無ければ恒等変換なのでクリーンrunで
# bit 一致を保ちつつ、.any() の reduction が起こす host 同期（GPU→CPU）を消す。
# "1" で旧来の if .any(): _warn_nan; nan_to_num 経路（診断 print 復活）に戻す。
_FWD_NANCHECK = os.environ.get("PCN_FWD_NANCHECK", "0") == "1"
# update 内の入力(obs/desired_return/desired_horizon) と出力(prediction_logits) の
# 純診断 NaN/Inf チェック(print のみ・補正なし)を有効化する（既定 OFF=スキップ）。
# クリーンrunでは元々 print されないので bit 一致のまま、.any() 同期コストだけが消える。
# 出力側の step skip は AMP の GradScaler / 非AMP の _NAN_SKIP_STEP が backward 後に二重に担うため、
# この診断ブロックを既定スキップしても安全網は維持される。"1" で旧来の診断 print 経路に戻す。
_UPDATE_NANCHECK = os.environ.get("PCN_UPDATE_NANCHECK", "0") == "1"

# 固定シードを削除して多様性を確保
# np.random.seed(42)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Linuxで使用可能なフォントを自動設定
def setup_linux_fonts():
    """Linuxで使用可能なフォントを自動設定"""
    import matplotlib.font_manager as fm
    import platform
    
    # Linuxシステムでのみ実行
    if platform.system() == 'Linux':
        # 利用可能なフォントを取得
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 優先順位付きでフォントを選択
        preferred_fonts = [
            'DejaVu Sans',
            'Liberation Sans',
            'Ubuntu',
            'Noto Sans CJK JP',
            'Noto Sans',
            'Arial',
            'Helvetica'
        ]
        
        # 利用可能なフォントから選択
        selected_font = None
        for font in preferred_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        # フォントが見つからない場合は、利用可能なフォントから最初のものを使用
        if selected_font is None and available_fonts:
            # sans-serif系のフォントを優先
            sans_serif_fonts = [f for f in available_fonts if 'sans' in f.lower() or 'sans' in f.lower()]
            if sans_serif_fonts:
                selected_font = sans_serif_fonts[0]
            else:
                selected_font = available_fonts[0]
        
        if selected_font:
            plt.rcParams['font.family'] = selected_font
            plt.rcParams['font.sans-serif'] = [selected_font]
            print(f"フォントを設定しました: {selected_font}")
        else:
            print("警告: 利用可能なフォントが見つかりませんでした")

# Linuxフォントを自動設定
# setup_linux_fonts()

from numba import njit

from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.performance_indicators import hypervolume
from src.utils.map_visualizer import visualize_map


# 非支配解を取得（Numba JIT版・精度同一）
@njit(cache=True)
def _get_non_dominated_inds_maximize_numba(points):
    """最大化: iがjを支配 → i>=j全次元 かつ i>jが1次元以上。jを除去するのはiがjを支配するとき"""
    n, d = points.shape
    is_efficient = np.ones(n, dtype=np.bool_)
    for i in range(n):
        if is_efficient[i]:
            for j in range(n):
                if is_efficient[j] and i != j:
                    all_ge = True
                    some_gt = False
                    for k in range(d):
                        if points[i, k] < points[j, k]:
                            all_ge = False
                            break
                        if points[i, k] > points[j, k]:
                            some_gt = True
                    if all_ge and some_gt:
                        is_efficient[j] = False
            is_efficient[i] = True
    return np.nonzero(is_efficient)[0]


@njit(cache=True)
def _get_non_dominated_inds_minimize_numba(points):
    """最小化: iがjを支配 → i<=j全次元 かつ i<jが1次元以上"""
    n, d = points.shape
    is_efficient = np.ones(n, dtype=np.bool_)
    for i in range(n):
        if is_efficient[i]:
            for j in range(n):
                if is_efficient[j] and i != j:
                    all_le = True
                    some_lt = False
                    for k in range(d):
                        if points[i, k] > points[j, k]:
                            all_le = False
                            break
                        if points[i, k] < points[j, k]:
                            some_lt = True
                    if all_le and some_lt:
                        is_efficient[j] = False
            is_efficient[i] = True
    return np.nonzero(is_efficient)[0]


def get_non_dominated_inds(points):
    """非支配解（最大化問題用）のインデックスを取得（Numba JITで高速化・精度同一）"""
    if len(points) == 0:
        return np.array([])
    pts = np.ascontiguousarray(np.array(points, dtype=np.float64))
    return _get_non_dominated_inds_maximize_numba(pts)


def get_non_dominated_inds_minimize(points):
    """非支配解（最小化問題用）のインデックスを取得（Numba JITで高速化・精度同一）"""
    if len(points) == 0:
        return np.array([])
    pts = np.ascontiguousarray(np.array(points, dtype=np.float64))
    return _get_non_dominated_inds_minimize_numba(pts)


@njit(cache=True)
def _crowding_distance_numba(points):
    """混雑度計算（Numba JIT・精度同一）"""
    n, d = points.shape
    if n <= 2:
        return np.ones(n, dtype=np.float64)
    min_vals = np.empty(d)
    ptp_vals = np.empty(d)
    for k in range(d):
        min_vals[k] = np.min(points[:, k])
        ptp_vals[k] = np.max(points[:, k]) - np.min(points[:, k])
    pts = np.empty_like(points)
    for i in range(n):
        for k in range(d):
            pts[i, k] = (points[i, k] - min_vals[k]) / (ptp_vals[k] + 1e-8)
    crowding = np.zeros((n, d))
    for dim in range(d):
        dim_sorted = np.argsort(pts[:, dim])
        point_sorted = pts[dim_sorted, dim]
        dist_0 = np.abs(point_sorted[0] - point_sorted[1]) if n >= 2 else 0.0
        dist_n = np.abs(point_sorted[n - 1] - point_sorted[n - 2]) if n >= 2 else 0.0
        crowding[dim_sorted[0], dim] = dist_0
        crowding[dim_sorted[n - 1], dim] = dist_n
        if n > 4:
            for i in range(1, n - 1):
                crowding[dim_sorted[i], dim] = np.abs(point_sorted[i] - point_sorted[i + 1])
    return np.sum(crowding, axis=1)


def crowding_distance(points):
    """端点特別処理を除去した混雑度計算（Numba JITで高速化）"""
    pts = np.ascontiguousarray(np.array(points, dtype=np.float64))
    return _crowding_distance_numba(pts)


# [PCN_REPLAY_REGIME_FAIR] uid 中のレジーム標識 ":r{scale}" (gpu_factory._mix_groups 由来)
_REGIME_UID_RE = re.compile(r":r([0-9]+(?:\.[0-9]+)?)(?::|$)")


def episode_regime_scale(transitions) -> float:
    """エピソードの到着スケール(レジーム標識)を返す。標識が取れないものは 1.0(基準)扱い。

    経路別の標識:
      - GPU工場(FactoryArrayEpisode): episodeオブジェクトの uid 末尾 ":r{scale}"
      - CPU actor(通常/種/heuristic全て): transitions[0]._pcn_arrival_scale (_run_episode で付与)
      - どちらも無い(単一レジーム学習・旧run由来など): 1.0
    """
    uid = getattr(transitions, "uid", None)  # FactoryArrayEpisode の配列エピソード
    if uid is None and len(transitions) > 0:
        first = transitions[0]
        scale = getattr(first, "_pcn_arrival_scale", None)
        if scale is not None:
            try:
                return float(scale)
            except (TypeError, ValueError):
                return 1.0
        uid = getattr(first, "_pcn_episode_uid", None)
    if isinstance(uid, str):
        m = _REGIME_UID_RE.search(uid)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return 1.0
    return 1.0


def episode_is_seed(transitions) -> bool:
    """NSGA種まき由来エピソードか（淘汰保護・種生存カウント用）。"""
    return bool(len(transitions) > 0 and getattr(transitions[0], "_pcn_seed_episode", False))


@dataclass
class Transition:
    """Transition dataclass."""

    observation: np.ndarray
    action: Union[int, int]
    reward: np.ndarray
    next_observation: np.ndarray
    terminal: bool

def _make_fourier_freqs(n_bands: int) -> th.Tensor:
    """Fourier 周波数列（_FOURIER_MODE 準拠）を n_bands 本生成。

    従来 __init__ 内の生成と同一演算（geometric/linear は式なので任意 n で prefix 整合。
    gaussian は seed 固定 randn なので同 seed から先頭 n 本＝長い列の prefix と一致）。"""
    if _FOURIER_MODE == "linear":
        # 線形倍音 (k+1)*base = 2π,4π,6π… の古典フーリエ級数。高周波が急増しないので小規模で崩壊しにくい。
        return th.tensor([float(k + 1) for k in range(n_bands)], dtype=th.float32) * _FOURIER_BASE
    elif _FOURIER_MODE == "gaussian":
        # RFF (Tancik 2020): |N(0,1)|*scale。scale が周波数帯域を支配。seed固定で全モデル同一周波数。
        _g = th.Generator().manual_seed(_FOURIER_SEED)
        return th.abs(th.randn(n_bands, generator=_g)) * _FOURIER_SCALE
    else:  # geometric (既定, NeRF式 2^k)
        return th.tensor([2.0 ** k for k in range(n_bands)], dtype=th.float32)


class BasePCNModel(nn.Module, ABC):
    """Base Model for the PCN."""

    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, scaling_factor: np.ndarray, hidden_dim: int):
        """Initialize the PCN model."""
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.scaling_factor = nn.Parameter(th.tensor(scaling_factor).float(), requires_grad=False)
        self.register_buffer("desired_return_center", th.zeros(reward_dim, dtype=th.float32))
        self.register_buffer("desired_return_scale", th.ones(reward_dim, dtype=th.float32))
        self.register_buffer("command_balance", th.ones(reward_dim, dtype=th.float32))
        self.hidden_dim = hidden_dim
        # b2: 条件線形層の入力次元。Fourier ON なら c[D] -> [c, sin(fk c), cos(fk c)] = D*(1+2L)。
        self._cmd_raw_dim = reward_dim + 1
        if _FOURIER_CMD:
            self.register_buffer("fourier_freqs", _make_fourier_freqs(_FOURIER_BANDS))
            if _FOURIER_BANDS_COST is not None and _FOURIER_BANDS_COST != _FOURIER_BANDS:
                # [per-channel bands] cost チャネルだけ別バンド数（同モードの先頭 L_cost 周波数）。
                # 入力次元 = 生c(D) + 非costチャネル(D-1)*2L + costチャネル 2L_cost。
                self.register_buffer(
                    "fourier_freqs_cost", _make_fourier_freqs(_FOURIER_BANDS_COST)
                )
                self.cmd_in_dim = (
                    self._cmd_raw_dim
                    + 2 * _FOURIER_BANDS * (self._cmd_raw_dim - 1)
                    + 2 * _FOURIER_BANDS_COST
                )
            else:
                # 未設定/同値なら従来と同一（buffer 構成・次元ともビット不変）。
                self.cmd_in_dim = self._cmd_raw_dim * (1 + 2 * _FOURIER_BANDS)
        else:
            self.cmd_in_dim = self._cmd_raw_dim
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def predict_archive_value(self, state, desired_return, desired_horizon, detach_repr=False):
        """(Cost, AvgWait) を Archive の objective_values スケールで予測。
        detach_repr=True で状態埋め込み s を detach（指令追従lossが方策の状態表現 s_emb を汚さない）。"""
        desired_return = th.clamp(desired_return, min=-1e12, max=1e12)
        if _ADAPTIVE_RETURN_NORMALIZATION:
            desired_return = (desired_return - self.desired_return_center) / self.desired_return_scale
        elif _DESIRED_RETURN_SCALE > 0:
            desired_return = desired_return / _DESIRED_RETURN_SCALE
        desired_return = self._balance_desired_return(desired_return)
        desired_return = _robust_cond_wait(desired_return)
        desired_horizon = th.clamp(desired_horizon, min=0.0, max=1e6)
        state = th.clamp(state.float(), min=-1e6, max=1e6)
        c = th.cat((desired_return, desired_horizon), dim=-1) * self.scaling_factor
        if th.isnan(c).any() or th.isinf(c).any():
            c = th.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        s = self.s_emb(state)
        if self.training and _S_EMB_DROPOUT > 0.0:
            s = F.dropout(s, p=min(max(_S_EMB_DROPOUT, 0.0), 0.95), training=True)
        if detach_repr:
            s = s.detach()  # [案1] 状態表現は読むだけ。方策の s_emb に勾配を流さない（既存value_reproの崩壊回避）。
        c = self.c_emb(self._encode_cmd(c))
        h = th.cat([s, c], dim=-1)
        out = self.value_head(h)
        scale = th.tensor(
            [_VALUE_COST_SCALE, _VALUE_WAIT_SCALE],
            device=out.device,
            dtype=out.dtype,
        )
        return out * scale

    def set_desired_return_normalization(self, center: np.ndarray, scale: np.ndarray) -> None:
        center_t = th.as_tensor(center, dtype=th.float32, device=self.desired_return_center.device)
        scale_t = th.as_tensor(scale, dtype=th.float32, device=self.desired_return_scale.device)
        scale_t = th.clamp(scale_t, min=_RETURN_NORM_MIN_SCALE)
        self.desired_return_center.copy_(center_t)
        self.desired_return_scale.copy_(scale_t)

    def set_command_balance(self, balance: np.ndarray) -> None:
        balance_t = th.as_tensor(balance, dtype=th.float32, device=self.command_balance.device)
        self.command_balance.copy_(balance_t)

    def _balance_desired_return(self, desired_return: th.Tensor) -> th.Tensor:
        if not _COMMAND_BALANCE:
            return desired_return
        # command_balance バッファに power は焼き込み済み(_command_balance_vector)。バッファ同期で actor にも伝わる。
        return desired_return * self.command_balance

    def _encode_cmd(self, c: th.Tensor) -> th.Tensor:
        """b2: Fourier command encoding。c[B,D] -> [c, sin(fk·c), cos(fk·c)] = [B, D*(1+2L)]。
        OFF時は c をそのまま返す。生 c を先頭に連結するので、学習で高周波を使う/捨てるを選べる。
        [per-channel bands] fourier_freqs_cost buffer がある時のみ cost チャネル(index1)だけ
        別周波数集合で展開（レイアウトは従来と同じ D-major: [c, d0ブロック, d1ブロック, ...]、
        各ブロック=[sin(L_d), cos(L_d)]。L_d のみチャネル別）。buffer 無し=従来コードと同一。"""
        if not _FOURIER_CMD:
            return c
        if getattr(self, "fourier_freqs_cost", None) is None:
            proj = c.unsqueeze(-1) * self.fourier_freqs            # [B, D, L]
            feats = th.cat([th.sin(proj), th.cos(proj)], dim=-1)   # [B, D, 2L]
            return th.cat([c, feats.flatten(start_dim=1)], dim=-1)  # [B, D*(1+2L)]
        parts = [c]
        for d in range(c.shape[-1]):
            ff = self.fourier_freqs_cost if d == _FOURIER_COST_CH else self.fourier_freqs
            proj = c[..., d : d + 1] * ff                          # [B, L_d]
            parts.append(th.cat([th.sin(proj), th.cos(proj)], dim=-1))
        return th.cat(parts, dim=-1)

    def forward(self, state, desired_return, desired_horizon):
        """Return log-probabilities of actions or return action directly in case of continuous action space."""
        # 入力値の検証とクリッピング（NaN/Infを防ぐ）
        #
        # NOTE:
        # このプロジェクトの desired_return（報酬）は桁が大きく（例: -1e5〜-1e6）なり得る。
        # ここで [-1000,1000] に強くクリップすると、異なるターゲットが同一値に潰れて
        # 条件付き方策が事実上「条件を無視」する挙動を引き起こす。
        # そのため、数値安全のためのクリップ幅は大きく取る（LogSoftmax 等で安定に扱える範囲）。
        desired_return = th.clamp(desired_return, min=-1e12, max=1e12)
        if _ADAPTIVE_RETURN_NORMALIZATION:
            desired_return = (desired_return - self.desired_return_center) / self.desired_return_scale
        elif _DESIRED_RETURN_SCALE > 0:
            desired_return = desired_return / _DESIRED_RETURN_SCALE
        desired_return = self._balance_desired_return(desired_return)
        desired_return = _robust_cond_wait(desired_return)
        desired_horizon = th.clamp(desired_horizon, min=0.0, max=1e6)
        state = th.clamp(state.float(), min=-1e6, max=1e6)
        if _OBS_LOG_COMPRESS:
            state = th.sign(state) * th.log1p(th.abs(state))

        c = th.cat((desired_return, desired_horizon), dim=-1)
        c = c * self.scaling_factor
        
        global _model_nan_warn_count

        def _warn_nan(tag: str, extra: str = "") -> None:
            global _model_nan_warn_count
            if _model_nan_warn_count >= _MODEL_NAN_WARN_LIMIT:
                return
            _model_nan_warn_count += 1
            print(f"[BasePCNModel] 警告: {tag}にNaN/Infが含まれています{extra}")

        if _FWD_NANCHECK:
            if th.isnan(c).any() or th.isinf(c).any():
                _warn_nan("条件ベクトルc")
                c = th.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # 高速パス: 無条件 nan_to_num（NaN/Inf 無しなら恒等＝bit 一致, .any() 同期を除去）
            c = th.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)

        s = self.s_emb(state)
        if self.training and _S_EMB_DROPOUT > 0.0:
            s = F.dropout(s, p=min(max(_S_EMB_DROPOUT, 0.0), 0.95), training=True)

        if _FWD_NANCHECK:
            if th.isnan(s).any() or th.isinf(s).any():
                _warn_nan("状態埋め込みs")
                s = th.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            s = th.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

        c = self._encode_cmd(c)  # b2: Fourier展開（OFF時は素通り）。FiLM/c_emb 両経路で共通。
        if _FILM and hasattr(self, "film_gamma"):
            # c はここでは Fourier展開後（scaling→encode 済み, c_emb前）。γ(c)=1+proj, β(c)=proj。
            gamma = 1.0 + self.film_gamma(c)
            beta = self.film_beta(c)
            gamma = th.nan_to_num(gamma, nan=1.0, posinf=1.0, neginf=1.0)
            beta = th.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
            prediction = self.fc(s * gamma + beta)
        else:
            c = self.c_emb(c)
            if _FWD_NANCHECK:
                if th.isnan(c).any() or th.isinf(c).any():
                    _warn_nan("条件埋め込みc")
                    c = th.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                c = th.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
            if _COND_ADD_SCALE > 0.0:
                prediction = self.fc(s * c + _COND_ADD_SCALE * c)
            else:
                prediction = self.fc(s * c)

        if _FWD_NANCHECK:
            if th.isnan(prediction).any() or th.isinf(prediction).any():
                _warn_nan(
                    "予測出力",
                    f" (s=[{float(s.min()):.3g},{float(s.max()):.3g}] "
                    f"c=[{float(c.min()):.3g},{float(c.max()):.3g}])",
                )
                prediction = th.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # 高速パス: 警告メッセージの s.min()/c.min() 自体が host 同期なので呼ばず無条件補正
            prediction = th.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)

        return prediction


class DiscreteActionsDefaultModel(BasePCNModel):
    """Model for the PCN with discrete actions."""

    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, scaling_factor: np.ndarray, hidden_dim: int):
        """Initialize the PCN model for discrete actions."""
        super().__init__(state_dim, action_dim, reward_dim, scaling_factor, hidden_dim)
        # TODO: 入力次元の最適化が必要
        # 現在: 205040次元 → 非常に重い
        # 提案: 特徴量選択や次元削減で1000-5000次元程度に削減
        # self.state_dim = 38440  # s← この値が性能ボトルネック
        self.state_dim = state_dim  # 観測サイズ（76840 for 512+2048 nodes, 38440 for 256+1024）
        self.s_emb = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim), nn.Sigmoid())
        self.c_emb = nn.Sequential(nn.Linear(self.cmd_in_dim, self.hidden_dim), nn.Sigmoid())
        if _FILM:
            # γ≈1, β≈0 で初期化（zero-init → 開始時 s*1+0 = s）。学習で条件変調を獲得。
            # b2: Fourier ON でも zero-init は不変なので安全スタート（s*1+0=s）はそのまま保たれる。
            self.film_gamma = nn.Linear(self.cmd_in_dim, self.hidden_dim)
            self.film_beta = nn.Linear(self.cmd_in_dim, self.hidden_dim)
            for _l in (self.film_gamma, self.film_beta):
                nn.init.zeros_(_l.weight); nn.init.zeros_(_l.bias)
        # fc の中間層数(容量アブレーション用)。既定2=従来(Linear+ReLU+Linear)とビット同一構造。
        _fc_depth = int(os.environ.get("PCN_FC_DEPTH", "2"))
        _fc_layers = []
        for _ in range(max(1, _fc_depth - 1)):
            _fc_layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
        _fc_layers += [nn.Linear(self.hidden_dim, self.action_dim), nn.LogSoftmax(dim=1)]
        self.fc = nn.Sequential(*_fc_layers)


class ContinuousActionsDefaultModel(BasePCNModel):
    """Model for the PCN with continuous actions."""

    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, scaling_factor: np.ndarray, hidden_dim: int):
        """Initialize the PCN model for continuous actions."""
        super().__init__(state_dim, action_dim, reward_dim, scaling_factor, hidden_dim)
        self.s_emb = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim), nn.Sigmoid())
        self.c_emb = nn.Sequential(nn.Linear(self.cmd_in_dim, self.hidden_dim), nn.Sigmoid())
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )


class _AttnStateEncoder(nn.Module):
    """PCN_ARCH=attn 用の状態エンコーダ（従来 s_emb の代替）。

    観測 flatten を分解してトークン列として自己注意で符号化する:
      state[:, :180]    -> イベント30トークン×6特徴（占有中ジョブ: 開始/終了/長さ/cloud/開始ノード/高さ）
      state[:, 180:220] -> ジョブキュー5トークン×8特徴（index0=現ジョブ）
      state[:, 220:]    -> グローバル特徴（urgency等, 次元は state_dim-220 で可変, 0 も可）
    flatten で失われる「現ジョブ×隙間」の関係構造を attention で復元するのが狙い。
    出力の値域整合は Sigmoid でなく LayerNorm で取る。
    BasePCNModel.forward が呼ぶ self.s_emb(state) と同じ [B, state_dim]->[B, hidden_dim] 契約。
    """

    N_EVENT_TOKENS = 30
    EVENT_FEAT = 6
    N_QUEUE_TOKENS = 5
    QUEUE_FEAT = 8
    MIN_STATE_DIM = N_EVENT_TOKENS * EVENT_FEAT + N_QUEUE_TOKENS * QUEUE_FEAT  # 220

    def __init__(self, state_dim: int, hidden_dim: int, d_model: int = 64, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.event_dim = self.N_EVENT_TOKENS * self.EVENT_FEAT   # 180
        self.queue_dim = self.N_QUEUE_TOKENS * self.QUEUE_FEAT   # 40
        self.global_dim = int(state_dim) - self.MIN_STATE_DIM    # 可変(weekA=1: urgency)
        assert self.global_dim >= 0, f"state_dim={state_dim} < {self.MIN_STATE_DIM} は AttnStateEncoder 不可"
        self.d_model = d_model
        self.event_proj = nn.Linear(self.EVENT_FEAT, d_model)
        self.queue_proj = nn.Linear(self.QUEUE_FEAT, d_model)
        # 種別埋め込み: 0=イベント, 1=現ジョブ(キューindex0), 2=キュー(index1..4)。現ジョブを識別。
        self.type_emb = nn.Parameter(th.zeros(3, d_model))
        nn.init.normal_(self.type_emb, std=0.02)
        # TransformerEncoder風(pre-LN + 残差, FFN=2d)を n_layers 積む
        self.layers = nn.ModuleList(
            nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                "ln2": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.ReLU(), nn.Linear(2 * d_model, d_model)),
            })
            for _ in range(n_layers)
        )
        # プーリング: 現ジョブトークン出力 + 全トークンmean の concat -> hidden_dim
        self.pool_proj = nn.Linear(2 * d_model, hidden_dim)
        # グローバル特徴(urgency等)を s_emb 相当ベクトルに concat して hidden_dim へ戻す
        self.global_fuse = nn.Linear(hidden_dim + self.global_dim, hidden_dim) if self.global_dim > 0 else None
        self.out_ln = nn.LayerNorm(hidden_dim)

    def reset_parameters(self):
        """reinit_network()(凍結run再試行)対応: reset_parameters を持たない部品をここで再初期化。
        子モジュール(Linear/LayerNorm)は呼び出し側の modules() 走査でも再初期化される(重複は無害)。"""
        nn.init.normal_(self.type_emb, std=0.02)
        for lyr in self.layers:
            if hasattr(lyr["attn"], "_reset_parameters"):
                lyr["attn"]._reset_parameters()

    def forward(self, state):
        state = state.reshape(-1, state.shape[-1])
        ev = state[:, : self.event_dim].reshape(-1, self.N_EVENT_TOKENS, self.EVENT_FEAT)
        qu = state[:, self.event_dim : self.event_dim + self.queue_dim].reshape(-1, self.N_QUEUE_TOKENS, self.QUEUE_FEAT)
        tok_ev = self.event_proj(ev) + self.type_emb[0]
        type_qu = th.cat(
            [self.type_emb[1:2], self.type_emb[2:3].expand(self.N_QUEUE_TOKENS - 1, -1)], dim=0
        )  # [5, d]: index0=現ジョブ種別, 1..4=キュー種別
        tok_qu = self.queue_proj(qu) + type_qu
        x = th.cat([tok_ev, tok_qu], dim=1)  # [B, 35, d]
        for lyr in self.layers:
            h = lyr["ln1"](x)
            a, _ = lyr["attn"](h, h, h, need_weights=False)
            x = x + a
            x = x + lyr["ffn"](lyr["ln2"](x))
        cur = x[:, self.N_EVENT_TOKENS, :]  # 現ジョブトークン(キューindex0 = トークン列の30番)
        pooled = th.cat([cur, x.mean(dim=1)], dim=-1)  # [B, 2d]
        h = self.pool_proj(pooled)
        if self.global_fuse is not None:
            g = state[:, self.event_dim + self.queue_dim :]
            h = self.global_fuse(th.cat([h, g], dim=-1))
        return self.out_ln(h)


class AttnActionsModel(BasePCNModel):
    """イベント集合attention版PCNモデル（オプトイン: PCN_ARCH=attn）。

    s_emb を _AttnStateEncoder（トークン化+自己注意, Sigmoid無し/LayerNorm整合）に置き換える以外は
    DiscreteActionsDefaultModel と同一構造（c_emb / FILM / Fourier / fc / LogSoftmax /
    desired_return 正規化バッファは BasePCNModel + 本クラスで同じに構築）。
    forward / predict_archive_value は BasePCNModel のものをそのまま使う（s_emb 契約が同じため）。
    """

    MIN_STATE_DIM = _AttnStateEncoder.MIN_STATE_DIM  # 220

    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, scaling_factor: np.ndarray, hidden_dim: int):
        super().__init__(state_dim, action_dim, reward_dim, scaling_factor, hidden_dim)
        self.state_dim = state_dim
        _d = int(os.environ.get("PCN_ATTN_DIM", "64"))
        _nl = int(os.environ.get("PCN_ATTN_LAYERS", "2"))
        _nh = int(os.environ.get("PCN_ATTN_HEADS", "4"))
        self.s_emb = _AttnStateEncoder(state_dim, self.hidden_dim, d_model=_d, n_layers=_nl, n_heads=_nh)
        self.c_emb = nn.Sequential(nn.Linear(self.cmd_in_dim, self.hidden_dim), nn.Sigmoid())
        if _FILM:
            # DiscreteActionsDefaultModel と同じ zero-init FiLM（開始時 s*1+0=s の安全スタート）
            self.film_gamma = nn.Linear(self.cmd_in_dim, self.hidden_dim)
            self.film_beta = nn.Linear(self.cmd_in_dim, self.hidden_dim)
            for _l in (self.film_gamma, self.film_beta):
                nn.init.zeros_(_l.weight); nn.init.zeros_(_l.bias)
        _fc_depth = int(os.environ.get("PCN_FC_DEPTH", "2"))
        _fc_layers = []
        for _ in range(max(1, _fc_depth - 1)):
            _fc_layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
        _fc_layers += [nn.Linear(self.hidden_dim, self.action_dim), nn.LogSoftmax(dim=1)]
        self.fc = nn.Sequential(*_fc_layers)


class EnhancedPCNModel(nn.Module):
    """スケジューリング環境のための拡張PCNモデル"""
    def __init__(self, 
                 observation_dim, 
                 n_premise_nodes,
                 n_cloud_nodes,
                 window_size,
                 job_feature_dim=40,
                 hidden_dim=256,
                 reward_dim=2,
                 action_dim=2,
                 debug_mode=True):
        super(EnhancedPCNModel, self).__init__()
        
        # 基本パラメータ初期化
        self.observation_dim = observation_dim
        self.n_premise_nodes = n_premise_nodes
        self.n_cloud_nodes = n_cloud_nodes
        self.window_size = window_size
        self.job_feature_dim = job_feature_dim
        self.hidden_dim = hidden_dim
        self.reward_dim = reward_dim
        self.action_dim = action_dim
        self.debug_mode = debug_mode
        self.register_buffer("desired_return_center", th.zeros(reward_dim, dtype=th.float32))
        self.register_buffer("desired_return_scale", th.ones(reward_dim, dtype=th.float32))
        self.register_buffer("command_balance", th.ones(reward_dim, dtype=th.float32))
        
        # デバッグ出力
        if self.debug_mode:
            print(f"==== モデル構築: 次元情報 ====")
            print(f"観測次元数: {observation_dim}")
            print(f"オンプレミスノード数: {n_premise_nodes}")
            print(f"クラウドノード数: {n_cloud_nodes}")
            print(f"ウィンドウサイズ: {window_size}")
            print(f"ジョブ特徴量次元: {job_feature_dim}")
            print(f"隠れ層次元: {hidden_dim}")
            print(f"報酬次元: {reward_dim}")
            print(f"行動次元: {action_dim}")
            print("============================")
        
        # 1. マップデータ用CNN処理部分（より多くの縮小を行う）
        self.map_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),  # ストライド2に変更
            nn.ReLU(),
            nn.MaxPool2d(2),  # 2x2のプーリングで更に縮小
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # ストライド2に変更
            nn.ReLU(),
            nn.MaxPool2d(2),  # さらに縮小
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # サイズを固定の4x4に
            nn.Flatten()
        )
        
        # CNNの出力サイズを計算（固定サイズに）
        self.cnn_output_dim = 64 * 4 * 4  # 固定サイズの出力
        
        if self.debug_mode:
            print(f"CNN出力次元: {self.cnn_output_dim}")
        
        # 2. ジョブキュー処理部分 - シンプルな線形層に変更
        self.job_embedding = nn.Linear(8, 32)  # 各ジョブは8次元
        self.job_encoder = nn.Sequential(
            nn.Linear(32 * 5, 64),  # 5ジョブ分を固定サイズに
            nn.ReLU()
        )
        
        # 3. 特徴結合層 - 次元を圧縮
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.cnn_output_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 4. PCN条件エンコーディング
        self.condition_encoder = nn.Sequential(
            nn.Linear(reward_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 5. ホライゾン処理
        self.horizon_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Sigmoid()
        )
        
        # 6. 統合と出力層
        self.pi_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.v_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, reward_dim)
        )
        
        # 7. 入力前処理メソッド
        self.preprocess = InputPreprocessor(
            n_premise_nodes=n_premise_nodes,
            n_cloud_nodes=n_cloud_nodes,
            window_size=window_size,
            debug_mode=debug_mode
        )

    def set_desired_return_normalization(self, center: np.ndarray, scale: np.ndarray) -> None:
        center_t = th.as_tensor(center, dtype=th.float32, device=self.desired_return_center.device)
        scale_t = th.as_tensor(scale, dtype=th.float32, device=self.desired_return_scale.device)
        scale_t = th.clamp(scale_t, min=_RETURN_NORM_MIN_SCALE)
        self.desired_return_center.copy_(center_t)
        self.desired_return_scale.copy_(scale_t)

    def set_command_balance(self, balance: np.ndarray) -> None:
        balance_t = th.as_tensor(balance, dtype=th.float32, device=self.command_balance.device)
        self.command_balance.copy_(balance_t)
    
    def extract_state_features(self, x):
        """観測データから状態特徴を抽出（シンプル化）"""
        # 入力データの分離と整形
        maps_data, job_data = self.preprocess(x)
        
        # マップデータのCNN処理
        map_features = self.map_cnn(maps_data)
        
        # ジョブデータの処理（シンプル化）
        batch_size, n_jobs, job_dim = job_data.shape
        job_emb = self.job_embedding(job_data)  # [B, n_jobs, 32]
        job_features = job_emb.view(batch_size, -1)  # フラット化
        job_features = self.job_encoder(job_features)
        
        # 特徴結合
        combined_features = th.cat([map_features, job_features], dim=1)
        state_features = self.feature_fusion(combined_features)
        
        return state_features
    
    def encode_condition(self, r, h=None):
        """報酬重みとホライゾンから条件を生成（シンプル化）"""
        # 入力値の検証とクリッピング（NaN/Infを防ぐ）
        r = th.clamp(r, min=-1000.0, max=1000.0)
        if h is not None:
            h = th.clamp(h, min=0.0, max=1000.0)
        
        condition = self.condition_encoder(r)
        
        # NaN/Infチェック
        if th.isnan(condition).any() or th.isinf(condition).any():
            if self.debug_mode:
                print(f"[EnhancedPCNModel] 警告: condition_encoder出力にNaN/Infが含まれています")
                print(f"  r範囲: min={r.min()}, max={r.max()}, mean={r.mean()}")
            condition = th.nan_to_num(condition, nan=0.0, posinf=0.0, neginf=0.0)
        
        if h is not None:
            horizon_cond = self.horizon_encoder(h)
            
            # NaN/Infチェック
            if th.isnan(horizon_cond).any() or th.isinf(horizon_cond).any():
                if self.debug_mode:
                    print(f"[EnhancedPCNModel] 警告: horizon_encoder出力にNaN/Infが含まれています")
                    print(f"  h範囲: min={h.min()}, max={h.max()}, mean={h.mean()}")
                horizon_cond = th.nan_to_num(horizon_cond, nan=0.0, posinf=0.0, neginf=0.0)
            
            condition = condition * horizon_cond
            
            # NaN/Infチェック
            if th.isnan(condition).any() or th.isinf(condition).any():
                if self.debug_mode:
                    print(f"[EnhancedPCNModel] 警告: condition * horizon_condにNaN/Infが含まれています")
                condition = th.nan_to_num(condition, nan=0.0, posinf=0.0, neginf=0.0)
        
        return condition
    
    def forward(self, x, r, h=None):
        """
        前方伝播処理
        Args:
            x: 観測データ
            r: 報酬重み
            h: ホライゾン（オプション）
        """
        if self.debug_mode:
            print(f"\n==== モデル前方伝播 ====")
            print(f"バッチ観測データ入力形状: {x.shape}")
            print(f"報酬重み入力形状: {r.shape}")
            if h is not None:
                print(f"ホライゾン入力形状: {h.shape}")
        
        # 入力値の検証とクリッピング（NaN/Infを防ぐ）
        # NOTE: desired_return が大きい桁になり得るため、条件が潰れないよう広めに取る。
        x = th.clamp(x, min=-1e6, max=1e6)
        r = th.clamp(r, min=-1e12, max=1e12)
        if _ADAPTIVE_RETURN_NORMALIZATION:
            r = (r - self.desired_return_center) / self.desired_return_scale
        elif _DESIRED_RETURN_SCALE > 0:
            r = r / _DESIRED_RETURN_SCALE
        if _COMMAND_BALANCE:
            r = r * self.command_balance
        if h is not None:
            h = th.clamp(h, min=0.0, max=1e6)
        
        # 状態特徴抽出
        state_features = self.extract_state_features(x)
        
        # NaN/Infチェック
        if th.isnan(state_features).any() or th.isinf(state_features).any():
            if self.debug_mode:
                print(f"[EnhancedPCNModel] 警告: 状態特徴にNaN/Infが含まれています")
            state_features = th.nan_to_num(state_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 条件エンコーディング
        condition = self.encode_condition(r, h)
        
        if self.debug_mode:
            print(f"状態特徴形状: {state_features.shape}")
            print(f"条件形状: {condition.shape}")
        
        # 条件付き予測
        # PCNのキーとなる部分: 状態特徴と条件の要素積
        conditioned_features = state_features * condition
        
        # NaN/Infチェック
        if th.isnan(conditioned_features).any() or th.isinf(conditioned_features).any():
            if self.debug_mode:
                print(f"[EnhancedPCNModel] 警告: 条件付き特徴にNaN/Infが含まれています")
                print(f"  state_features範囲: min={state_features.min()}, max={state_features.max()}, mean={state_features.mean()}")
                print(f"  condition範囲: min={condition.min()}, max={condition.max()}, mean={condition.mean()}")
            conditioned_features = th.nan_to_num(conditioned_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.debug_mode:
            print(f"条件付き特徴形状: {conditioned_features.shape}")
        
        # 方策と価値予測
        pi = self.pi_net(conditioned_features)
        v = self.v_net(conditioned_features)
        
        # NaN/Infチェック
        if th.isnan(pi).any() or th.isinf(pi).any():
            if self.debug_mode:
                print(f"[EnhancedPCNModel] 警告: π出力にNaN/Infが含まれています")
            pi = th.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
        
        if th.isnan(v).any() or th.isinf(v).any():
            if self.debug_mode:
                print(f"[EnhancedPCNModel] 警告: V出力にNaN/Infが含まれています")
            v = th.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.debug_mode:
            print(f"π出力形状: {pi.shape}")
            print(f"V出力形状: {v.shape}")
            print("============================")
        
        return pi, v


class InputPreprocessor(nn.Module):
    """生の観測データを処理して適切な形式に変換するプリプロセッサ"""
    def __init__(self, n_premise_nodes, n_cloud_nodes, window_size, debug_mode=True):
        super(InputPreprocessor, self).__init__()
        self.n_premise_nodes = n_premise_nodes
        self.n_cloud_nodes = n_cloud_nodes
        self.window_size = window_size
        self.debug_mode = debug_mode
        
    def forward(self, x):
        """
        観測データをマップデータとジョブデータに分離
        Returns:
            maps_data: [B, 2, max_nodes, window_size] - 2チャネル（オンプレとクラウド）
            job_data: [B, n_jobs, job_dim] - 各ジョブの特徴
        """
        batch_size = x.shape[0]
        
        if self.debug_mode:
            print(f"\n==== 入力前処理 ====")
            print(f"入力観測データ形状: {x.shape}")
        
        # マップデータのインデックス計算
        premise_size = self.n_premise_nodes * self.window_size
        cloud_size = self.n_cloud_nodes * self.window_size
        map_total_size = premise_size + cloud_size
        
        if self.debug_mode:
            print(f"オンプレミスマップサイズ: {premise_size}")
            print(f"クラウドマップサイズ: {cloud_size}")
            print(f"合計マップサイズ: {map_total_size}")
        
        # マップデータの抽出と整形
        map_data = x[:, :map_total_size]
        
        if self.debug_mode:
            print(f"抽出済みマップデータ形状: {map_data.shape}")
        
        premise_map = map_data[:, :premise_size].reshape(batch_size, self.n_premise_nodes, self.window_size)
        cloud_map = map_data[:, premise_size:map_total_size].reshape(batch_size, self.n_cloud_nodes, self.window_size)
        
        if self.debug_mode:
            print(f"整形後オンプレミスマップ形状: {premise_map.shape}")
            print(f"整形後クラウドマップ形状: {cloud_map.shape}")
        
        # 最大ノード数に合わせたパディング
        max_nodes = max(self.n_premise_nodes, self.n_cloud_nodes)
        
        if self.debug_mode:
            print(f"最大ノード数: {max_nodes}")
        
        if self.n_premise_nodes < max_nodes:
            padding = th.zeros(batch_size, max_nodes - self.n_premise_nodes, self.window_size, device=x.device)
            premise_map = th.cat([premise_map, padding], dim=1)
            
            if self.debug_mode:
                print(f"パディング後オンプレミスマップ形状: {premise_map.shape}")
        
        if self.n_cloud_nodes < max_nodes:
            padding = th.zeros(batch_size, max_nodes - self.n_cloud_nodes, self.window_size, device=x.device)
            cloud_map = th.cat([cloud_map, padding], dim=1)
            
            if self.debug_mode:
                print(f"パディング後クラウドマップ形状: {cloud_map.shape}")
        
        # 2チャネル形式に変換 [B, 2, max_nodes, window_size]
        maps_data = th.stack([premise_map, cloud_map], dim=1)
        
        if self.debug_mode:
            print(f"最終マップデータ形状: {maps_data.shape}")
        
        # ジョブデータの抽出
        job_data = x[:, map_total_size:]
        
        if self.debug_mode:
            print(f"抽出済みジョブデータ形状: {job_data.shape}")
            print(f"想定される残りサイズ: {x.shape[1] - map_total_size}")
        
        n_jobs = 5  # 固定値
        job_dim = 8  # 固定値
        
        try:
            job_data = job_data.reshape(batch_size, n_jobs, job_dim)
            
            if self.debug_mode:
                print(f"整形後ジョブデータ形状: {job_data.shape}")
        except:
            if self.debug_mode:
                print(f"エラー! 整形できません。入力サイズと期待形状の不一致:")
                print(f"現在のジョブデータサイズ: {job_data.shape}")
                print(f"期待される整形後サイズ: [{batch_size}, {n_jobs}, {job_dim}]")
                print(f"必要な要素数: {batch_size * n_jobs * job_dim}")
                print(f"実際の要素数: {job_data.numel()}")
        
        return maps_data, job_data


class PCN(MOAgent, MOPolicy):
    """Pareto Conditioned Networks (PCN).

    Reymond, M., Bargiacchi, E., & Nowé, A. (2022, May). Pareto Conditioned Networks.
    In Proceedings of the 21st International Conference on Autonomous Agents
    and Multiagent Systems (pp. 1110-1118).
    https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1110.pdf

    ## Credits

    This code is a refactor of the code from the authors of the paper, available at:
    https://github.com/mathieu-reymond/pareto-conditioned-networks
    """

    def __init__(
        self,
        env: Optional[gym.Env],
        scaling_factor: np.ndarray,
        device: Union[th.device, str],
        
        learning_rate: float,
        state_dim: int = 1,
        gamma: float = 1.0,
        batch_size: int = 1024,
        hidden_dim: int = 64,

        noise: float = 0.1,
        project_name: str = "temp",
        experiment_name: str = "PCN",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        seed: Optional[int] = None,

        model_class: Optional[Type[BasePCNModel]] = None,
        use_enhanced_model: bool = False,
        use_wandb: bool = True,
        log_episode_only: bool = True,
        debug_mode: bool = True,
    ) -> None:
        """Initialize PCN agent.

        Args:
            env (Optional[gym.Env]): Gym environment.
            scaling_factor (np.ndarray): Scaling factor for the desired return and horizon used in the model.
            learning_rate (float, optional): Learning rate. Defaults to 1e-2.
            gamma (float, optional): Discount factor. Defaults to 1.0.
            batch_size (int, optional): Batch size. Defaults to 32.
            hidden_dim (int, optional): Hidden dimension. Defaults to 64.
            noise (float, optional): Standard deviation of the noise to add to the action in the continuous action case. Defaults to 0.1.
            project_name (str, optional): Name of the project for wandb. Defaults to "MORL-Baselines".
            experiment_name (str, optional): Name of the experiment for wandb. Defaults to "PCN".
            wandb_entity (Optional[str], optional): Entity for wandb. Defaults to None.
            log (bool, optional): Whether to log to wandb. Defaults to True.
            seed (Optional[int], optional): Seed for reproducibility. Defaults to None.
            device (Union[th.device, str], optional): Device to use. Defaults to "auto".
            model_class (Optional[Type[BasePCNModel]], optional): Model class to use. Defaults to None.
            use_enhanced_model (bool, optional): Whether to use the enhanced model. Defaults to False.
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device)

        # 安全終了処理のためのフラグを初期化
        self.terminate_requested = False
        self.original_sigint = None
        self.original_sigterm = None

        # 既存の初期化コード
        self.reward_dim = env.reward_space.shape[0]
        # 観測次元: 分散Actorでは env がイベント生ベクトルでも NN はビットマップ次元で統一するため state_dim を優先
        if state_dim != 1:
            self.observation_dim = int(state_dim)
        else:
            self.observation_dim = self.env.observation_space.shape[0]
        self.continuous_action = isinstance(self.env.action_space, gym.spaces.Box)
        if self.continuous_action:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = self.env.action_space.n
        
        self.experience_replay = []
        self._frozen_pf_entries = []  # frozen-PF cloning: best-ever 非支配エピソード（劣化しない教師）
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.scaling_factor = scaling_factor
        self.noise = noise
        self.e_returns = []
        self.transitions = []
        self.mapmap = []
        
        self.use_enhanced_model = use_enhanced_model
        self.debug_mode = debug_mode
        
        # 混合精度学習の設定
        # 注意: AMP(fp16) は観測の桁が大きい場合(例: 1024ジョブで時刻特徴~2.8e6)に
        # fp16 上限(65504)を超えて s_emb が Inf/NaN 化し conditioning が死ぬ(policy_acc=0.5固定)。
        # obs が小さい 24 ジョブ等では問題ないため既定は有効のまま、PCN_USE_AMP=0 で無効化可能。
        self.use_amp = os.environ.get("PCN_USE_AMP", "1") == "1"
        self.scaler = th.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # パフォーマンス監視
        self.update_times = []
        self.last_update_time = None

        if use_enhanced_model:
            if self.debug_mode:
                print("拡張モデルを使用します。")
                print(f"オンプレミスノード数: {env.n_on_premise_node}")
                print(f"クラウドノード数: {env.n_cloud_node}")
                print(f"ウィンドウサイズ: {env.n_window}")
            
            self.network = EnhancedPCNModel(
                observation_dim=self.observation_dim,
                n_premise_nodes=env.n_on_premise_node,
                n_cloud_nodes=env.n_cloud_node,
                window_size=env.n_window,
                hidden_dim=self.hidden_dim,
                reward_dim=self.reward_dim,
                action_dim=self.action_dim,
                debug_mode=self.debug_mode
            ).to(self.device)
            
            self.target_network = EnhancedPCNModel(
                observation_dim=self.observation_dim,
                n_premise_nodes=env.n_on_premise_node,
                n_cloud_nodes=env.n_cloud_node,
                window_size=env.n_window,
                hidden_dim=self.hidden_dim,
                reward_dim=self.reward_dim,
                action_dim=self.action_dim,
                debug_mode=self.debug_mode
            ).to(self.device)
            
            self.target_network.load_state_dict(self.network.state_dict())
            if os.environ.get('DISTRIBUTED_PCN_USE_TORCH_COMPILE', '0') == '1':
                if hasattr(th, 'compile') and callable(getattr(th, 'compile', None)) and str(self.device) != 'cpu':
                    self.network = th.compile(self.network, mode='reduce-overhead')
            self.opt = th.optim.Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=_WEIGHT_DECAY)
        else:
            if model_class is None:
                if self.continuous_action:
                    model_class = ContinuousActionsDefaultModel
                else:
                    model_class = DiscreteActionsDefaultModel
                    # オプトイン: PCN_ARCH=attn でイベント集合attentionモデル(未設定=完全従来でビット不変)
                    if os.environ.get("PCN_ARCH", "") == "attn":
                        if self.observation_dim >= AttnActionsModel.MIN_STATE_DIM:
                            model_class = AttnActionsModel
                        else:
                            print(
                                f"[PCN] 警告: PCN_ARCH=attn だが state_dim={self.observation_dim} < "
                                f"{AttnActionsModel.MIN_STATE_DIM}(イベント180+キュー40)。旧環境の観測形式のため "
                                f"従来モデル(DiscreteActionsDefaultModel)にフォールバックします。"
                            )

            # [主犯特定インフラ] NN初期化seed固定。PCN_INIT_SEED 設定時のみ(空=従来=ランダム=ビット一致)。
            # 確率的崩壊(seed依存)を排除し、同一初期化のまま変更を1つずつ振って主犯を切り分けるため。
            _init_seed = os.environ.get("PCN_INIT_SEED", "")
            if _init_seed != "":
                th.manual_seed(int(_init_seed)); np.random.seed(int(_init_seed))
            self.model = model_class(
                self.observation_dim, self.action_dim, self.reward_dim, self.scaling_factor, hidden_dim=self.hidden_dim
            ).to(self.device, non_blocking=True)
            # torch.compile（デフォルト無効: このワークロードではオーバーヘッドが大きい）
            if os.environ.get('DISTRIBUTED_PCN_USE_TORCH_COMPILE', '0') == '1':
                if hasattr(th, 'compile') and callable(getattr(th, 'compile', None)) and str(self.device) != 'cpu':
                    self.model = th.compile(self.model, mode='reduce-overhead')
            self.opt = th.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=_WEIGHT_DECAY)

        self.log = log
        if log:
            experiment_name_to_log = experiment_name + (" continuous action" if self.continuous_action else "")
            self.setup_wandb(project_name, experiment_name_to_log, wandb_entity)

        self.evaluation_history = []
        self.evaluation_timestamps = []
        self.global_steps_at_evaluation = []
        self.wall_seconds_at_evaluation = []
        self.return_norm_center = np.zeros(self.reward_dim, dtype=np.float32)
        self.return_norm_scale = np.ones(self.reward_dim, dtype=np.float32)
        self._return_norm_initialized = False
        self._ema_shadow = None   # EMA重み dict[name->Tensor]。Phase3直前に reinit_ema で初期化
        self._ema_backup = None   # swap中の online 退避先
        self._pf_weight_mul = 1.0  # PF点学習重みのruntime倍率(段階的スケジュール用: 後半に膝へ集中)
        self._loss_ema = None      # 損失移動平均(スパイク検出用)
        self._nan_skip_total = 0   # step skip 累計(非有限勾配+損失スパイク)。警告上限と独立に常時カウント
        self._opt_step_total = 0   # optimizer.step 成功累計。skip率 = skip/(skip+step) で凍結検知に使う
        self._anchor_model = None  # anchor-KL の凍結教師(learn()境界で update_anchor_snapshot)

    def _policy_module(self):
        module = self.network if self.use_enhanced_model else self.model
        return getattr(module, "_orig_mod", module)

    def update_anchor_snapshot(self):
        """anchor-KL の教師を現在の方策スナップショットに更新。learn() 境界(イテレーション毎)に呼ぶ。
        anchor はイテレーション内のドリフトを抑える proximal 項であり、毎回追従するので学習は進む。"""
        if _ANCHOR_KL_WEIGHT <= 0.0:
            return False
        import copy as _copy
        with th.no_grad():
            self._anchor_model = _copy.deepcopy(self._policy_module()).eval()
            for p in self._anchor_model.parameters():
                p.requires_grad_(False)
        return True

    def reinit_network(self):
        """凍結run対策(初期NaN勾配ロックイン): ネットワーク重みとoptimizerを再初期化する。
        Phase2 epoch1損失が異常(正常0.3-5 vs 凍結44-50)な場合に呼ばれ、ハズレ初期値を引き直す。
        torch RNG は初期構築から進んでいるため別の初期値が出る。buffer(Fourier freqs/条件付け正規化)は
        意図的に保持(決定論的 or データ由来で重み初期値と無関係)。"""
        global _nan_skip_count
        module = self._policy_module()
        with th.no_grad():
            for m in module.modules():
                if hasattr(m, "reset_parameters") and callable(getattr(m, "reset_parameters")):
                    m.reset_parameters()
        if self.use_enhanced_model and hasattr(self, "target_network"):
            self.target_network.load_state_dict(self.network.state_dict())
        params = self.network.parameters() if self.use_enhanced_model else self.model.parameters()
        self.opt = th.optim.Adam(params, lr=self.learning_rate, weight_decay=_WEIGHT_DECAY)
        self.scaler = th.cuda.amp.GradScaler(enabled=self.use_amp)
        self._loss_ema = None
        self._ema_shadow = None
        self._ema_backup = None
        self._nan_skip_total = 0
        self._opt_step_total = 0
        _nan_skip_count = 0
        return True

    def reinit_ema(self):
        """EMA shadow を現在の(online)重みで初期化。Phase3突入直前に呼ぶ(Phase2学習済み重みから開始)。"""
        if not _EMA_ENABLED:
            return False
        with th.no_grad():
            mod = self._policy_module()
            self._ema_shadow = {n: p.detach().clone() for n, p in mod.named_parameters()}
        self._ema_backup = None
        return True

    def _apply_weight_maxnorm(self):
        """重みノルム天井: 各重み行列のノルムが基準(初回更新時)比 FACTOR 倍を超えたら縮める。
        壁1の重み膨張を物理的に止め、後半の一点collapseを防ぐ。step 成功直後に呼ぶ。"""
        f = _WEIGHT_MAXNORM_FACTOR
        if f <= 0:
            return
        mod = self._policy_module()
        base = getattr(self, "_weight_norm_base", None)
        if base is None:
            base = {}
            self._weight_norm_base = base
        with th.no_grad():
            for n, p in mod.named_parameters():
                if p.ndim < 2:  # 重み行列のみ(bias等1Dは対象外)
                    continue
                cur = float(p.norm())
                b = base.get(n)
                if b is None or b <= 0:
                    base[n] = cur  # 初回=基準として記録
                    continue
                cap = b * f
                if cur > cap:
                    p.mul_(cap / cur)

    def _ema_update(self):
        """optimizer.step 成功直後に呼ぶ: shadow = decay*shadow + (1-decay)*online (in-place, 追加メモリ0)。"""
        self._apply_weight_maxnorm()
        if not _EMA_ENABLED or self._ema_shadow is None:
            return
        with th.no_grad():
            mod = self._policy_module()
            for n, p in mod.named_parameters():
                s = self._ema_shadow.get(n)
                if s is None or s.shape != p.shape:
                    self._ema_shadow[n] = p.detach().clone()
                    continue
                s.mul_(_EMA_DECAY).add_(p.detach(), alpha=1.0 - _EMA_DECAY)

    def swap_in_ema_weights(self):
        """online重みを退避しEMA重みをモデルへ載せる(eval/save直前)。必ず restore と try/finally で対にする。"""
        if not _EMA_ENABLED or self._ema_shadow is None:
            return False
        with th.no_grad():
            mod = self._policy_module()
            self._ema_backup = {n: p.detach().clone() for n, p in mod.named_parameters()}
            for n, p in mod.named_parameters():
                s = self._ema_shadow.get(n)
                if s is not None and s.shape == p.shape:
                    p.data.copy_(s)
        return True

    def set_pf_weight_mul(self, m):
        """PF点学習重みのruntime倍率を設定(段階的スケジュール: Phase3後半で膝=希薄PF点に学習集中)。"""
        self._pf_weight_mul = float(m)

    def restore_online_weights(self):
        """swap_in_ema_weights で退避した online 重みを戻す(finally側で必ず呼ぶ)。"""
        if self._ema_backup is None:
            return
        with th.no_grad():
            mod = self._policy_module()
            for n, p in mod.named_parameters():
                b = self._ema_backup.get(n)
                if b is not None and b.shape == p.shape:
                    p.data.copy_(b)
        self._ema_backup = None

    def _apply_return_normalization_to_model(self) -> None:
        module = self._policy_module()
        if hasattr(module, "set_desired_return_normalization"):
            module.set_desired_return_normalization(self.return_norm_center, self.return_norm_scale)
        if hasattr(module, "set_command_balance"):
            module.set_command_balance(self._command_balance_vector())

    def _policy_n_jobs(self) -> int:
        env = getattr(self, "env", None)
        if env is None:
            return 1
        n_jobs = int(getattr(env, "n_jobs", 0) or 0)
        if n_jobs <= 0 and hasattr(env, "unwrapped"):
            n_jobs = int(getattr(env.unwrapped, "n_jobs", 0) or 0)
        if n_jobs <= 0:
            try:
                n_jobs = int(len(env.jobs))
            except Exception:
                n_jobs = 0
        if n_jobs <= 1:
            # [2026-08-27 バグ修正] Learner の env は n_jobs を持たない/1 を名乗るため、ここが 1 に
            # 退化していた。→ 注文の wait 成分(-avg_wait×nj)が「平均値のまま」になり、総待ちスケールの
            # 正規化(5万で~1.1e8)に対して実質0 = 全rollout注文のwait軸が無信号(v9/v10予行のbuffer実測で
            # 数値一致確認)。真値は構築時に明示で渡される scaling_factor[2]=1/n_jobs から導出する。
            try:
                sf = np.asarray(self.scaling_factor, dtype=np.float64).reshape(-1)
                if sf.size >= 3 and 0.0 < float(sf[2]) <= 1.0:
                    derived = int(round(1.0 / float(sf[2])))
                    if derived > n_jobs:
                        if not getattr(self, "_njobs_fallback_logged", False):
                            self._njobs_fallback_logged = True
                            print(f"[POLICY_NJOBS] env由来n_jobs={n_jobs} → "
                                  f"scaling_factorから{derived}を導出して使用 "
                                  "(旧挙動=1: 注文waitがほぼ0になるバグ)", flush=True)
                        n_jobs = derived
            except Exception:
                pass
        return max(1, n_jobs)

    @staticmethod
    def _objectives_to_desired_return(cost: float, avg_wait: float, n_jobs: int) -> np.ndarray:
        """均等 command 評価と同じスケール: r0=-total_wait, r1=-cost。"""
        nj = max(1, int(n_jobs))
        return np.array([-float(avg_wait) * nj, -float(cost)], dtype=np.float32)

    def _archive_objective_points(self, entries=None) -> np.ndarray:
        if entries is None:
            entries = self._valid_replay_entries()
        points = []
        for entry in entries:
            episode = entry[2]
            if not episode:
                continue
            first = episode[0]
            if hasattr(first, "objective_values") and first.objective_values is not None:
                obj = first.objective_values
                points.append([float(obj[0]), float(obj[2])])
        if not points:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(points, dtype=np.float64)

    @staticmethod
    def _cost_frac_band_limits(points: np.ndarray, min_frac: float, max_frac: float) -> tuple:
        if points.size == 0:
            return 0.0, 0.0
        cmax = float(np.max(points[:, 0]))
        if cmax <= 0:
            return 0.0, 0.0
        return min_frac * cmax, max_frac * cmax

    def _mid_cost_band_limits(self, points: np.ndarray) -> tuple:
        return self._cost_frac_band_limits(
            points, _TRAIN_MID_COST_MIN_FRAC, _TRAIN_MID_COST_MAX_FRAC
        )

    def _knee_cost_band_limits(self, points: np.ndarray) -> tuple:
        return self._cost_frac_band_limits(
            points, _TRAIN_KNEE_COST_MIN_FRAC, _TRAIN_KNEE_COST_MAX_FRAC
        )

    def _low_slope_cost_band_limits(self, points: np.ndarray) -> tuple:
        return self._cost_frac_band_limits(
            points, _TRAIN_LOW_SLOPE_COST_MIN_FRAC, _TRAIN_LOW_SLOPE_COST_MAX_FRAC
        )

    def _collect_archive_commands_in_cost_frac(
        self, min_frac: float, max_frac: float
    ) -> np.ndarray:
        """Archive PF の指定 cost 帯を eval スケールの desired_return に変換。

        update_many 中は experience_replay が凍結されているので、(min_frac,max_frac) 毎の
        結果（Archive PF 抽出は replay のみの関数）を memo して再計算を避ける（結果ビット一致）。
        戻り値は呼び出し側が必ず .copy()/fancy-index してから書き換えるので共有しても安全。
        """
        cache = getattr(self, "_cond_pool_cache", None)
        key = None
        if cache is not None:
            key = (round(float(min_frac), 9), round(float(max_frac), 9))
            hit = cache.get(key)
            if hit is not None:
                return hit
        entries = self._valid_replay_entries()
        points = self._archive_objective_points(entries)
        if points.size == 0:
            res = np.empty((0, 2), dtype=np.float32)
        else:
            cost_lo, cost_hi = self._cost_frac_band_limits(points, min_frac, max_frac)
            pf_i = get_non_dominated_inds_minimize(points) if cost_hi > cost_lo else []
            if cost_hi <= cost_lo or len(pf_i) == 0:
                res = np.empty((0, 2), dtype=np.float32)
            else:
                pf = points[pf_i]
                band = pf[(pf[:, 0] >= cost_lo) & (pf[:, 0] <= cost_hi)]
                if band.size == 0:
                    res = np.empty((0, 2), dtype=np.float32)
                else:
                    nj = self._policy_n_jobs()
                    cmds = [self._objectives_to_desired_return(c, w, nj) for c, w in band]
                    res = np.asarray(cmds, dtype=np.float32)
        if key is not None:
            cache[key] = res
        return res

    def _command_cost_in_mid_band(self, cost_command: float, cost_lo: float, cost_hi: float) -> bool:
        """desired_return[1] は -cost 累積なので、中域 cost は r1 が [-hi, -lo]。"""
        if cost_hi <= cost_lo:
            return False
        return (-cost_hi) <= cost_command <= (-cost_lo)

    def _episode_cost_cmax(self, valid_entries) -> float:
        costs = []
        for entry in valid_entries:
            episode = entry[2]
            if not episode:
                continue
            first = episode[0]
            if hasattr(first, "objective_values") and first.objective_values is not None:
                costs.append(float(first.objective_values[0]))
        return float(np.max(costs)) if costs else 0.0

    def _flat_step_band_sets(self, valid_entries):
        """ステップ重み用の cost 帯と ep_mid / ep_knee（valid_entries のグローバル index）。"""
        ep_costs = []
        for ep_i, entry in enumerate(valid_entries):
            episode = entry[2]
            if not episode:
                continue
            first = episode[0]
            if hasattr(first, "objective_values") and first.objective_values is not None:
                obj = first.objective_values
                ep_costs.append((ep_i, float(obj[0])))
        costs_only = np.array([c for _, c in ep_costs], dtype=np.float64) if ep_costs else np.empty(0)
        ref_pts = costs_only.reshape(-1, 1) if costs_only.size else np.empty((0, 2))
        cost_lo, cost_hi = self._mid_cost_band_limits(ref_pts)
        knee_lo, knee_hi = self._knee_cost_band_limits(ref_pts)
        low_lo, low_hi = self._low_slope_cost_band_limits(ref_pts)
        ep_mid = set()
        if cost_hi > cost_lo:
            for ep_i, c in ep_costs:
                if cost_lo <= c <= cost_hi:
                    ep_mid.add(ep_i)
        ep_knee = set()
        if knee_hi > knee_lo:
            for ep_i, c in ep_costs:
                if knee_lo <= c <= knee_hi:
                    ep_knee.add(ep_i)
        return cost_lo, cost_hi, knee_lo, knee_hi, low_lo, low_hi, ep_mid, ep_knee

    def _episode_flat_step_weight(
        self,
        block: Dict[str, np.ndarray],
        ep_i: int,
        cost_lo: float,
        cost_hi: float,
        knee_lo: float,
        knee_hi: float,
        low_lo: float,
        low_hi: float,
        ep_mid: set,
        ep_knee: set,
    ) -> np.ndarray:
        ep_len = int(block["episode_length"])
        dr = block["desired_returns"]
        w = np.ones(ep_len, dtype=np.float64)
        # [PCN_TRAIN_HEAD_STEP_WEIGHT] 全エピソードの先頭区間(=評価で使う G_0 に近い条件)を重くする。
        if _TRAIN_HEAD_STEP_WEIGHT > 1.0:
            n_head = max(1, int(np.ceil(_TRAIN_HEAD_STEP_FRAC * ep_len)))
            w[:n_head] = np.maximum(w[:n_head], _TRAIN_HEAD_STEP_WEIGHT)
        if _TRAIN_EVALIKE_STEP_WEIGHT > 1.0 and ep_i in ep_mid:
            n_early = max(1, int(np.ceil(_TRAIN_EVALIKE_STEP_FRAC * ep_len)))
            w[:n_early] = np.maximum(w[:n_early], _TRAIN_EVALIKE_STEP_WEIGHT)
        if _TRAIN_MID_STEP_WEIGHT > 1.0 and cost_hi > cost_lo:
            for t in range(ep_len):
                if self._command_cost_in_mid_band(float(dr[t, 1]), cost_lo, cost_hi):
                    w[t] = np.maximum(w[t], _TRAIN_MID_STEP_WEIGHT)
        if _TRAIN_KNEE_STEP_WEIGHT > 1.0 and knee_hi > knee_lo:
            for t in range(ep_len):
                if self._command_cost_in_mid_band(float(dr[t, 1]), knee_lo, knee_hi):
                    w[t] = np.maximum(w[t], _TRAIN_KNEE_STEP_WEIGHT)
        if _TRAIN_LOW_SLOPE_STEP_WEIGHT > 1.0 and low_hi > low_lo:
            for t in range(ep_len):
                if self._command_cost_in_mid_band(float(dr[t, 1]), low_lo, low_hi):
                    w[t] = np.maximum(w[t], _TRAIN_LOW_SLOPE_STEP_WEIGHT)
        if _TRAIN_GIANT_STEP_WEIGHT > 1.0 and ep_len > 2:
            # 各ステップの cost 報酬 = -(dr[t+1,1]-dr[t,1])。巨大ジョブをクラウドに置いた決定ほど大きい。
            # エピソード内で |cost報酬| 上位 frac のステップ(=巨大ジョブの高レバレッジ決定)へ重みを集中。
            r1 = dr[:, 1].astype(np.float64)
            cost_step = np.abs(np.diff(r1, append=r1[-1]))
            k = max(1, int(np.ceil(_TRAIN_GIANT_FRAC * ep_len)))
            if np.any(cost_step > 0):
                thr = np.partition(cost_step, ep_len - k)[ep_len - k]
                giant = cost_step >= max(thr, 1e-12)
                w[giant] = np.maximum(w[giant], _TRAIN_GIANT_STEP_WEIGHT)
        return self._apply_eval_gap_step_boost(w, dr)

    def _apply_eval_gap_step_boost(self, w: np.ndarray, dr: np.ndarray) -> np.ndarray:
        """Eval で検出した弱点 cost 帯のステップ replay 重みを増幅。"""
        bands = getattr(self, "_eval_gap_band_boosts", None)
        if not bands:
            return w
        for t in range(int(w.shape[0])):
            r1 = float(dr[t, 1])
            for cost_lo, cost_hi, mult in bands:
                if self._command_cost_in_mid_band(r1, cost_lo, cost_hi):
                    w[t] *= mult
                    break
        return w

    def set_eval_gap_band_boosts(self, boosts: Optional[List]) -> None:
        """boosts: [(cost_lo, cost_hi, mult), ...] or None で無効化。"""
        if boosts:
            self._eval_gap_band_boosts = [
                (float(lo), float(hi), float(m)) for lo, hi, m in boosts
            ]
        else:
            self._eval_gap_band_boosts = None

    def _training_flat_step_weights(self, valid_entries, blocks) -> Optional[np.ndarray]:
        """エピソード内ステップの replay 重み（中域 command / eval 相当序盤を厚く）。"""
        if (
            _TRAIN_MID_STEP_WEIGHT <= 1.0
            and _TRAIN_EVALIKE_STEP_WEIGHT <= 1.0
            and _TRAIN_KNEE_STEP_WEIGHT <= 1.0
            and _TRAIN_LOW_SLOPE_STEP_WEIGHT <= 1.0
            and _TRAIN_GIANT_STEP_WEIGHT <= 1.0
        ):
            return None
        cost_lo, cost_hi, knee_lo, knee_hi, low_lo, low_hi, ep_mid, ep_knee = (
            self._flat_step_band_sets(valid_entries)
        )
        flat_weights = [
            self._episode_flat_step_weight(
                block, ep_i, cost_lo, cost_hi, knee_lo, knee_hi, low_lo, low_hi, ep_mid, ep_knee
            )
            for ep_i, block in enumerate(blocks)
        ]
        if not flat_weights:
            return None
        flat = np.concatenate(flat_weights, axis=0)
        total = float(flat.sum())
        if total <= 0 or not np.isfinite(total):
            return None
        return flat

    def _collect_mid_band_archive_commands(self) -> np.ndarray:
        """Archive PF の中域点を均等 eval と同スケールの desired_return に変換。"""
        if _MID_BAND_COND_FOCUS_FRAC <= 0:
            return self._collect_archive_commands_in_cost_frac(
                _TRAIN_MID_COST_MIN_FRAC, _TRAIN_MID_COST_MAX_FRAC
            )
        half = max(_MID_BAND_COND_FOCUS_HALF_WIDTH_FRAC, 1e-6)
        lo_f = max(0.0, _MID_BAND_COND_FOCUS_FRAC - half)
        hi_f = _MID_BAND_COND_FOCUS_FRAC + half
        return self._collect_archive_commands_in_cost_frac(lo_f, hi_f)

    def _collect_low_slope_archive_commands(self) -> np.ndarray:
        """Archive PF の低 cost 帯（左上先端の下降部）。"""
        return self._collect_archive_commands_in_cost_frac(
            _TRAIN_LOW_SLOPE_COST_MIN_FRAC, _TRAIN_LOW_SLOPE_COST_MAX_FRAC
        )

    def _command_balance_vector(self) -> np.ndarray:
        if not _COMMAND_BALANCE:
            return np.ones(self.reward_dim, dtype=np.float32)
        scale = np.maximum(self.return_norm_scale.astype(np.float64), _RETURN_NORM_MIN_SCALE)
        geo = float(np.sqrt(scale[0] * scale[1]))
        balance = geo / scale
        # power を焼き込む(<1で緩め, 適応時は _bal_power_cur が動的に変わる)。geo/scale=full=power1。
        p = float(getattr(self, "_bal_power_cur", _COMMAND_BALANCE_POWER))
        if p != 1.0:
            balance = np.maximum(balance, 1e-6) ** p
        return balance.astype(np.float32)

    @staticmethod
    def _normalize_points_for_selection(points: np.ndarray) -> np.ndarray:
        arr = np.asarray(points, dtype=np.float64)
        if arr.size == 0:
            return arr
        center = np.percentile(arr, 50, axis=0)
        lo = np.percentile(arr, 5, axis=0)
        hi = np.percentile(arr, 95, axis=0)
        scale = hi - lo
        std = np.std(arr, axis=0)
        scale = np.where(scale > _RETURN_NORM_MIN_SCALE, scale, std)
        scale = np.where(scale > _RETURN_NORM_MIN_SCALE, scale, 1.0)
        return (arr - center) / scale

    def update_desired_return_normalization(self, entries=None) -> None:
        """Update command normalization from observed archive returns only.

        PCN原理に合わせ「中心化なし・目的ごとのスケールのみ」で正規化する。
        中心(median)を引くと R<-R-r の加法構造が壊れ、エピソード途中で command が
        正規化中心(=最頻 command)付近に落ち込み、方策が中央値挙動へ潰れる。
        そこで center=0 固定、scale=目的ごとの到達レンジ(robust max-abs)とし、各目的の
        command を約[-1,0]へ写す。scale はデータ由来なのでジョブ数に自動追従する。
        """
        if not _ADAPTIVE_RETURN_NORMALIZATION:
            return
        if entries is None:
            entries = self.experience_replay
        returns = [
            np.asarray(entry[2][0].reward, dtype=np.float64)
            for entry in entries
            if len(entry[2]) > 0
        ]
        if not returns:
            return
        arr = np.asarray(returns, dtype=np.float64)
        # 中心化はしない（加法構造を保つ）。各目的の到達レンジでスケールのみ。
        center = np.zeros(arr.shape[1], dtype=np.float64)
        scale = np.percentile(np.abs(arr), 99.5, axis=0)
        std = np.std(arr, axis=0)
        scale = np.where(scale > _RETURN_NORM_MIN_SCALE, scale, std)
        scale = np.where(scale > _RETURN_NORM_MIN_SCALE, scale, 1.0)
        center = center.astype(np.float32)
        scale = np.maximum(scale.astype(np.float32), np.float32(_RETURN_NORM_MIN_SCALE))
        if self._return_norm_initialized:
            ema = float(np.clip(_RETURN_NORM_EMA, 0.0, 1.0))
            self.return_norm_center = (1.0 - ema) * self.return_norm_center + ema * center
            self.return_norm_scale = (1.0 - ema) * self.return_norm_scale + ema * scale
        else:
            self.return_norm_center = center
            self.return_norm_scale = scale
            self._return_norm_initialized = True
        self._apply_return_normalization_to_model()

    def register_signal_handlers(self):
        """シグナルハンドラを登録する"""
        # 元のシグナルハンドラを保存
        self.original_sigint = signal.getsignal(signal.SIGINT)
        self.original_sigterm = signal.getsignal(signal.SIGTERM)
        
        def graceful_shutdown_handler(sig, frame):
            print("\n\n中断シグナルを受信しました。安全に終了処理を実行します...")
            self.terminate_requested = True
            # ここでは即終了せず、トレーニングループが終了確認するのを待つ
        
        # シグナルハンドラを設定
        signal.signal(signal.SIGINT, graceful_shutdown_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, graceful_shutdown_handler)  # killコマンド
    
    def restore_signal_handlers(self):
        """元のシグナルハンドラを復元する"""
        if self.original_sigint:
            signal.signal(signal.SIGINT, self.original_sigint)
        if self.original_sigterm:
            signal.signal(signal.SIGTERM, self.original_sigterm)
    
    def save_results_on_termination(self, eval_env, max_return, num_points_pf=200):
        """終了時に結果を保存するメソッド"""
        try:
            print("現在の学習状態を保存しています...")
            
            # 特別なフラグ付きでモデルを保存
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.save(filename=f"PCN_model_interrupted_{timestamp}")
            
            # 最終評価を実行
            print("最終評価を実行しています...")
            self.e_returns, _, _, self.mapmap = self.evaluate(
                eval_env, max_return, n=num_points_pf, save_history=True
            )
            
            # 評価結果の可視化
            self.visualize_evaluation_history(save_dir=f"interrupted_results_{timestamp}")
            
            # パレート解の保存
            self.save_pareto_solutions_to_txt(mode_name=f"interrupted_{timestamp}")
            
            print("\n終了処理が完了しました。結果は以下に保存されました：")
            print(f"- モデル: weights/PCN_model_interrupted_{timestamp}.pt")
            print(f"- 評価結果: interrupted_results_{timestamp}/")
            print(f"- パレート解: pareto_solutions/pareto_solutions_interrupted_{timestamp}_*.txt")
        except Exception as e:
            print(f"終了処理中にエラーが発生しました: {e}")
            traceback.print_exc()

    def get_config(self) -> dict:
        """Get configuration of PCN model."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "hidden_dim": self.hidden_dim,
            "scaling_factor": self.scaling_factor,
            "continuous_action": self.continuous_action,
            "noise": self.noise,
            "seed": self.seed,
        }

    @staticmethod
    def _build_padded_step_cdf(
        episode_lengths: np.ndarray,
        episode_offsets: np.ndarray,
        flat_step_probs: np.ndarray,
    ) -> np.ndarray:
        """エピソード内ステップ重みの CDF（逆変換サンプリング用、get_training_batch と同分布）。"""
        n_ep = int(len(episode_lengths))
        max_len = int(episode_lengths.max()) if n_ep else 0
        if max_len <= 0:
            return np.zeros((0, 0), dtype=np.float32)
        padded = np.zeros((n_ep, max_len), dtype=np.float32)
        for ep_i in range(n_ep):
            ln = int(episode_lengths[ep_i])
            if ln <= 0:
                continue
            off = int(episode_offsets[ep_i])
            seg = flat_step_probs[off : off + ln]
            s = float(seg.sum())
            if s > 0.0 and np.isfinite(s):
                padded[ep_i, :ln] = np.cumsum(seg / s, dtype=np.float32)
            else:
                padded[ep_i, :ln] = np.linspace(1.0 / ln, 1.0, ln, dtype=np.float32)
        return padded

    def _sample_training_flat_indices(self, cache: dict, batch_size: int) -> np.ndarray:
        """教師 cache から transition フラット index を一括サンプル（RNG 呼び出し順は従来と同じ）。"""
        lengths = cache["episode_lengths"]
        episode_probs = cache.get("episode_probs")
        if episode_probs is None:
            episode_indices = self.np_random.integers(0, len(lengths), size=batch_size)
        else:
            episode_indices = self.np_random.choice(
                len(lengths),
                size=batch_size,
                replace=True,
                p=episode_probs,
            )
        padded_cdf = cache.get("padded_step_cdf")
        if padded_cdf is not None and padded_cdf.size:
            ln = lengths[episode_indices]
            u = self.np_random.random(batch_size)
            # 旧実装は rows=padded_cdf[episode_indices] ([B,max_len] のコピー＋全要素比較) で
            # O(B×max_ep_len)。1万ジョブ級ではこれが update 時間の8割を占めた(512×9071≈37MB/更新)。
            # 同じ量 step = count(cdf_row[:ln] < u) − 1 を行ごとの searchsorted(side='left'=「<u の個数」)
            # で求める。結果は整数単位で同一・乱数消費も同一=ビット一致、O(B×log ep_len)。
            step_indices = np.empty(batch_size, dtype=np.int64)
            for bi in range(batch_size):
                ep_i = int(episode_indices[bi])
                n_i = int(ln[bi])
                step_indices[bi] = np.searchsorted(padded_cdf[ep_i, :n_i], u[bi], side="left") - 1
            step_indices = np.clip(step_indices, 0, np.maximum(ln - 1, 0))
        elif cache.get("flat_step_probs") is not None:
            offsets = cache["episode_offsets"]
            flat_step_probs = cache["flat_step_probs"]
            step_indices = np.empty(batch_size, dtype=np.int64)
            for bi in range(batch_size):
                ep_i = int(episode_indices[bi])
                off = int(offsets[ep_i])
                ep_len = int(lengths[ep_i])
                sp = flat_step_probs[off : off + ep_len]
                sp_sum = float(sp.sum())
                if sp_sum > 0 and np.isfinite(sp_sum):
                    step_indices[bi] = int(
                        self.np_random.choice(ep_len, p=sp / sp_sum)
                    )
                else:
                    step_indices[bi] = int(self.np_random.integers(0, ep_len))
        else:
            step_indices = (self.np_random.random(batch_size) * lengths[episode_indices]).astype(
                np.int64
            )
        return cache["episode_offsets"][episode_indices] + step_indices

    def get_training_batch(self):
        """学習用バッチをサンプリングして返す（JAX等の外部学習用）。
        Returns: (observations, actions, desired_returns, desired_horizons) の numpy 配列"""
        batch_size = self.batch_size
        cache = getattr(self, "_training_batch_cache", None)
        if cache is not None:
            flat_indices = self._sample_training_flat_indices(cache, batch_size)
            if cache.get("on_device", False):
                flat_indices_t = th.as_tensor(flat_indices, dtype=th.long, device=self.device)
                return (
                    cache["observations"].index_select(0, flat_indices_t),
                    cache["actions"].index_select(0, flat_indices_t),
                    cache["desired_returns"].index_select(0, flat_indices_t),
                    cache["desired_horizons"].index_select(0, flat_indices_t),
                )
            return (
                cache["observations"][flat_indices],
                cache["actions"][flat_indices],
                cache["desired_returns"][flat_indices],
                cache["desired_horizons"][flat_indices],
            )

        buffer_size = len(self.experience_replay)
        sample_indices = self.np_random.choice(buffer_size, size=batch_size, replace=True)
        
        # 3. 事前に配列を確保（メモリアロケーション削減）
        obs_shape = self.experience_replay[0][2][0].observation.shape
        reward_shape = self.experience_replay[0][2][0].reward.shape
        
        # 4. 効率的なデータ抽出（メモリ効率化）
        observations = np.empty((batch_size,) + obs_shape, dtype=np.float32)
        actions = np.empty(batch_size, dtype=np.int64)
        desired_returns = np.empty((batch_size,) + reward_shape, dtype=np.float32)
        desired_horizons = np.empty(batch_size, dtype=np.float32)
        
        # 5. ベクトル化されたデータ抽出（PCNの正しい学習方法）
        for i, idx in enumerate(sample_indices):
            episode = self.experience_replay[idx][2]
            episode_length = len(episode)
            
            # エピソード内のランダムなステップを選択（各ステップで学習）
            t = self.np_random.integers(0, episode_length)
            transition = episode[t]
            
            # コピーを避けて直接代入
            obs_data = transition.observation
            # NaN/Infチェック
            if np.any(np.isnan(obs_data)) or np.any(np.isinf(obs_data)):
                if self.debug_mode:
                    print(f"[PCN] 警告: エピソード {idx}, ステップ {t} の観測にNaN/Infが含まれています")
                    print(f"  観測範囲: min={np.min(obs_data)}, max={np.max(obs_data)}")
                # NaN/Infの場合は0に置き換え
                obs_data = np.nan_to_num(obs_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            observations[i] = obs_data
            actions[i] = transition.action
            
            if _LABEL_G:
                # [PCN_LABEL_G] 格納値は _add_episode で既に累積化済み（episode[t].reward = G_t）。
                # 再累積せずそのままラベルに使う（元論文実装と同じ・実行時の G 規約と整合）。
                remaining_return = np.nan_to_num(
                    np.asarray(episode[t].reward, dtype=np.float32),
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
            else:
                # 論文に厳密に従った累積報酬計算: R_t = Σ_{i=t}^T γ^i r_i（ベクトル化）
                # NOTE: 既知バグ（既定挙動として温存）: episode[j].reward は格納時に累積化済み
                # (=G_j) なので、これは二重累積 D_t = Σ_j γ^{j-t} G_j になる。修正は PCN_LABEL_G=1。
                n_remaining = episode_length - t
                rewards_slice = np.array([episode[j].reward for j in range(t, episode_length)], dtype=np.float32)
                rewards_slice = np.nan_to_num(rewards_slice, nan=0.0, posinf=0.0, neginf=0.0)
                discounts = np.power(self.gamma, np.arange(n_remaining, dtype=np.float32))
                remaining_return = np.dot(discounts, rewards_slice)
            
            # NaN/Infチェックと値のクリッピング
            if np.any(np.isnan(remaining_return)) or np.any(np.isinf(remaining_return)):
                if self.debug_mode:
                    print(f"[PCN] 警告: 累積報酬にNaN/Infが含まれています")
                    print(f"  累積報酬: {remaining_return}")
                # NaN/Infの場合は0に置き換え
                remaining_return = np.nan_to_num(remaining_return, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 値の範囲をクリッピング（異常に大きい値を防ぐ）
            # 累積報酬が異常に大きい場合、モデルの入力が不安定になる可能性がある
            # 注意: scaling_factorが[1, 1, 1]の場合、desired_returnが大きすぎると
            # モデルの内部で数値的不安定性が発生する可能性がある
            # より小さな範囲にクリッピング（-1000から1000の範囲）
            clip_value = _DESIRED_RETURN_CLIP
            if clip_value > 0:
                remaining_return = np.clip(remaining_return, -clip_value, clip_value)
            
            if self.debug_mode and (np.abs(remaining_return).max() > 500):
                print(f"[PCN] 警告: 累積報酬が大きすぎます: {remaining_return}")
                print(f"  エピソード長: {episode_length}, 開始ステップ: {t}")
                print(f"  クリッピング後の値: {np.clip(remaining_return, -clip_value, clip_value)}")
            
            desired_returns[i] = remaining_return  # 論文通りの割引累積報酬
            desired_horizons[i] = np.float32(episode_length - t)  # 残りのステップ数
        
        # 6. 一括GPU転送（非同期化 + メモリ効率化）
        # desired_returnの値が異常に大きい場合、モデルの入力が不安定になる可能性があるため、
        # ここでもクリッピングを確認
        # 注意: scaling_factorが[1, 1, 1]の場合、desired_returnが大きすぎると
        # モデルの内部で数値的不安定性が発生する可能性がある
        # より小さな範囲にクリッピング（-1000から1000の範囲）
        if _DESIRED_RETURN_CLIP > 0 and np.any(np.abs(desired_returns) > _DESIRED_RETURN_CLIP):
            if self.debug_mode:
                print(f"[PCN] 警告: desired_returnsに異常に大きい値が含まれています")
                print(f"  min={np.min(desired_returns)}, max={np.max(desired_returns)}, mean={np.mean(desired_returns)}")
            desired_returns = np.clip(desired_returns, -_DESIRED_RETURN_CLIP, _DESIRED_RETURN_CLIP)
        return observations, actions, desired_returns, desired_horizons

    def _valid_replay_entries(self):
        base = [entry for entry in self.experience_replay if len(entry[2]) > 0]
        if not _FROZEN_PF_CLONE or not self._frozen_pf_entries:
            return base
        if _TEACH_FRONT_ONLY:
            # 教師=凍結アーカイブ(非支配点)のみ。replay 全件(希釈データ)は使わない。
            return [e for e in self._frozen_pf_entries if len(e[2]) > 0]
        # 凍結 best-ever フロントを常時含める（replay から evict されていても再投入）
        seen = {entry[1][1] for entry in base}
        extra = [e for e in self._frozen_pf_entries if len(e[2]) > 0 and e[1][1] not in seen]
        return base + extra

    def _entry_objective_point(self, entry):
        """entry の (cost, avg_wait) を返す。"""
        ep = entry[2]
        if not ep:
            return None
        t0 = ep[0]
        if getattr(t0, "objective_values", None) is not None:
            obj = t0.objective_values
            return (float(obj[0]), float(obj[2]))
        r = np.asarray(t0.reward, dtype=np.float64)
        nj = max(1, int(self._policy_n_jobs()))
        return (float(-r[1]), float(-r[0]) / nj)

    def update_frozen_pf(self):
        """best-ever 非支配エピソードを凍結保持・更新する（劣化しない教師ターゲット）。

        現 replay + 既存 frozen の和集合から非支配集合を取り直すので、方策が劣化しても
        過去に発見した良いフロントは保持され続ける（自己強化崩壊の遮断）。戻り値: 変化したか。
        """
        if not _FROZEN_PF_CLONE:
            return False
        seen = set()
        cand_entries = []
        cand_pts = []
        for entry in list(self.experience_replay) + list(self._frozen_pf_entries):
            if len(entry[2]) == 0:
                continue
            h = entry[1][1]
            if h in seen:
                continue
            pt = self._entry_objective_point(entry)
            if pt is None or not np.all(np.isfinite(pt)):
                continue
            seen.add(h)
            cand_entries.append(entry)
            cand_pts.append(pt)
        if not cand_entries:
            return False
        pts = np.asarray(cand_pts, dtype=np.float64)
        nd = get_non_dominated_inds_minimize(pts)
        frozen = [cand_entries[i] for i in nd]
        if len(frozen) > _FROZEN_PF_MAX:
            fp = np.asarray([self._entry_objective_point(e) for e in frozen], dtype=np.float64)
            order = np.argsort(fp[:, 0])
            keep = np.unique(np.linspace(0, len(order) - 1, _FROZEN_PF_MAX).astype(int))
            frozen = [frozen[order[i]] for i in keep]
        changed = len(frozen) != len(self._frozen_pf_entries)
        self._frozen_pf_entries = frozen
        try:
            fp = np.asarray([self._entry_objective_point(e) for e in frozen], dtype=np.float64)
            print(f"[FROZEN_PF] frozen={len(frozen)} cost[{fp[:,0].min():.2e},{fp[:,0].max():.2e}] "
                  f"wait[{fp[:,1].min():.2e},{fp[:,1].max():.2e}]")
        except Exception:
            pass
        # 教師 cache を作り直して frozen を確実に反映
        if changed:  # [修正] `or True` で無条件staleだった(毎iter全件再構築を強制)
            self.mark_training_batch_cache_stale()
        return True

    def _encode_episode_training_block(self, episode) -> Dict[str, np.ndarray]:
        """1エピソード分の教師 cache ブロックを構築する。

        エピソードは replay 投入後に不変で、この encode は episode と定数(gamma/CLIP)のみに
        依存する決定的計算。そこで結果を先頭 Transition にメモ化し、cache 全件再構築のたびに
        全エピソードを encode し直す O(総transition) の Python ループを回避する(1万ジョブ級で
        再構築が ~50s/iter に達していた主因)。ヒット時も同一入力→同一出力なのでビット一致。
        """
        episode_length = len(episode)
        _first = episode[0]
        _cached = getattr(_first, "_pcn_training_block", None)
        if (
            _cached is not None
            and _cached.get("episode_length") == episode_length
            and _cached.get("label_g", False) == _LABEL_G
        ):
            return _cached
        obs_shape = episode[0].observation.shape
        reward_shape = episode[0].reward.shape
        observations = np.empty((episode_length,) + obs_shape, dtype=np.float32)
        actions = np.empty(episode_length, dtype=np.int64)
        desired_returns = np.empty((episode_length,) + reward_shape, dtype=np.float32)
        desired_horizons = np.empty(episode_length, dtype=np.float32)
        rewards = np.empty((episode_length,) + reward_shape, dtype=np.float32)

        for step_i, transition in enumerate(episode):
            obs_data = transition.observation
            if np.any(np.isnan(obs_data)) or np.any(np.isinf(obs_data)):
                obs_data = np.nan_to_num(obs_data, nan=0.0, posinf=0.0, neginf=0.0)
            observations[step_i] = obs_data
            actions[step_i] = transition.action
            rewards[step_i] = np.nan_to_num(
                np.asarray(transition.reward, dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        if _LABEL_G:
            # [PCN_LABEL_G] 格納値は既に累積化済み（rewards[step_i] = G_t）。再累積しない。
            desired_returns[:] = rewards
        else:
            # NOTE: 既知バグ（既定挙動として温存）: rewards は格納時に累積化済み(=G)なので
            # これは二重累積。修正は PCN_LABEL_G=1（非キャッシュ経路 get_training_batch と対）。
            running_return = np.zeros(reward_shape, dtype=np.float32)
            for step_i in range(episode_length - 1, -1, -1):
                running_return = rewards[step_i] + self.gamma * running_return
                desired_returns[step_i] = running_return

        desired_horizons[:] = np.arange(episode_length, 0, -1, dtype=np.float32)
        if _DESIRED_RETURN_CLIP > 0:
            desired_returns = np.clip(
                desired_returns,
                -_DESIRED_RETURN_CLIP,
                _DESIRED_RETURN_CLIP,
            )
        block = {
            "observations": observations,
            "actions": actions,
            "desired_returns": desired_returns,
            "desired_horizons": desired_horizons,
            "episode_length": int(episode_length),
            "label_g": _LABEL_G,  # メモ化キー: フラグ違いのブロック混入防止
        }
        try:
            _first._pcn_training_block = block  # メモ化(エピソード淘汰でGC、id再利用の誤ヒットなし)
        except Exception:
            pass  # __slots__ 等で貼れない場合は従来動作(毎回計算)
        return block

    def _needs_flat_step_weights(self) -> bool:
        return (
            _TRAIN_MID_STEP_WEIGHT > 1.0
            or _TRAIN_EVALIKE_STEP_WEIGHT > 1.0
            or _TRAIN_KNEE_STEP_WEIGHT > 1.0
            or _TRAIN_LOW_SLOPE_STEP_WEIGHT > 1.0
            or _TRAIN_GIANT_STEP_WEIGHT > 1.0
            or _TRAIN_HEAD_STEP_WEIGHT > 1.0
        )

    def _compute_flat_step_probs(
        self,
        valid_entries,
        blocks: Optional[List[Dict[str, np.ndarray]]],
        *,
        prev_cache: Optional[dict] = None,
        new_blocks_only: Optional[List[Dict[str, np.ndarray]]] = None,
    ) -> Optional[np.ndarray]:
        """flat_step_probs を構築。extend 時は cmax 不変なら新規 ep 分だけ追記（分布は全件再計算と同じ）。"""
        if not self._needs_flat_step_weights():
            return None
        if blocks is not None:
            return self._training_flat_step_weights(valid_entries, blocks)
        if (
            prev_cache is not None
            and new_blocks_only
            and _TRAINING_CACHE_INCREMENTAL
            and prev_cache.get("flat_step_probs") is not None
        ):
            cmax = self._episode_cost_cmax(valid_entries)
            prev_cmax = prev_cache.get("_band_cmax")
            if prev_cmax is not None and cmax == prev_cmax:
                cost_lo, cost_hi, knee_lo, knee_hi, low_lo, low_hi, ep_mid, ep_knee = (
                    self._flat_step_band_sets(valid_entries)
                )
                start_ep = int(prev_cache.get("n_episodes", 0))
                new_parts = [
                    self._episode_flat_step_weight(
                        block,
                        start_ep + i,
                        cost_lo,
                        cost_hi,
                        knee_lo,
                        knee_hi,
                        low_lo,
                        low_hi,
                        ep_mid,
                        ep_knee,
                    )
                    for i, block in enumerate(new_blocks_only)
                ]
                if new_parts:
                    return np.concatenate(
                        [prev_cache["flat_step_probs"], *new_parts], axis=0
                    )
        all_blocks = [
            self._encode_episode_training_block(entry[2]) for entry in valid_entries
        ]
        return self._training_flat_step_weights(valid_entries, all_blocks)

    def _finalize_training_batch_cache(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        desired_returns: np.ndarray,
        desired_horizons: np.ndarray,
        episode_lengths: np.ndarray,
        episode_offsets: np.ndarray,
        valid_entries,
        on_device: bool,
        blocks: Optional[List[Dict[str, np.ndarray]]] = None,
        *,
        prev_cache: Optional[dict] = None,
        new_blocks_only: Optional[List[Dict[str, np.ndarray]]] = None,
    ) -> int:
        (
            episode_probs,
            pf_count,
            endpoint_count,
            recent_count,
            cost_endpoint_count,
            cost_endpoint_action0_rate,
            mid_pf_count,
            low_wait_pf_count,
        ) = self._training_episode_probs(valid_entries)
        if blocks is None and self._needs_flat_step_weights() and prev_cache is None:
            blocks = [
                self._encode_episode_training_block(entry[2]) for entry in valid_entries
            ]
        flat_step_probs = self._compute_flat_step_probs(
            valid_entries,
            blocks,
            prev_cache=prev_cache,
            new_blocks_only=new_blocks_only,
        )
        band_cmax = self._episode_cost_cmax(valid_entries) if self._needs_flat_step_weights() else None

        total_steps = int(episode_lengths.sum())
        cache = {
            "observations": observations,
            "actions": actions,
            "desired_returns": desired_returns,
            "desired_horizons": desired_horizons,
            "episode_lengths": episode_lengths,
            "episode_offsets": episode_offsets,
            "episode_probs": episode_probs,
            "flat_step_probs": flat_step_probs,
            "padded_step_cdf": (
                self._build_padded_step_cdf(episode_lengths, episode_offsets, flat_step_probs)
                if flat_step_probs is not None
                else None
            ),
            "pf_episode_count": pf_count,
            "endpoint_episode_count": endpoint_count,
            "recent_episode_count": recent_count,
            "cost_endpoint_episode_count": cost_endpoint_count,
            "cost_endpoint_action0_rate": cost_endpoint_action0_rate,
            "mid_pf_episode_count": mid_pf_count,
            "low_wait_pf_episode_count": low_wait_pf_count,
            "n_episodes": int(len(episode_lengths)),
            "_band_cmax": band_cmax,
            "on_device": False,
            "nbytes": (
                observations.nbytes
                + actions.nbytes
                + desired_returns.nbytes
                + desired_horizons.nbytes
            ),
        }
        if on_device and str(self.device).startswith("cuda"):
            # [メモリ] 旧 cache の GPU テンソルを先に解放してからで確保する。
            # 解放しないと「旧 + 新」でピークが最終サイズの2倍になり、
            # 長期学習(累積エピソード数に比例)で OOM の主因になる。
            _old = getattr(self, "_training_batch_cache", None)  # 初回は未作成
            if isinstance(_old, dict) and _old.get("on_device"):
                for _k in ("observations", "actions", "desired_returns", "desired_horizons"):
                    _old.pop(_k, None)
                self._training_batch_cache = None
                if th.cuda.is_available():
                    th.cuda.empty_cache()
            try:
                cache["observations"] = th.as_tensor(observations, dtype=th.float32, device=self.device)
                cache["actions"] = th.as_tensor(actions, dtype=th.long, device=self.device)
                cache["desired_returns"] = th.as_tensor(desired_returns, dtype=th.float32, device=self.device)
                cache["desired_horizons"] = th.as_tensor(desired_horizons, dtype=th.float32, device=self.device)
                cache["on_device"] = True
            except RuntimeError as e:
                print(f"[PCN] GPU教師データcache構築に失敗したためCPU cacheへフォールバックします: {e}")
                if th.cuda.is_available():
                    th.cuda.empty_cache()

        self._training_batch_cache = cache
        self._training_batch_cache_stale = False
        return total_steps

    def build_training_batch_cache(self, on_device: bool = False) -> int:
        """現在のreplayをPhase2/3用の教師データへ前計算する。

        サンプリング分布は従来の get_training_batch() と同じく
        「エピソード一様 -> その中のステップ一様」を保つ。
        """
        valid_entries = self._valid_replay_entries()
        valid_episodes = [entry[2] for entry in valid_entries]
        if not valid_episodes:
            self._training_batch_cache = None
            self._training_batch_cache_stale = False
            return 0

        blocks = [self._encode_episode_training_block(episode) for episode in valid_episodes]
        episode_lengths = np.array([block["episode_length"] for block in blocks], dtype=np.int64)
        episode_offsets = np.empty(len(blocks), dtype=np.int64)
        offset = 0
        for episode_i, block in enumerate(blocks):
            episode_offsets[episode_i] = offset
            offset += block["episode_length"]

        observations = np.concatenate([block["observations"] for block in blocks], axis=0)
        actions = np.concatenate([block["actions"] for block in blocks], axis=0)
        desired_returns = np.concatenate([block["desired_returns"] for block in blocks], axis=0)
        desired_horizons = np.concatenate([block["desired_horizons"] for block in blocks], axis=0)
        return self._finalize_training_batch_cache(
            observations,
            actions,
            desired_returns,
            desired_horizons,
            episode_lengths,
            episode_offsets,
            valid_entries,
            on_device,
            blocks=blocks,
        )

    def extend_training_batch_cache(
        self,
        new_episodes: List[List],
        on_device: bool = False,
    ) -> int:
        """既存の教師 cache に新規エピソード分だけ追記する（サンプリング分布は全件で再計算）。"""
        cache = getattr(self, "_training_batch_cache", None)
        if cache is None or not new_episodes:
            return 0

        valid_entries = self._valid_replay_entries()
        expected_n = int(cache.get("n_episodes", len(cache["episode_lengths"])))
        if expected_n + len(new_episodes) != len(valid_entries):
            return self.build_training_batch_cache(on_device=on_device)

        blocks = [self._encode_episode_training_block(episode) for episode in new_episodes]
        new_lengths = np.array([block["episode_length"] for block in blocks], dtype=np.int64)
        old_total = int(cache["episode_lengths"].sum())
        new_offsets = np.empty(len(blocks), dtype=np.int64)
        offset = old_total
        for episode_i, block in enumerate(blocks):
            new_offsets[episode_i] = offset
            offset += block["episode_length"]

        if cache.get("on_device", False):
            device = cache["observations"].device
            new_obs = th.as_tensor(
                np.concatenate([block["observations"] for block in blocks], axis=0),
                dtype=th.float32,
                device=device,
            )
            new_actions = th.as_tensor(
                np.concatenate([block["actions"] for block in blocks], axis=0),
                dtype=th.long,
                device=device,
            )
            new_returns = th.as_tensor(
                np.concatenate([block["desired_returns"] for block in blocks], axis=0),
                dtype=th.float32,
                device=device,
            )
            new_horizons = th.as_tensor(
                np.concatenate([block["desired_horizons"] for block in blocks], axis=0),
                dtype=th.float32,
                device=device,
            )
            observations = th.cat([cache["observations"], new_obs], dim=0)
            actions = th.cat([cache["actions"], new_actions], dim=0)
            desired_returns = th.cat([cache["desired_returns"], new_returns], dim=0)
            desired_horizons = th.cat([cache["desired_horizons"], new_horizons], dim=0)
        else:
            observations = np.concatenate(
                [cache["observations"], *[block["observations"] for block in blocks]],
                axis=0,
            )
            actions = np.concatenate(
                [cache["actions"], *[block["actions"] for block in blocks]],
                axis=0,
            )
            desired_returns = np.concatenate(
                [cache["desired_returns"], *[block["desired_returns"] for block in blocks]],
                axis=0,
            )
            desired_horizons = np.concatenate(
                [cache["desired_horizons"], *[block["desired_horizons"] for block in blocks]],
                axis=0,
            )

        episode_lengths = np.concatenate([cache["episode_lengths"], new_lengths], axis=0)
        episode_offsets = np.concatenate([cache["episode_offsets"], new_offsets], axis=0)
        return self._finalize_training_batch_cache(
            observations,
            actions,
            desired_returns,
            desired_horizons,
            episode_lengths,
            episode_offsets,
            valid_entries,
            on_device,
            prev_cache=cache,
            new_blocks_only=blocks,
        )

    def mark_training_batch_cache_stale(self) -> None:
        """heap 追い出しなど replay 構成が cache と一致しなくなったときに呼ぶ。"""
        self._training_batch_cache_stale = True

    def sync_training_batch_cache(
        self,
        on_device: bool = False,
        new_episodes: Optional[List[List]] = None,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """教師 cache を更新する。新規分のみ追記できれば追記、それ以外は全件再構築。"""
        new_episodes = new_episodes or []
        stale = bool(getattr(self, "_training_batch_cache_stale", False))
        cache = getattr(self, "_training_batch_cache", None)
        mode = "reuse"

        if force_rebuild or stale or cache is None:
            steps = self.build_training_batch_cache(on_device=on_device)
            mode = "rebuild"
        elif new_episodes and _TRAINING_CACHE_INCREMENTAL:
            steps = self.extend_training_batch_cache(new_episodes, on_device=on_device)
            mode = "extend"
        elif new_episodes:
            steps = self.build_training_batch_cache(on_device=on_device)
            mode = "rebuild"
        else:
            steps = int(cache["episode_lengths"].sum()) if cache is not None else 0

        cache = getattr(self, "_training_batch_cache", {}) or {}
        return {
            "steps": steps,
            "mode": mode,
            "cache": cache,
            "n_new_episodes": len(new_episodes),
        }

    def clear_training_batch_cache(self) -> None:
        self._training_batch_cache = None
        self._training_batch_cache_stale = False

    @staticmethod
    def _phase1_sweep_episode_weight(episode) -> float:
        """Phase1 の action 比率スイープ由来エピソードを識別して学習重みを調整する。

        <1.0 で下げ、>1.0 で上げる。アップ重みは「多様な return→action 対応」を持つ
        sweep エピソードを学習で強く効かせ、方策が command を無視して 1 点へ潰れる
        conditioning 崩壊を抑える（崩壊の根本対策）。
        """
        if _PHASE1_SWEEP_TRAIN_WEIGHT == 1.0 or not episode:
            return 1.0
        first = episode[0]
        if getattr(first, "random_action_prob", None) is not None:
            return _PHASE1_SWEEP_TRAIN_WEIGHT
        uid = getattr(first, "_pcn_episode_uid", "")
        if isinstance(uid, str) and uid.startswith("phase1:"):
            return _PHASE1_SWEEP_TRAIN_WEIGHT
        return 1.0

    def _training_episode_probs(self, entries):
        """Archive PF/端点/直近Achieved episodeをやや多く学習するためのepisode重み。

        使うデータはarchive内の経験だけで、到達点を評価結果へ直接混ぜるものではない。
        """
        episodes = [entry[2] for entry in entries]
        n_episodes = len(episodes)
        weights = np.ones(n_episodes, dtype=np.float64)
        if _PHASE1_SWEEP_TRAIN_WEIGHT != 1.0:
            for i, episode in enumerate(episodes):
                weights[i] *= self._phase1_sweep_episode_weight(episode)
        if _SEED_EPISODE_WEIGHT != 1.0:
            _n_seed = 0
            for i, episode in enumerate(episodes):
                if episode and getattr(episode[0], "_pcn_seed_episode", False):
                    weights[i] *= _SEED_EPISODE_WEIGHT
                    _n_seed += 1
            if _n_seed and not getattr(self, "_seed_w_logged", False):
                self._seed_w_logged = True
                print(f"[SEED_EP_WEIGHT] 種エピソード{_n_seed}本に×{_SEED_EPISODE_WEIGHT}の学習重み適用")
        # [PCN_DEDUP_TRAIN_WEIGHT] 同一達成点の重複は 1/本数 に薄める(点ごとに均等な学習機会)。
        if _DEDUP_TRAIN_WEIGHT and n_episodes > 0:
            keys = []
            for episode in episodes:
                first = episode[0]
                obj = getattr(first, "objective_values", None)
                if obj is not None:
                    k = (round(float(obj[0]), _DEDUP_TRAIN_DECIMALS),
                         round(float(obj[2]), _DEDUP_TRAIN_DECIMALS))
                else:
                    r = np.asarray(first.reward, dtype=np.float64)
                    k = (round(float(r[1]), _DEDUP_TRAIN_DECIMALS),
                         round(float(r[0]), _DEDUP_TRAIN_DECIMALS))
                keys.append(k)
            counts: Dict[Any, int] = {}
            for k in keys:
                counts[k] = counts.get(k, 0) + 1
            for i, k in enumerate(keys):
                weights[i] /= float(counts[k])

        # [PCN_ADV_WEIGHT] 帯内相対成績で重み付け(良し悪しの勾配)。
        if _ADV_WEIGHT > 0.0 and n_episodes > 1:
            cw = np.empty((n_episodes, 2), dtype=np.float64)
            for i, episode in enumerate(episodes):
                first = episode[0]
                obj = getattr(first, "objective_values", None)
                if obj is not None:
                    cw[i] = (float(obj[0]), float(obj[2]))
                else:
                    r = np.asarray(first.reward, dtype=np.float64)
                    cw[i] = (-float(r[1]), -float(r[0]))
            edges = np.quantile(cw[:, 0], np.linspace(0.0, 1.0, _ADV_BANDS + 1))
            edges[-1] = np.inf
            band = np.clip(np.searchsorted(edges[1:], cw[:, 0], side="left"), 0, _ADV_BANDS - 1)
            adv = np.zeros(n_episodes, dtype=np.float64)
            for b in range(_ADV_BANDS):
                idx = np.where(band == b)[0]
                if len(idx) < 2:
                    continue
                w_b = cw[idx, 1]
                med = float(np.median(w_b))
                scale = float(np.std(w_b))
                if scale <= 0.0 or not np.isfinite(scale):
                    continue
                adv[idx] = (med - w_b) / scale  # 待ちが短いほど正
            mult = np.exp(np.clip(adv * _ADV_WEIGHT, -np.log(_ADV_CLIP), np.log(_ADV_CLIP)))
            weights *= mult
            if not getattr(self, "_adv_w_logged", False):
                self._adv_w_logged = True
                print(f"[ADV_WEIGHT] 帯内相対成績で重み付け: 係数{_ADV_WEIGHT} 帯{_ADV_BANDS} "
                      f"倍率レンジ[{mult.min():.2f}, {mult.max():.2f}] 中央{np.median(mult):.2f}")
            if not getattr(self, "_dedup_w_logged", False):
                self._dedup_w_logged = True
                _dup = sum(1 for c in counts.values() if c > 1)
                _mx = max(counts.values()) if counts else 0
                print(f"[DEDUP_TRAIN_WEIGHT] ユニーク達成点{len(counts)}/{n_episodes}本 "
                      f"(重複点{_dup}件・最頻{_mx}本) を点ごと均等重みに補正")

        pf_count = 0
        endpoint_count = 0
        recent_count = 0
        cost_endpoint_count = 0
        cost_endpoint_action0_rate = float("nan")
        mid_pf_count = 0
        low_wait_pf_count = 0
        if n_episodes == 0 or (
            not _DEDUP_TRAIN_WEIGHT
            and _ADV_WEIGHT <= 0.0
            and _PHASE1_SWEEP_TRAIN_WEIGHT == 1.0
            and _TRAIN_PF_WEIGHT <= 1.0
            and _TRAIN_ENDPOINT_WEIGHT <= 1.0
            and _TRAIN_RECENT_WEIGHT <= 1.0
            and _TRAIN_COST_ENDPOINT_WEIGHT <= 1.0
            and _TRAIN_MID_PF_WEIGHT <= 1.0
            and _TRAIN_KNEE_PF_WEIGHT <= 1.0
            and _TRAIN_LOW_SLOPE_PF_WEIGHT <= 1.0
            and _TRAIN_LOW_WAIT_PF_WEIGHT <= 1.0
        ):
            return None, pf_count, endpoint_count, recent_count, cost_endpoint_count, cost_endpoint_action0_rate, mid_pf_count, low_wait_pf_count

        if _TRAIN_RECENT_WEIGHT > 1.0:
            step_values = np.array([self._entry_step_value(entry[1]) for entry in entries], dtype=np.float64)
            finite_steps = step_values[np.isfinite(step_values)]
            if len(finite_steps) > 0:
                max_step = float(np.max(finite_steps))
                recent_i = np.where(step_values == max_step)[0]
                if len(recent_i) > 0 and len(recent_i) < n_episodes:
                    weights[recent_i] = np.maximum(weights[recent_i], _TRAIN_RECENT_WEIGHT)
                    recent_count = int(len(recent_i))

        values = []
        has_values = True
        for episode in episodes:
            first = episode[0]
            if hasattr(first, "objective_values") and first.objective_values is not None:
                obj = first.objective_values
                values.append([obj[0], obj[2]])
            else:
                has_values = False
                break

        if has_values:
            points = np.asarray(values, dtype=np.float64)
            pf_i = get_non_dominated_inds_minimize(points)
            minimize_space = True
        else:
            points = np.asarray([episode[0].reward for episode in episodes], dtype=np.float64)
            pf_i = get_non_dominated_inds(points)
            minimize_space = False

        if len(pf_i) > 0:
            weights[pf_i] = np.maximum(weights[pf_i], _TRAIN_PF_WEIGHT * getattr(self, "_pf_weight_mul", 1.0))
            pf_count = int(len(pf_i))
            pf_points = points[pf_i]
            endpoint_local = []
            for obj_i in range(points.shape[1]):
                if minimize_space:
                    endpoint_local.append(int(np.argmin(pf_points[:, obj_i])))
                else:
                    endpoint_local.append(int(np.argmax(pf_points[:, obj_i])))
            endpoint_i = np.unique(pf_i[np.array(endpoint_local, dtype=np.int64)])
            weights[endpoint_i] = np.maximum(weights[endpoint_i], _TRAIN_ENDPOINT_WEIGHT)
            endpoint_count = int(len(endpoint_i))

            if _TRAIN_PF_DENSITY_WEIGHT > 1.0 and len(pf_i) >= 3:
                # PF点を cost-wait 正規化空間へ（軸スケール差を消す）→ 各点の k番目最近傍距離 r_k。
                # r_k 大 = 周囲がスカスカ(低密度) = 重み大。r_k/mean で平均1基準化し weight と同スケールに。
                span = pf_points.max(axis=0) - pf_points.min(axis=0)
                span = np.where(span > 0, span, 1.0)
                norm = pf_points / span
                d = np.linalg.norm(norm[:, None, :] - norm[None, :, :], axis=2)
                d.sort(axis=1)  # 0列目=自分(0)
                kk = min(_TRAIN_PF_DENSITY_K, d.shape[1] - 1)
                r_k = d[:, kk]
                r_mean = float(r_k.mean())
                if r_mean > 0 and np.isfinite(r_mean):
                    dw = _TRAIN_PF_DENSITY_WEIGHT * (r_k / r_mean) ** _TRAIN_PF_DENSITY_ALPHA
                    weights[pf_i] = np.maximum(weights[pf_i], dw)

            if _TRAIN_COST_ENDPOINT_WEIGHT > 1.0:
                # value空間ではcost最小、reward空間では-cost最大がCost端。
                cost_values = pf_points[:, 0] if minimize_space else pf_points[:, 1]
                best_cost_value = np.min(cost_values) if minimize_space else np.max(cost_values)
                cost_endpoint_local = np.where(np.isclose(cost_values, best_cost_value))[0]
                cost_endpoint_i = np.unique(pf_i[cost_endpoint_local])
                if len(cost_endpoint_i) > 0:
                    weights[cost_endpoint_i] = np.maximum(weights[cost_endpoint_i], _pf_region_balanced_weight(_TRAIN_COST_ENDPOINT_WEIGHT, len(cost_endpoint_i)))
                    cost_endpoint_count = int(len(cost_endpoint_i))
                    endpoint_actions = []
                    for idx in cost_endpoint_i:
                        endpoint_actions.extend(int(t.action) for t in episodes[int(idx)])
                    if endpoint_actions:
                        cost_endpoint_action0_rate = float(np.mean(np.asarray(endpoint_actions) == 0))

            if _TRAIN_MID_PF_WEIGHT > 1.0 and minimize_space and len(points) > 0:
                cmax = float(np.max(points[:, 0]))
                if cmax > 0:
                    lo = _TRAIN_MID_COST_MIN_FRAC * cmax
                    hi = _TRAIN_MID_COST_MAX_FRAC * cmax
                    mid_i = np.where((points[:, 0] >= lo) & (points[:, 0] <= hi))[0]
                    if len(mid_i) > 0:
                        weights[mid_i] = np.maximum(weights[mid_i], _pf_region_balanced_weight(_TRAIN_MID_PF_WEIGHT, len(mid_i)))
                        mid_pf_count = int(len(mid_i))

            if _TRAIN_KNEE_PF_WEIGHT > 1.0 and minimize_space and len(points) > 0:
                cmax = float(np.max(points[:, 0]))
                if cmax > 0:
                    lo = _TRAIN_KNEE_COST_MIN_FRAC * cmax
                    hi = _TRAIN_KNEE_COST_MAX_FRAC * cmax
                    knee_i = np.where((points[:, 0] >= lo) & (points[:, 0] <= hi))[0]
                    if len(knee_i) > 0:
                        weights[knee_i] = np.maximum(weights[knee_i], _TRAIN_KNEE_PF_WEIGHT)

            if _TRAIN_LOW_SLOPE_PF_WEIGHT > 1.0 and minimize_space and len(points) > 0:
                cmax = float(np.max(points[:, 0]))
                if cmax > 0:
                    lo = _TRAIN_LOW_SLOPE_COST_MIN_FRAC * cmax
                    hi = _TRAIN_LOW_SLOPE_COST_MAX_FRAC * cmax
                    low_i = np.where((points[:, 0] >= lo) & (points[:, 0] <= hi))[0]
                    if len(low_i) > 0:
                        weights[low_i] = np.maximum(weights[low_i], _TRAIN_LOW_SLOPE_PF_WEIGHT)

            if _TRAIN_LOW_WAIT_PF_WEIGHT > 1.0 and minimize_space and len(points) > 0:
                low_wait_i = np.array([], dtype=np.int64)
                if _TRAIN_LOW_WAIT_MAX > 0:
                    low_wait_i = np.where(points[:, 1] <= _TRAIN_LOW_WAIT_MAX)[0]
                elif _TRAIN_LOW_WAIT_FRAC > 0:
                    q = min(max(_TRAIN_LOW_WAIT_FRAC, 0.0), 1.0) * 100.0
                    wait_hi = float(np.percentile(points[:, 1], q))
                    low_wait_i = np.where(points[:, 1] <= wait_hi)[0]
                if len(low_wait_i) > 0:
                    weights[low_wait_i] = np.maximum(
                        weights[low_wait_i], _pf_region_balanced_weight(_TRAIN_LOW_WAIT_PF_WEIGHT, len(low_wait_i))
                    )
                    low_wait_pf_count = int(len(low_wait_i))

        weights_sum = weights.sum()
        if weights_sum <= 0 or not np.isfinite(weights_sum):
            return None, pf_count, endpoint_count, recent_count, cost_endpoint_count, cost_endpoint_action0_rate, mid_pf_count, low_wait_pf_count
        return weights / weights_sum, pf_count, endpoint_count, recent_count, cost_endpoint_count, cost_endpoint_action0_rate, mid_pf_count, low_wait_pf_count

    @staticmethod
    def _entry_step_value(step_info) -> float:
        if isinstance(step_info, (tuple, list)) and len(step_info) > 0:
            step_info = step_info[0]
        try:
            return float(step_info)
        except (TypeError, ValueError):
            return float("nan")

    @staticmethod
    def _sym_kl_hinge_mean(logits_a: th.Tensor, logits_b: th.Tensor, kl_margin: float) -> th.Tensor:
        log_pa = F.log_softmax(logits_a, dim=-1)
        log_pb = F.log_softmax(logits_b, dim=-1)
        if _KL_NANSAFE:
            # logit飽和(p=厳密0)時に xlogy backward が nan になる→学習全停止(凍結run)。
            # log確率を log(1e-12) で床打ち。健全域(p≫1e-12)では値不変。
            log_pa = log_pa.clamp_min(_KL_LOG_FLOOR)
            log_pb = log_pb.clamp_min(_KL_LOG_FLOOR)
        pa = th.exp(log_pa)
        pb = th.exp(log_pb)
        sym_kl = 0.5 * (
            F.kl_div(log_pa, pb, reduction="none").sum(dim=-1)
            + F.kl_div(log_pb, pa, reduction="none").sum(dim=-1)
        )
        return F.relu(kl_margin - sym_kl).mean()

    @staticmethod
    def _sym_kl_hinge_mean_batched(
        logits_a: th.Tensor, logits_b: th.Tensor, kl_margin: float
    ) -> th.Tensor:
        """[k,n,A] の全ペアを 1 回でまとめて評価。

        旧来の `stack([_sym_kl_hinge_mean(a[i],b[i]) for i]).mean()` と同じ二段平均
        （ペア内 n 平均 → ペア間平均）を保つので値は一致。ペア毎の Python ループ＋小カーネル
        大量発行を 1 本にまとめ、CPU ディスパッチ（launch）コストを削減する。
        """
        log_pa = F.log_softmax(logits_a, dim=-1)
        log_pb = F.log_softmax(logits_b, dim=-1)
        if _KL_NANSAFE:
            log_pa = log_pa.clamp_min(_KL_LOG_FLOOR)
            log_pb = log_pb.clamp_min(_KL_LOG_FLOOR)
        pa = th.exp(log_pa)
        pb = th.exp(log_pb)
        sym_kl = 0.5 * (
            F.kl_div(log_pa, pb, reduction="none").sum(dim=-1)
            + F.kl_div(log_pb, pa, reduction="none").sum(dim=-1)
        )  # [k, n]
        return F.relu(kl_margin - sym_kl).mean(dim=1).mean()

    def _batched_paired_command_forward(
        self,
        obs_n: th.Tensor,
        hz_n: th.Tensor,
        dr_a_list: list,
        dr_b_list: list,
    ):
        """command ペアごとの forward を 1 回にまとめる（数学は従来の2回 forward と同値）。"""
        n_obs = int(obs_n.shape[0])
        k_pairs = len(dr_a_list)
        if k_pairs == 0 or len(dr_b_list) != k_pairs:
            return None, None
        dev = obs_n.device
        dr_a = th.stack(
            [th.as_tensor(dr, device=dev, dtype=th.float32) for dr in dr_a_list]
        )
        dr_b = th.stack(
            [th.as_tensor(dr, device=dev, dtype=th.float32) for dr in dr_b_list]
        )
        obs_exp = obs_n.unsqueeze(0).expand(k_pairs, n_obs, -1).reshape(k_pairs * n_obs, -1)
        if hz_n.dim() > 1:
            hz_exp = hz_n.unsqueeze(0).expand(k_pairs, n_obs, -1).reshape(k_pairs * n_obs, -1)
        else:
            hz_exp = hz_n.unsqueeze(0).expand(k_pairs, n_obs).reshape(k_pairs * n_obs, 1)
        dr_a_exp = dr_a.unsqueeze(1).expand(k_pairs, n_obs, 2).reshape(k_pairs * n_obs, 2)
        dr_b_exp = dr_b.unsqueeze(1).expand(k_pairs, n_obs, 2).reshape(k_pairs * n_obs, 2)
        cat_obs = th.cat([obs_exp, obs_exp], dim=0)
        cat_dr = th.cat([dr_a_exp, dr_b_exp], dim=0)
        cat_hz = th.cat([hz_exp, hz_exp], dim=0)
        out = self.model(cat_obs, cat_dr, cat_hz)
        if isinstance(out, tuple):
            out = out[0]
        half = k_pairs * n_obs
        logits_a = out[:half].view(k_pairs, n_obs, -1)
        logits_b = out[half:].view(k_pairs, n_obs, -1)
        return logits_a, logits_b

    def _archive_wait_conditioning_loss(
        self,
        obs: th.Tensor,
        desired_horizon: th.Tensor,
        archive_cmds: np.ndarray,
        cond_weight: float,
        n_cost_levels: int,
        n_wait_levels: int,
        kl_margin: float,
        max_samples: int,
    ):
        """Archive PF 上の cost command 固定で wait command を振り方策分岐を促す。"""
        if cond_weight <= 0.0 or self.use_enhanced_model:
            return th.zeros((), device=obs.device), 0
        if archive_cmds.shape[0] < 2:
            return th.zeros((), device=obs.device), 0

        n_obs = min(int(obs.shape[0]), max_samples)
        if n_obs < 1:
            return th.zeros((), device=obs.device), 0

        obs_n = obs[:n_obs]
        hz_n = desired_horizon[:n_obs]
        n_cost = min(n_cost_levels, int(archive_cmds.shape[0]))
        cost_pick = self.np_random.choice(archive_cmds.shape[0], size=n_cost, replace=False)
        n_wait = max(2, n_wait_levels)
        r0_span = np.linspace(
            float(archive_cmds[:, 0].min()),
            float(archive_cmds[:, 0].max()),
            n_wait,
            dtype=np.float32,
        )
        dr_a_list, dr_b_list = [], []
        for ci in cost_pick:
            dr_base = archive_cmds[int(ci)].copy()
            for wi in range(n_wait - 1):
                dr_a = dr_base.copy()
                dr_b = dr_base.copy()
                dr_a[0] = r0_span[wi]
                dr_b[0] = r0_span[wi + 1]
                dr_a_list.append(dr_a)
                dr_b_list.append(dr_b)
        logits_a, logits_b = self._batched_paired_command_forward(obs_n, hz_n, dr_a_list, dr_b_list)
        if logits_a is None:
            return th.zeros((), device=obs.device), 0
        if _FAST_UPDATE:
            loss = self._sym_kl_hinge_mean_batched(logits_a, logits_b, kl_margin)
        else:
            losses = [
                self._sym_kl_hinge_mean(logits_a[i], logits_b[i], kl_margin)
                for i in range(logits_a.shape[0])
            ]
            loss = th.stack(losses).mean()
        return loss, len(dr_a_list)

    def _archive_pf_arc_conditioning_loss(
        self,
        obs: th.Tensor,
        desired_horizon: th.Tensor,
        archive_cmds: np.ndarray,
        cond_weight: float,
        kl_margin: float,
        max_samples: int,
        min_r1_sep_frac: float,
    ):
        """Archive PF 上の隣接 command（主に r1/cost 側）で方策が分岐するよう促す（左上先端の滑らかな下降）。"""
        if cond_weight <= 0.0 or self.use_enhanced_model:
            return th.zeros((), device=obs.device), 0
        if archive_cmds.shape[0] < 2:
            return th.zeros((), device=obs.device), 0

        n_obs = min(int(obs.shape[0]), max_samples)
        if n_obs < 1:
            return th.zeros((), device=obs.device), 0

        order = np.argsort(archive_cmds[:, 1])
        cmds = archive_cmds[order]
        r1_span = float(cmds[:, 1].max() - cmds[:, 1].min())
        min_sep = max(abs(r1_span) * min_r1_sep_frac, 1.0)
        pair_idx = []
        for i in range(len(cmds) - 1):
            if abs(float(cmds[i + 1, 1] - cmds[i, 1])) >= min_sep:
                pair_idx.append(i)
        if not pair_idx:
            return th.zeros((), device=obs.device), 0

        n_pairs_take = min(len(pair_idx), _LOW_BAND_COND_COST_LEVELS)
        pick = self.np_random.choice(len(pair_idx), size=n_pairs_take, replace=False)

        obs_n = obs[:n_obs]
        hz_n = desired_horizon[:n_obs]
        dr_a_list, dr_b_list = [], []
        for pi in pick:
            i = pair_idx[int(pi)]
            dr_a_list.append(cmds[i].copy())
            dr_b_list.append(cmds[i + 1].copy())
        logits_a, logits_b = self._batched_paired_command_forward(obs_n, hz_n, dr_a_list, dr_b_list)
        if logits_a is None:
            return th.zeros((), device=obs.device), 0
        if _FAST_UPDATE:
            loss = self._sym_kl_hinge_mean_batched(logits_a, logits_b, kl_margin)
        else:
            losses = [
                self._sym_kl_hinge_mean(logits_a[i], logits_b[i], kl_margin)
                for i in range(logits_a.shape[0])
            ]
            loss = th.stack(losses).mean()
        return loss, len(dr_a_list)

    def _archive_r1_sweep_conditioning_loss(
        self,
        obs: th.Tensor,
        desired_horizon: th.Tensor,
        archive_cmds: np.ndarray,
        cond_weight: float,
        n_r1_levels: int,
        kl_margin: float,
        max_samples: int,
    ):
        """r0(wait) 固定で r1(cost) を振り、cost command への応答を促す。"""
        if cond_weight <= 0.0 or self.use_enhanced_model:
            return th.zeros((), device=obs.device), 0
        if archive_cmds.shape[0] < 2:
            return th.zeros((), device=obs.device), 0

        n_obs = min(int(obs.shape[0]), max_samples)
        if n_obs < 1:
            return th.zeros((), device=obs.device), 0

        obs_n = obs[:n_obs]
        hz_n = desired_horizon[:n_obs]
        # cost 最小（r1 が 0 に最も近い）Archive 点の r0＝高 wait 端の command
        cost_ep_i = int(np.argmax(archive_cmds[:, 1]))
        r0_fix = float(archive_cmds[cost_ep_i, 0])
        r1_sorted = np.unique(np.sort(archive_cmds[:, 1].astype(np.float64)))
        if len(r1_sorted) >= 2:
            if len(r1_sorted) > n_r1_levels:
                idx = np.linspace(0, len(r1_sorted) - 1, n_r1_levels).astype(int)
                r1_span = r1_sorted[idx].astype(np.float32)
            else:
                r1_span = r1_sorted.astype(np.float32)
        else:
            r1_span = np.linspace(
                float(archive_cmds[:, 1].min()),
                float(archive_cmds[:, 1].max()),
                max(2, n_r1_levels),
                dtype=np.float32,
            )
        dr_a_list, dr_b_list = [], []
        for wi in range(len(r1_span) - 1):
            dr_a_list.append(np.array([r0_fix, r1_span[wi]], dtype=np.float32))
            dr_b_list.append(np.array([r0_fix, r1_span[wi + 1]], dtype=np.float32))
        logits_a, logits_b = self._batched_paired_command_forward(obs_n, hz_n, dr_a_list, dr_b_list)
        if logits_a is None:
            return th.zeros((), device=obs.device), 0
        if _FAST_UPDATE:
            loss = self._sym_kl_hinge_mean_batched(logits_a, logits_b, kl_margin)
        else:
            losses = [
                self._sym_kl_hinge_mean(logits_a[i], logits_b[i], kl_margin)
                for i in range(logits_a.shape[0])
            ]
            loss = th.stack(losses).mean()
        return loss, len(dr_a_list)

    def _mid_band_wait_conditioning_loss(
        self, obs: th.Tensor, desired_horizon: th.Tensor
    ):
        """Archive 中域 PF: cost command 固定で wait command を振ったとき方策が分岐するよう促す。"""
        return self._archive_wait_conditioning_loss(
            obs,
            desired_horizon,
            self._collect_mid_band_archive_commands(),
            _MID_BAND_COND_WEIGHT,
            _MID_BAND_COND_COST_LEVELS,
            _MID_BAND_COND_WAIT_LEVELS,
            _MID_BAND_COND_KL_MARGIN,
            _MID_BAND_COND_MAX_SAMPLES,
        )

    def _low_slope_conditioning_loss(
        self, obs: th.Tensor, desired_horizon: th.Tensor
    ):
        """Archive 低 cost PF: 左上先端の滑らかな下降。"""
        archive_cmds = self._collect_low_slope_archive_commands()
        if _LOW_BAND_COND_MODE in ("r1_sweep", "dual"):
            l1, p1 = self._archive_r1_sweep_conditioning_loss(
                obs,
                desired_horizon,
                archive_cmds,
                1.0,
                _LOW_BAND_COND_COST_LEVELS,
                _LOW_BAND_COND_KL_MARGIN,
                _LOW_BAND_COND_MAX_SAMPLES,
            )
            if _LOW_BAND_COND_MODE == "r1_sweep":
                return l1 * _LOW_BAND_COND_WEIGHT, p1
        else:
            l1, p1 = th.zeros((), device=obs.device), 0
        if _LOW_BAND_COND_MODE in ("arc", "dual"):
            l2, p2 = self._archive_pf_arc_conditioning_loss(
                obs,
                desired_horizon,
                archive_cmds,
                1.0,
                _LOW_BAND_COND_KL_MARGIN,
                _LOW_BAND_COND_MAX_SAMPLES,
                _LOW_BAND_COND_MIN_R1_SEP_FRAC,
            )
            if _LOW_BAND_COND_MODE == "arc":
                return l2 * _LOW_BAND_COND_WEIGHT, p2
            r1f = float(np.clip(_LOW_BAND_DUAL_R1_FRAC, 0.05, 0.95))
            return (
                _LOW_BAND_COND_WEIGHT * r1f * l1 + _LOW_BAND_COND_WEIGHT * (1.0 - r1f) * l2,
                p1 + p2,
            )
        if _LOW_BAND_COND_MODE == "r0_sweep":
            return self._archive_wait_conditioning_loss(
                obs,
                desired_horizon,
                archive_cmds,
                _LOW_BAND_COND_WEIGHT,
                _LOW_BAND_COND_COST_LEVELS,
                _LOW_BAND_COND_WAIT_LEVELS,
                _LOW_BAND_COND_KL_MARGIN,
                _LOW_BAND_COND_MAX_SAMPLES,
            )
        if _LOW_BAND_COND_MODE in ("both", "r1r0"):
            # cost軸(r1_sweep)と wait軸(r0_sweep)の両方の分岐を同時に促す。
            # 圧縮seedは r1_sweep で cost追従は得るが、cost-0 corner の wait条件付けが
            # スクランブル（高wait指令→高cost）して極に届かない → r0_sweep を足して両軸を訓練。
            l1, p1 = self._archive_r1_sweep_conditioning_loss(
                obs, desired_horizon, archive_cmds, 1.0,
                _LOW_BAND_COND_COST_LEVELS, _LOW_BAND_COND_KL_MARGIN, _LOW_BAND_COND_MAX_SAMPLES,
            )
            l0, p0 = self._archive_wait_conditioning_loss(
                obs, desired_horizon, archive_cmds, _LOW_BAND_COND_WEIGHT,
                _LOW_BAND_COND_COST_LEVELS, _LOW_BAND_COND_WAIT_LEVELS,
                _LOW_BAND_COND_KL_MARGIN, _LOW_BAND_COND_MAX_SAMPLES,
            )
            return _LOW_BAND_COND_WEIGHT * (l1 + l0), p1 + p0
        return th.zeros((), device=obs.device), 0

    def _conditioning_sensitivity_loss(
        self, obs: th.Tensor, desired_return: th.Tensor, desired_horizon: th.Tensor
    ):
        """同じ obs・異なる command のペアで sym_kl が margin 未満のときだけ押し離す（ヒンジ）。"""
        n = min(int(obs.shape[0]), 256)
        obs_n = obs[:n]
        dr_n = desired_return[:n]
        dh_n = desired_horizon[:n]
        with th.no_grad():
            obs_diff = th.cdist(obs_n, obs_n, p=2)
            dr_diff = th.cdist(dr_n, dr_n, p=2)
            wait_thresh = (
                _CONDITIONING_SENS_WAIT_DR_THRESH
                if _CONDITIONING_SENS_WAIT_DR_THRESH > 0
                else _CONDITIONING_SENS_DR_THRESH
            )
            dr_r0 = th.cdist(dr_n[:, :1], dr_n[:, :1], p=2)
            dr_r1 = th.cdist(dr_n[:, 1:2], dr_n[:, 1:2], p=2)
            cmd_sep = (dr_r0 >= wait_thresh) | (dr_r1 >= _CONDITIONING_SENS_DR_THRESH)
            pair_mask = (obs_diff <= _CONDITIONING_SENS_OBS_THRESH) & cmd_sep
            pair_mask.fill_diagonal_(False)
            pair_idx = pair_mask.nonzero(as_tuple=False)
            if pair_idx.shape[0] > _CONDITIONING_SENS_MAX_PAIRS:
                perm = th.randperm(pair_idx.shape[0], device=pair_idx.device)[:_CONDITIONING_SENS_MAX_PAIRS]
                pair_idx = pair_idx[perm]
        if pair_idx.shape[0] == 0:
            return th.zeros((), device=obs.device), 0

        i_idx = pair_idx[:, 0]
        j_idx = pair_idx[:, 1]
        logits_i = self.model(obs_n[i_idx], dr_n[i_idx], dh_n[i_idx])
        logits_j = self.model(obs_n[i_idx], dr_n[j_idx], dh_n[i_idx])
        if isinstance(logits_i, tuple):
            logits_i = logits_i[0]
        if isinstance(logits_j, tuple):
            logits_j = logits_j[0]
        log_p_i = F.log_softmax(logits_i, dim=-1)
        log_p_j = F.log_softmax(logits_j, dim=-1)
        if _KL_NANSAFE:
            log_p_i = log_p_i.clamp_min(_KL_LOG_FLOOR)
            log_p_j = log_p_j.clamp_min(_KL_LOG_FLOOR)
        p_i = th.exp(log_p_i)
        p_j = th.exp(log_p_j)
        kl_ij = F.kl_div(log_p_i, p_j, reduction="none").sum(dim=-1)
        kl_ji = F.kl_div(log_p_j, p_i, reduction="none").sum(dim=-1)
        sym_kl = 0.5 * (kl_ij + kl_ji)
        margin = _CONDITIONING_KL_MARGIN
        sens_loss = F.relu(margin - sym_kl).mean()
        return sens_loss, int(pair_idx.shape[0])

    def _archive_value_reproduction_loss_from_replay(self) -> Optional[th.Tensor]:
        """エピソード先頭の (obs, command) から Archive (Cost, AvgWait) を回帰。"""
        if _VALUE_REPRO_WEIGHT <= 0:
            return None
        valid = [entry[2] for entry in self.experience_replay if len(entry[2]) > 0]
        if not valid:
            return None
        with_obj = []
        for episode in valid:
            first = episode[0]
            if hasattr(first, "objective_values") and first.objective_values is not None:
                with_obj.append(episode)
        if not with_obj:
            return None
        max_eps = _VALUE_REPRO_MAX_EPISODES
        if len(with_obj) > max_eps:
            pick = self.np_random.choice(len(with_obj), size=max_eps, replace=False)
            with_obj = [with_obj[int(i)] for i in pick]
        obs_list, dr_list, hz_list, targets = [], [], [], []
        for episode in with_obj:
            first = episode[0]
            obj = first.objective_values
            obs_list.append(np.asarray(first.observation, dtype=np.float32))
            dr_list.append(np.asarray(first.reward, dtype=np.float32))
            hz_list.append(float(len(episode)))
            targets.append([float(obj[0]), float(obj[2])])
        obs_t = th.tensor(np.stack(obs_list), device=self.device, dtype=th.float32)
        dr_t = th.tensor(np.stack(dr_list), device=self.device, dtype=th.float32)
        hz_t = th.tensor(np.array(hz_list, dtype=np.float32), device=self.device).unsqueeze(1)
        target_t = th.tensor(np.array(targets, dtype=np.float32), device=self.device)
        model = self.network if self.use_enhanced_model else self.model
        if not hasattr(model, "predict_archive_value"):
            return None
        pred = model.predict_archive_value(obs_t, dr_t, hz_t)
        return F.smooth_l1_loss(pred, target_t)

    def _command_track_loss_from_replay(self) -> Optional[th.Tensor]:
        """[案1] 生成指令(command_return)に「達成costが届かない側だけ」を二乗罰する片側MSE + 回帰アンカー。
        value_head proxy(s.detach)で達成を予測し、指令costへ届かせる方向の勾配を c_emb 経由で方策へ流す。
        v̂ を実達成に固定する回帰(anchor)が無いと v̂ が嘘をついて track を騙すので必須。OFF(weight<=0)で None。"""
        if _CMD_TRACK_WEIGHT <= 0:
            return None
        eps = []
        for entry in self.experience_replay:
            episode = entry[2]
            if not episode:
                continue
            first = episode[0]
            if (getattr(first, "command_return", None) is not None
                    and getattr(first, "objective_values", None) is not None):
                eps.append(episode)
        if not eps:
            return None
        if len(eps) > _CMD_TRACK_MAX_EPISODES:
            pick = self.np_random.choice(len(eps), size=_CMD_TRACK_MAX_EPISODES, replace=False)
            eps = [eps[int(i)] for i in pick]
        obs_list, dr_list, hz_list, cmd_cost_list, ach_list = [], [], [], [], []
        for episode in eps:
            first = episode[0]
            obj = first.objective_values            # [cost, _, avg_wait]（達成）
            cr = np.asarray(first.command_return, dtype=np.float32)  # [-avg_wait*nj, -cost]（指令, reward空間）
            obs_list.append(np.asarray(first.observation, dtype=np.float32))
            dr_list.append(cr)                       # value_head の条件入力 = 生成指令そのもの
            hz_list.append(float(len(episode)))
            cmd_cost_list.append(float(-cr[1]))      # 指令の cost 目標（nj不要）
            ach_list.append([float(obj[0]), float(obj[2])])  # 達成 [cost, avg_wait]
        model = self.network if self.use_enhanced_model else self.model
        if not hasattr(model, "predict_archive_value"):
            return None
        obs_t = th.tensor(np.stack(obs_list), device=self.device, dtype=th.float32)
        dr_t = th.tensor(np.stack(dr_list), device=self.device, dtype=th.float32)
        hz_t = th.tensor(np.array(hz_list, dtype=np.float32), device=self.device).unsqueeze(1)
        cmd_cost_t = th.tensor(np.array(cmd_cost_list, dtype=np.float32), device=self.device)  # detach定数
        ach_t = th.tensor(np.array(ach_list, dtype=np.float32), device=self.device)            # detach定数
        # データ追従の正規化スケール（return_norm_scale = reward空間[-wait*nj,-cost]のp99.5）。
        # 固定 _VALUE_COST_SCALE(1e5) では trace256(cost~2.25億)を [-1,0] に正規化できず loss が桁外れになる
        # （他項を支配・崩壊）。学習データから自動追従する desired_return_scale を使い O(1) に揃える。
        _drs = model.desired_return_scale.detach().to(th.float32)
        # [2026-08-27 単位統一] wait系は全て「総待ちスケール drs[0] を1とする分数」で扱う。
        # 価値ヘッドのwait出力は _VALUE_WAIT_SCALE(=workload較正で総待ちスケール)で校正されるため、
        # 平均待ちスケール(drs[0]/nj)で割ると nj倍(5万)の不整合が anchor に化けて暴走する
        # (smoke20bで実測: anchor~6000。v9はnj=1バグが偶然これを隠していた)。
        _nj = self._policy_n_jobs()
        cs = max(float(_drs[1].item()), 1.0)        # cost scale（~5.56e8 for trace256）
        ws = max(float(_drs[0].item()), 1.0)        # 総待ちスケール（v̂/ach/cmd を同一単位に）
        # fp32 で計算（AMP fp16 の cost~1e5 overflow を回避。MEMORY の log2 stuck と同根を予防）。
        with th.cuda.amp.autocast(enabled=False):
            v_hat = model.predict_archive_value(obs_t.float(), dr_t.float(), hz_t.float(), detach_repr=True)  # [B,2] objスケール
            v_cost_n = v_hat[:, 0] / cs
            v_wait_n = v_hat[:, 1] / ws
            cmd_cost_n = cmd_cost_t / cs
            ach_cost_n = ach_t[:, 0] / cs
            ach_wait_n = (ach_t[:, 1] * float(_nj)) / ws  # 達成avg_wait→総待ち→分数
            # 片側ヒンジ: 予測達成cost が指令cost を「上回る=届かない」側だけ二乗罰（cost は小さいほど良い）
            miss = th.relu(v_cost_n - cmd_cost_n)
            track = (miss ** 2).mean()
            # [v10] wait側の片側ヒンジ: 予測達成waitが指令waitを上回る側だけ罰。
            # 指令wait(正規化) = (-command_return[0]) / drs[0]（cr[0]=-avg_wait*nj, drs[0]=wait*njのscale）。
            if _CMD_TRACK_WAIT_WEIGHT > 0:
                cmd_wait_n = th.tensor(
                    np.array([float(-c[0]) for c in dr_list], dtype=np.float32),
                    device=self.device) / max(float(_drs[0].item()), 1.0)
                if _COND_WAIT_ROBUST == "logexpand":
                    # [critic C1対応] ヒンジを対数空間で取る。線形zのままだと目標帯(0-30秒,
                    # v_wait_n<0.013)で miss² が 1e-4 級に潰れ、罰が数値的に不在になる。
                    # z0はヒンジ専用(_CMD_TRACK_WAIT_Z0)。入力側の_COND_WAIT_Z0とは空間が別。
                    _z0 = _CMD_TRACK_WAIT_Z0
                    _den = float(np.log1p(1.0 / _z0))
                    _yv = th.log1p(th.clamp(v_wait_n, min=0.0) / _z0) / _den
                    _yc = th.log1p(th.clamp(cmd_wait_n, min=0.0) / _z0) / _den
                    miss_w = th.relu(_yv - _yc)
                else:
                    miss_w = th.relu(v_wait_n - cmd_wait_n)
                track = track + (_CMD_TRACK_WAIT_WEIGHT / max(_CMD_TRACK_WEIGHT, 1e-12)) \
                    * (miss_w ** 2).mean()
                try:
                    self._cmd_track_parts = {
                        "cost_miss2": float((miss ** 2).mean().detach().item()),
                        "wait_miss2": float((miss_w ** 2).mean().detach().item()),
                    }
                except Exception:
                    pass
            # 回帰アンカー: v̂ を実達成に固定（両成分）。v̂ が嘘をつくのを防ぐ。
            anchor = F.smooth_l1_loss(v_cost_n, ach_cost_n) + F.smooth_l1_loss(v_wait_n, ach_wait_n)
            loss = track + _CMD_TRACK_ANCHOR_WEIGHT * anchor
        # [breaker分離用] 遮断時もアンカー(校正)だけは学習に通すため、部品を保持する。
        # アンカーまで遮断すると「未校正→大損失→遮断→未校正」のデッドロックになる(smoke20cで実測)。
        self._cmd_track_split = (track, anchor)
        if not th.isfinite(loss):
            return None
        return loss

    def probe_wait_sensitivity(self, n: int = 64) -> Optional[dict]:
        """[v10計装] 同一obs・同一cost指令でwait指令だけ変えた行動分布のTV距離平均。
        学習不介入(no_grad, 一時eval=dropoutのRNG消費なし)。v9で「wait死を検知する信号が
        無かった」の再発防止。0なら網はwait指令を完全無視している。"""
        cache = getattr(self, "_training_batch_cache", None)
        if not cache or "observations" not in cache:
            return None
        obs = cache["observations"][:n]
        obs_t = (obs.to(self.device).float() if th.is_tensor(obs)
                 else th.tensor(np.asarray(obs), device=self.device, dtype=th.float32))
        model = self.network if self.use_enhanced_model else self.model
        drs = model.desired_return_scale.detach()
        B = int(obs_t.shape[0])
        if B == 0:
            return None
        cost_cmd = -0.5 * float(drs[1].item())
        # 3水準(z相当 0 / 0.02 / 0.2)で単調性まで見る(critic対応: 2水準tvは未学習網でも
        # 非ゼロが出る弱い計器。P(cloud)の単調応答が本命の観察量)。
        levels = (0.0, 0.02, 0.2)
        nj = self._policy_n_jobs()
        hz = th.full((B, 1), float(nj), device=self.device, dtype=th.float32)
        was_training = model.training
        model.eval()
        try:
            pcs = []
            probs = []
            with th.no_grad():
                for lv in levels:
                    dr = th.tensor([[-lv * float(drs[0].item()), cost_cmd]],
                                   device=self.device, dtype=th.float32).repeat(B, 1)
                    o = model(obs_t, dr, hz)
                    if isinstance(o, tuple):
                        o = o[0]
                    p = th.exp(F.log_softmax(o, dim=-1))
                    probs.append(p)
                    pcs.append(float(p[:, 1].mean().item()) if p.shape[-1] > 1 else 0.0)
                tv01 = float((0.5 * (probs[0] - probs[1]).abs().sum(-1).mean()).item())
                tv12 = float((0.5 * (probs[1] - probs[2]).abs().sum(-1).mean()).item())
            return {"tv01": tv01, "tv12": tv12,
                    "pc": [round(x, 4) for x in pcs]}
        finally:
            model.train(was_training)

    def update(self, learning_rate=None, compute_metrics: bool = True):
        """Update PCN model - 最適化版

        compute_metrics=False では、勾配に寄与しない診断 metrics（policy_acc 等や
        conditioning 損失値の .item()）を計算しない。損失テンソル l 自体（=勾配）は不変なので
        重み更新はビット一致。update_many が最終 update 以外で False を渡す。
        """
        start_time = time.time()
        if _PROFILE and not hasattr(self, "_prof_acc"):
            self._prof_acc = {"batch": 0.0, "fwd": 0.0, "loss": 0.0, "bwd": 0.0, "opt": 0.0, "n": 0}
        _pt = time.perf_counter() if _PROFILE else 0.0
        original_lr = None
        if learning_rate is not None:
            original_lr = self.opt.param_groups[0]['lr']
            self.opt.param_groups[0]['lr'] = learning_rate
        observations, actions, desired_returns, desired_horizons = self.get_training_batch()
        if _PROFILE:
            _now = time.perf_counter(); self._prof_acc["batch"] += _now - _pt; self._prof_acc["n"] += 1; _pt = _now
        if os.environ.get("PCN_DIAG_BATCH") == "1" and getattr(self, "_diag_batch_n", 0) < 3:
            self._diag_batch_n = getattr(self, "_diag_batch_n", 0) + 1
            import torch as _th
            _o = observations.detach().float() if _th.is_tensor(observations) else _th.as_tensor(np.asarray(observations), dtype=_th.float32)
            _dr = desired_returns.detach().float() if _th.is_tensor(desired_returns) else _th.as_tensor(np.asarray(desired_returns), dtype=_th.float32)
            _a = actions.detach() if _th.is_tensor(actions) else _th.as_tensor(np.asarray(actions))
            _sc = self.model.desired_return_scale.detach().float().cpu().numpy() if hasattr(self.model, "desired_return_scale") else None
            _drn = (_dr.cpu() / _th.as_tensor(_sc)) if _sc is not None else _dr.cpu()
            _ab = _th.bincount(_a.long().flatten().cpu(), minlength=2).tolist()
            print(f"[DIAG_BATCH#{self._diag_batch_n}] B={_o.shape} act_bincount={_ab} "
                  f"raw_dr[min,max]=[{float(_dr.min()):.3g},{float(_dr.max()):.3g}] "
                  f"norm_scale={_sc.tolist() if _sc is not None else None} "
                  f"norm_dr[min,max]=[{float(_drn.min()):.3g},{float(_drn.max()):.3g}] "
                  f"norm_dr_std_per_dim={_drn.std(0).tolist()} "
                  f"obs[min,max]=[{float(_o.min()):.3g},{float(_o.max()):.3g}] "
                  f"obs_nan={int(_th.isnan(_o).sum())} dr_nan={int(_th.isnan(_dr).sum())}", flush=True)
        metrics = {}
        with th.cuda.amp.autocast(enabled=self.use_amp):  # 混合精度学習
            if th.is_tensor(observations):
                obs = observations
                actions = actions
                desired_return = desired_returns
                desired_horizon = desired_horizons.unsqueeze(1)
            else:
                obs = th.from_numpy(observations).to(self.device, non_blocking=True)
                actions = th.from_numpy(actions).to(self.device, non_blocking=True)
                desired_return = th.from_numpy(desired_returns).to(self.device, non_blocking=True)
                desired_horizon = th.from_numpy(desired_horizons).to(self.device, non_blocking=True).unsqueeze(1)
            
            # desired_returnの値を正規化（異常に大きい値を防ぐ）
            # モデルの入力が不安定になるのを防ぐため、値を適切な範囲にクリッピング
            # 注意: scaling_factorが[1, 1, 1]の場合、desired_returnが大きすぎると
            # モデルの内部で数値的不安定性が発生する可能性がある
            # より小さな範囲にクリッピング（-1000から1000の範囲）
            if _DESIRED_RETURN_CLIP > 0:
                desired_return = th.clamp(desired_return, min=-_DESIRED_RETURN_CLIP, max=_DESIRED_RETURN_CLIP)
            
            # desired_horizonもクリッピング（異常に大きい値を防ぐ）
            desired_horizon = th.clamp(desired_horizon, min=0.0, max=1e6)
            
            # 観測データの値もクリッピング（Inf 保護用に広く取る。±1000 だと 1024 ジョブの
            # イベント観測(処理時間~1e4)が潰れて情報を失うため、モデル forward と同じ ±1e6 に合わせる）
            obs = th.clamp(obs, min=-1e6, max=1e6)
            
            # 7. 最適化された勾配計算
            self.opt.zero_grad(set_to_none=True)
            
            # 8. モデル推論前のデータ検証（純診断 print のみ・補正なし。既定スキップで .any() 同期を除去）
            if _UPDATE_NANCHECK:
                if th.isnan(obs).any() or th.isinf(obs).any():
                    print(f"[PCN] 警告: 観測データにNaN/Infが含まれています")
                    print(f"  NaN: {th.isnan(obs).any()}, Inf: {th.isinf(obs).any()}")
                    print(f"  観測データ範囲: min={obs.min()}, max={obs.max()}, mean={obs.mean()}")

                if th.isnan(desired_return).any() or th.isinf(desired_return).any():
                    print(f"[PCN] 警告: desired_returnにNaN/Infが含まれています")
                    print(f"  NaN: {th.isnan(desired_return).any()}, Inf: {th.isinf(desired_return).any()}")
                    print(f"  desired_return範囲: min={desired_return.min()}, max={desired_return.max()}, mean={desired_return.mean()}")

                if th.isnan(desired_horizon).any() or th.isinf(desired_horizon).any():
                    print(f"[PCN] 警告: desired_horizonにNaN/Infが含まれています")
                    print(f"  NaN: {th.isnan(desired_horizon).any()}, Inf: {th.isinf(desired_horizon).any()}")
                    print(f"  desired_horizon範囲: min={desired_horizon.min()}, max={desired_horizon.max()}, mean={desired_horizon.mean()}")

            # モデル推論
            try:
                if self.use_enhanced_model:
                    prediction_output = self.network(obs, desired_return, desired_horizon)
                else:
                    prediction_output = self.model(obs, desired_return, desired_horizon)
                if _PROFILE:
                    _now = time.perf_counter(); self._prof_acc["fwd"] += _now - _pt; _pt = _now
            except Exception as e:
                print(f"[PCN] エラー: モデル推論中にエラーが発生しました: {e}")
                print(f"  観測データ統計: min={obs.min()}, max={obs.max()}, mean={obs.mean()}")
                print(f"  desired_return統計: min={desired_return.min()}, max={desired_return.max()}, mean={desired_return.mean()}")
                print(f"  desired_horizon統計: min={desired_horizon.min()}, max={desired_horizon.max()}, mean={desired_horizon.mean()}")
                import traceback
                traceback.print_exc()
                # エラーの場合は損失を0に設定してスキップ
                l = th.tensor(0.0, device=self.device, requires_grad=False)
                return l, {}
            
            # 9. 損失計算
            if isinstance(prediction_output, tuple):
                prediction_logits = prediction_output[0]
            else:
                prediction_logits = prediction_output
            
            # モデル出力の検証（NaN/Infチェック）。既定では skip。
            # 出力 NaN→重みNaN化の安全網は backward 後に二重に存在する:
            #   AMP経路   = scaler.unscale_()/step() が Inf/NaN grad を検出して step をスキップ
            #   非AMP経路 = _NAN_SKIP_STEP(:3276) が th.isfinite(_gnorm) で step をスキップ
            # ゆえに本ブロック(巨大な診断 print 群 + 早期 return skip)は安全網が重複するので
            # _UPDATE_NANCHECK gate で囲み既定スキップ（クリーンrunでは未到達＝bit 一致, .any() 同期を除去）。
            if _UPDATE_NANCHECK and (th.isnan(prediction_logits).any() or th.isinf(prediction_logits).any()):
                print(f"[PCN] 警告: モデル出力にNaN/Infが含まれています")
                print(f"  NaN: {th.isnan(prediction_logits).any()}, Inf: {th.isinf(prediction_logits).any()}")
                print(f"  prediction_logits範囲: min={prediction_logits.min()}, max={prediction_logits.max()}")
                print(f"  観測データ統計: min={obs.min()}, max={obs.max()}, mean={obs.mean()}")
                print(f"  desired_return統計: min={desired_return.min()}, max={desired_return.max()}, mean={desired_return.mean()}")
                print(f"  desired_horizon統計: min={desired_horizon.min()}, max={desired_horizon.max()}, mean={desired_horizon.mean()}")

                # モデルの重みを確認
                if self.use_enhanced_model:
                    for name, param in self.network.named_parameters():
                        if th.isnan(param).any() or th.isinf(param).any():
                            print(f"  モデル重み {name} にNaN/Infが含まれています")
                            print(f"    範囲: min={param.min()}, max={param.max()}, mean={param.mean()}")
                else:
                    for name, param in self.model.named_parameters():
                        if th.isnan(param).any() or th.isinf(param).any():
                            print(f"  モデル重み {name} にNaN/Infが含まれています")
                            print(f"    範囲: min={param.min()}, max={param.max()}, mean={param.mean()}")
                print(f"  観測データ統計: min={obs.min()}, max={obs.max()}, mean={obs.mean()}")
                print(f"  desired_return統計: min={desired_return.min()}, max={desired_return.max()}, mean={desired_return.mean()}")
                print(f"  行動統計: {th.bincount(actions.long())}")
                # モデル出力がNaNの場合は、損失を0に設定してスキップ（勾配を計算しない）
                l = th.tensor(0.0, device=self.device, requires_grad=False)
                # 勾配を計算しないため、最適化をスキップ
                return l, {}

            if self.continuous_action:
                l = F.mse_loss(actions.float(), prediction_logits)
            else:
                metrics = {}
                if self.use_enhanced_model:
                    l = F.cross_entropy(prediction_logits, actions.long(), label_smoothing=_LABEL_SMOOTH)
                    pred_probs = F.softmax(prediction_logits, dim=-1)
                else:
                    # ②ラベル平滑化(既定0=従来とビット一致): (1-ε)·NLL + ε·一様CE。
                    # CE の最適解が p=1 でなくなり logit ギャップが有界化 → 飽和 lock-in を封印。
                    l = F.nll_loss(prediction_logits, actions.long())
                    if _LABEL_SMOOTH > 0.0:
                        l = (1.0 - _LABEL_SMOOTH) * l + _LABEL_SMOOTH * (-prediction_logits.mean(dim=-1)).mean()
                    pred_probs = th.exp(prediction_logits)
                # ①anchor-KL(既定0=OFF): イテレーション開始時の凍結方策からの逸脱を関数空間で罰する。
                if (
                    _ANCHOR_KL_WEIGHT > 0.0
                    and not self.use_enhanced_model
                    and self._anchor_model is not None
                ):
                    with th.no_grad():
                        _anch_logp = self._anchor_model(obs, desired_return, desired_horizon)
                        if isinstance(_anch_logp, tuple):
                            _anch_logp = _anch_logp[0]
                    if _anch_logp.shape == prediction_logits.shape and th.isfinite(_anch_logp).all():
                        _akl = (th.exp(_anch_logp) * (_anch_logp - prediction_logits)).sum(dim=-1).mean()
                        if th.isfinite(_akl):
                            l = l + _ANCHOR_KL_WEIGHT * _akl
                            if compute_metrics:
                                metrics["anchor_kl"] = float(_akl.detach().item())
                if (
                    _CONDITIONING_SENS_WEIGHT > 0.0
                    and not self.use_enhanced_model
                    and obs.shape[0] >= 4
                ):
                    sens_loss, sens_pairs = self._conditioning_sensitivity_loss(
                        obs, desired_return, desired_horizon
                    )
                    l = l + _CONDITIONING_SENS_WEIGHT * sens_loss
                    if compute_metrics:
                        metrics["conditioning_sens_loss"] = float(sens_loss.detach().item())
                        metrics["conditioning_sens_pairs"] = int(sens_pairs)
                if _MID_BAND_COND_WEIGHT > 0.0 and not self.use_enhanced_model and obs.shape[0] >= 2:
                    mid_loss, mid_pairs = self._mid_band_wait_conditioning_loss(
                        obs, desired_horizon
                    )
                    l = l + _MID_BAND_COND_WEIGHT * mid_loss
                    if compute_metrics:
                        metrics["mid_band_cond_loss"] = float(mid_loss.detach().item())
                        metrics["mid_band_cond_pairs"] = int(mid_pairs)
                if _LOW_BAND_COND_WEIGHT > 0.0 and not self.use_enhanced_model and obs.shape[0] >= 2:
                    low_loss, low_pairs = self._low_slope_conditioning_loss(
                        obs, desired_horizon
                    )
                    l = l + _LOW_BAND_COND_WEIGHT * low_loss
                    if compute_metrics:
                        metrics["low_band_cond_loss"] = float(low_loss.detach().item())
                        metrics["low_band_cond_pairs"] = int(low_pairs)
                if _VALUE_REPRO_WEIGHT > 0.0 and not self.use_enhanced_model:
                    val_loss = self._archive_value_reproduction_loss_from_replay()
                    if val_loss is not None and th.isfinite(val_loss):
                        l = l + _VALUE_REPRO_WEIGHT * val_loss
                        if compute_metrics:
                            metrics["value_repro_loss"] = float(val_loss.detach().item())
                # [案1: 指令追従loss] 指令costに達成が届かない側だけを片側MSEで罰す（value_head proxy/s.detach）。
                # gate OFF(=0)で _command_track_loss_from_replay を呼ばず np_random も引かない→ビット一致。
                if _CMD_TRACK_WEIGHT > 0.0 and not self.use_enhanced_model:
                    cmd_track_loss = self._command_track_loss_from_replay()
                    if cmd_track_loss is not None and th.isfinite(cmd_track_loss):
                        _ct_val = float(cmd_track_loss.detach().item())
                        if _CMD_TRACK_BREAKER > 0 and _ct_val > _CMD_TRACK_BREAKER:
                            # 暴走時はヒンジ(track)のみ遮断し、アンカー(価値ヘッド校正)は通す。
                            # 全遮断だとヘッドが永遠に校正されないデッドロック(smoke20c実測)。
                            self._cmd_track_tripped = getattr(self, "_cmd_track_tripped", 0) + 1
                            _split = getattr(self, "_cmd_track_split", None)
                            if _split is not None and th.isfinite(_split[1]):
                                l = l + _CMD_TRACK_WEIGHT * _CMD_TRACK_ANCHOR_WEIGHT * _split[1]
                            if self._cmd_track_tripped <= 5 or self._cmd_track_tripped % 200 == 0:
                                print(f"[CMD_TRACK_BREAKER] loss={_ct_val:.2f}>"
                                      f"{_CMD_TRACK_BREAKER:g} → ヒンジ不加算・アンカーのみ "
                                      f"(累計{self._cmd_track_tripped}回)", flush=True)
                        else:
                            l = l + _CMD_TRACK_WEIGHT * cmd_track_loss
                        if compute_metrics:
                            metrics["cmd_track_loss"] = _ct_val
                if compute_metrics:
                    predicted_actions = th.argmax(pred_probs, dim=-1)
                    metrics.update({
                        "policy_acc": float((predicted_actions == actions.long()).float().mean().item()),
                        "true_prob_mean": float(pred_probs.gather(1, actions.long().unsqueeze(1)).mean().item()),
                        "pred_entropy": float((-(pred_probs * th.log(pred_probs + 1e-8)).sum(dim=-1)).mean().item()),
                    })
            
            # 損失の検証
            if th.isnan(l) or th.isinf(l):
                print(f"[PCN] エラー: 損失がNaN/Infになりました")
                print(f"  損失値: {l.item()}")
                print(f"  観測データ統計: min={obs.min()}, max={obs.max()}, mean={obs.mean()}")
                print(f"  desired_return統計: min={desired_return.min()}, max={desired_return.max()}, mean={desired_return.mean()}")
                print(f"  prediction_logits統計: min={prediction_logits.min()}, max={prediction_logits.max()}, mean={prediction_logits.mean()}")
                print(f"  行動統計: {th.bincount(actions.long())}")
                # NaNの場合は損失を0に設定してスキップ（勾配を計算しない）
                l = th.tensor(0.0, device=self.device, requires_grad=False)
                # 勾配を計算しないため、最適化をスキップ
                return l, {}
            
            # 10. 逆伝播と最適化（混合精度対応）
            if self.use_amp and self.scaler is not None:
                # GradScalerの正しい使用方法:
                # 1. scale(loss).backward() - スケールされた損失で逆伝播
                # 2. unscale_(optimizer) - 勾配をunscale（Inf/NaNチェックも行う）
                # 3. clip_grad_norm_ - 勾配クリッピング（オプション）
                # 4. step(optimizer) - オプティマイザのステップ（unscale_()の後に必ず呼ぶ必要がある）
                # 5. update() - スケーラーの更新
                self.scaler.scale(l).backward()
                
                # unscale_()を呼んで勾配をunscale（Inf/NaNチェックも行う）
                # unscale_()が成功した場合、必ずstep()を呼ぶ必要がある
                try:
                    self.scaler.unscale_(self.opt)
                    # 勾配クリッピングを追加（勾配爆発を防ぐ）
                    th.nn.utils.clip_grad_norm_(self.network.parameters() if self.use_enhanced_model else self.model.parameters(), max_norm=1.0)
                    # unscale_()が成功した場合、必ずstep()を呼ぶ必要がある
                    # [凍結検知] GradScaler は inf/nan 勾配のとき step を無言でスキップし、
                    # そのとき scale を下げる。scale の低下で skip を判定して計上する
                    # (非AMP経路と同じ _nan_skip_total/_opt_step_total を埋め、
                    #  driver の [STEP_SKIP] 診断を AMP でも機能させる)。
                    _scale_before = float(self.scaler.get_scale())
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    if float(self.scaler.get_scale()) < _scale_before:
                        self._nan_skip_total += 1
                    else:
                        self._opt_step_total += 1
                    self._ema_update()
                except RuntimeError as e:
                    # unscale_()が既に呼ばれている場合、またはその他のエラーの場合
                    if "unscale_() has already been called" in str(e):
                        if self.debug_mode:
                            print(f"[PCN] 警告: unscale_()が既に呼ばれています。通常の最適化を実行します。")
                        # 勾配クリッピングのみ実行
                        th.nn.utils.clip_grad_norm_(self.network.parameters() if self.use_enhanced_model else self.model.parameters(), max_norm=1.0)
                        # 通常の最適化を実行（GradScalerを使わない）
                        self.opt.step()
                        self._ema_update()
                    elif "No inf checks were recorded" in str(e):
                        if self.debug_mode:
                            print(f"[PCN] 警告: Infチェックが記録されていません。通常の最適化を実行します。")
                        # 通常の最適化を実行（GradScalerを使わない）
                        self.opt.step()
                        self._ema_update()
                    else:
                        # その他のエラーの場合は再発生
                        raise
            else:
                global _nan_skip_count
                if _PROFILE:
                    _now = time.perf_counter(); self._prof_acc["loss"] += _now - _pt; _pt = _now
                l.backward()
                if _PROFILE:
                    _now = time.perf_counter(); self._prof_acc["bwd"] += _now - _pt; _pt = _now
                # 勾配クリッピングを追加（勾配爆発を防ぐ）。total_norm を捕捉して非有限なら step をスキップ。
                _gnorm = th.nn.utils.clip_grad_norm_(self.network.parameters() if self.use_enhanced_model else self.model.parameters(), max_norm=1.0)
                _lval = float(l.detach())
                _spike = (_LOSS_SPIKE_SKIP and self._loss_ema is not None
                          and np.isfinite(_lval) and _lval > self._loss_ema * _LOSS_SPIKE_RATIO)
                if _NAN_SKIP_STEP and not bool(th.isfinite(_gnorm)):
                    # 非有限な勾配: step をスキップし勾配を捨てる（NaN が重みに焼き付く永久崩壊を防ぐ）
                    self.opt.zero_grad(set_to_none=True)
                    self._nan_skip_total += 1
                    if _nan_skip_count < _NAN_SKIP_WARN_LIMIT:
                        _nan_skip_count += 1
                        print(f"[PCN] 非有限な勾配 grad_norm={float(_gnorm)} → step スキップ（重みのNaN化を防止）")
                elif _spike:
                    # 損失スパイク: 発散した勾配で重みを壊さないよう step をスキップ（前半の良い方策を守る）
                    self.opt.zero_grad(set_to_none=True)
                    self._nan_skip_total += 1
                    if _nan_skip_count < _NAN_SKIP_WARN_LIMIT:
                        _nan_skip_count += 1
                        print(f"[PCN] 損失スパイク loss={_lval:.1f} > {_LOSS_SPIKE_RATIO}×ema({self._loss_ema:.1f}) → step スキップ（発散制動）")
                else:
                    self.opt.step()
                    self._opt_step_total += 1
                    self._ema_update()
                # 損失移動平均の更新。スパイク時も cap値(ema×RATIO)で緩やかに追従させる:
                # フェーズ移行等で損失レジームが変わったときに永久skipデッドロックにならず数十updateで追いつき、
                # 一過性スパイク(発散の初動)だけを捨てる。
                if np.isfinite(_lval):
                    if self._loss_ema is None:
                        self._loss_ema = _lval
                    else:
                        _track = min(_lval, self._loss_ema * _LOSS_SPIKE_RATIO)
                        self._loss_ema = 0.9 * self._loss_ema + 0.1 * _track

        if _PROFILE:
            # backward後〜ここまで(clip+opt.step+ema)を opt に計上。AMP経路でも loss後の残り時間を回収。
            self._prof_acc["opt"] += time.perf_counter() - _pt
        # 11. メモリクリーンアップ
        del observations, actions, desired_returns, desired_horizons
        if str(self.device).startswith("cuda") and os.environ.get("PCN_EMPTY_CACHE_EVERY_UPDATE", "0") == "1":
            th.cuda.empty_cache()
        
        # 12. パフォーマンス監視
        end_time = time.time()
        update_time = end_time - start_time
        self.update_times.append(update_time)
        
        # 学習率を元に戻す
        if original_lr is not None:
            self.opt.param_groups[0]['lr'] = original_lr

        # 詳細なデバッグ情報（100回ごと）
        if len(self.update_times) % 100 == 0 and self.debug_mode:
            avg_time = np.mean(self.update_times[-100:])
            print(f"Update performance: {avg_time:.4f}s per update (avg of last 100)")

        return l, metrics

    def update_many(self, n_updates: int, learning_rate=None):
        """Phase3/Phase2: 複数 update を連続実行（Python/Ray 往復オーバーヘッド削減）。

        _FAST_UPDATE 時（既定）は、この呼び出し中だけ experience_replay が凍結である事を使い、
        (1) Archive 帯域 command の memo、(2) 破棄される診断 metrics を最終 update のみ計算、
        (3) per-update の loss.item() 同期をまとめて 1 回に、を行う。数式・乱数・更新回数は不変。
        """
        if n_updates <= 0:
            return 0.0, {}, []
        if not _FAST_UPDATE:
            losses = []
            last_metrics: Dict[str, Any] = {}
            for _ in range(n_updates):
                loss, metrics = self.update(learning_rate=learning_rate)
                loss_value = loss.item() if hasattr(loss, "item") else float(loss)
                if np.isnan(loss_value) or np.isinf(loss_value):
                    loss_value = 0.0
                losses.append(loss_value)
                if isinstance(metrics, dict) and metrics:
                    last_metrics = metrics
            return float(np.mean(losses)), last_metrics, losses

        # --- fast path（結果ビット一致）---
        self._cond_pool_cache = {}  # update_many スコープ限定 memo（replay 凍結期間のみ有効）
        last_metrics = {}
        loss_tensors = []
        try:
            for i in range(n_updates):
                compute_metrics = i == n_updates - 1
                loss, metrics = self.update(
                    learning_rate=learning_rate, compute_metrics=compute_metrics
                )
                # grad graph を保持しないよう detach（100 個保持してもメモリ増を防ぐ）
                loss_tensors.append(loss.detach() if hasattr(loss, "detach") else loss)
                if isinstance(metrics, dict) and metrics:
                    last_metrics = metrics
        finally:
            self._cond_pool_cache = None  # memo を必ず破棄（次イテレーションへ持ち越さない）
        # per-update loss を 1 回の転送でまとめて取得（同期 100→1）。非有限は旧挙動どおり 0。
        if loss_tensors and hasattr(loss_tensors[0], "detach"):
            arr = th.stack([t.reshape(()) for t in loss_tensors]).to("cpu", dtype=th.float64).numpy()
        else:
            arr = np.asarray([float(t) for t in loss_tensors], dtype=np.float64)
        arr[~np.isfinite(arr)] = 0.0
        losses = arr.tolist()
        if _PROFILE and getattr(self, "_prof_acc", None) and self._prof_acc["n"] > 0:
            a = self._prof_acc; tot = a["batch"] + a["fwd"] + a["loss"] + a["bwd"] + a["opt"]
            if tot > 0:
                print(f"[PROFILE update内訳] n={a['n']} 計{tot:.2f}s | "
                      f"batch {a['batch']/tot*100:.0f}% / forward {a['fwd']/tot*100:.0f}% / "
                      f"loss {a['loss']/tot*100:.0f}% / backward {a['bwd']/tot*100:.0f}% / opt {a['opt']/tot*100:.0f}% "
                      f"(1更新 fwd={a['fwd']/a['n']*1000:.2f}ms bwd={a['bwd']/a['n']*1000:.2f}ms opt={a['opt']/a['n']*1000:.2f}ms)", flush=True)
            self._prof_acc = {"batch": 0.0, "fwd": 0.0, "loss": 0.0, "bwd": 0.0, "opt": 0.0, "n": 0}
        return float(arr.mean()) if len(arr) else 0.0, last_metrics, losses

    def _add_episode(self, transitions: List[Transition], max_size: int, step: int) -> None:
        # compute return
        for i in reversed(range(len(transitions) - 1)):
            transitions[i].reward += self.gamma * transitions[i + 1].reward
        # pop smallest episode of heap if full, add new episode
        # heap is sorted by negative distance, (updated in nlargest)
        # put positive number to ensure that new item stays in the heap
        unique_step = (step, id(transitions))
        self._replay_push((1, unique_step, transitions), max_size)

    def _replay_push(self, entry, max_size: int) -> bool:
        """experience_replay へ entry=(priority, unique_step, transitions) を追加する唯一の入口。

        満杯時の淘汰は既定 heappushpop（従来と完全ビット一致）。PCN_REPLAY_REGIME_FAIR=1 の
        ときのみレジーム公平淘汰(_replay_push_regime_fair)へ切替。戻り値: entry がバッファに残ったか。
        """
        if len(self.experience_replay) >= max_size:  # [修正] ==だとmax_sizeが揺れた時に淘汰が永久に発火しない
            if _REPLAY_REGIME_FAIR:
                return self._replay_push_regime_fair(entry)
            popped = heapq.heappushpop(self.experience_replay, entry)
            return popped is not entry
        heapq.heappush(self.experience_replay, entry)
        return True

    def _replay_push_regime_fair(self, entry) -> bool:
        """[PCN_REPLAY_REGIME_FAIR] レジーム別クォータ淘汰（満杯時のみ呼ばれる）。

        グローバル heappushpop はレジーム盲目（(cost,wait)で見かけ優秀な最空きレジームが
        バッファを占拠）なので、①クォータ=(バッファ+新規1)/観測レジーム数の均等割、
        ②最もクォータ超過したレジームから追い出す、③レジーム内の追い出し順は
        「非種優先 → priority昇順(_nlargestがレジーム内crowdingで更新) → 古い順」、
        ④新規エピソード自身も自レジームが対象なら淘汰候補(=従来 heappushpop と同じ意味論)。
        種エピソードは同レジームに非種が残る限り保護（全滅ケースでは種も淘汰=デッドロックなし）。
        """
        buf = self.experience_replay
        regimes = [episode_regime_scale(e[2]) for e in buf]
        new_regime = episode_regime_scale(entry[2])
        counts: Dict[float, int] = {}
        for r in regimes:
            counts[r] = counts.get(r, 0) + 1
        counts[new_regime] = counts.get(new_regime, 0) + 1
        quota = (len(buf) + 1) / max(1, len(counts))
        new_is_seed = episode_is_seed(entry[2])
        # 超過が大きいレジーム順（同率はスケール値昇順で決定的に）
        for scale, cnt in sorted(counts.items(), key=lambda kv: (-(kv[1] - quota), kv[0])):
            cand = [i for i, r in enumerate(regimes) if r == scale]
            include_new = scale == new_regime
            non_seed = [i for i in cand if not episode_is_seed(buf[i][2])]
            if non_seed or (include_new and not new_is_seed):
                # 種保護: 非種の候補が存在する限り種は追い出さない
                cand = non_seed
                include_new = include_new and not new_is_seed
            if not cand and not include_new:
                continue
            evict_i = min(cand, key=lambda i: (buf[i][0], buf[i][1])) if cand else None
            if include_new and (
                evict_i is None or (entry[0], entry[1]) < (buf[evict_i][0], buf[evict_i][1])
            ):
                return False  # 新規自身が最弱: 従来 heappushpop と同じく「入れない」
            buf[evict_i] = entry
            heapq.heapify(buf)
            return True
        # 理論上到達しない全滅ケース: 従来淘汰へフォールバック
        popped = heapq.heappushpop(buf, entry)
        return popped is not entry

    def refresh_replay_priorities(self) -> int:
        """[2026-08-26] replay ヒープの優先度を「非支配 + crowding distance」で更新する。

        なぜ必要か: 淘汰(_replay_push の heappushpop)は優先度の最小を捨てる。ところが
        _add_episode は priority=1 の固定値で push しており、優先度を実際に書き戻すのは
        _nlargest だけである。本番の指令選択は PCN_CHOOSE_COMMANDS_MODE=pf_mixed のとき
        _choose_commands_batch_from_pf で早期 return するため **_nlargest に到達せず**、
        全エントリの優先度が 1 のまま=淘汰が unique_step 順(=古い順の FIFO)になる。

        FIFO 淘汰は Phase1 のランダム掃引(p=0..1、PF の両端を含む)から先に捨てるため、
        反復を重ねるほどアーカイブが「現在の方策の周辺」に凝縮する。これは原論文 §4.4 が
        明示的に警告している失敗モード(「カバレッジ集合だけに絞ると、似た軌跡ばかりを
        集めて少数の方策に凝縮し学習が壊れる」)そのものである。

        本メソッドは _nlargest と同じ規則で優先度だけを更新する(非支配点は
        crowding distance + 1.0、被支配点は 0)。戻り値=非支配点の数。
        """
        if len(self.experience_replay) == 0:
            return 0
        valid_indices = [i for i, e in enumerate(self.experience_replay) if len(e[2]) > 0]
        if not valid_indices:
            return 0
        valid_returns = np.array(
            [self.experience_replay[i][2][0].reward for i in valid_indices])
        non_dominated_i = get_non_dominated_inds(valid_returns)
        if len(non_dominated_i) == 0:
            return 0
        _lo = valid_returns.min(axis=0)
        _span = np.maximum(valid_returns.max(axis=0) - _lo, 1e-12)
        valid_returns_norm = (valid_returns - _lo) / _span
        # [論文 Eq.6-7 忠実] I_l2,i = -min_j ||p_i - p_j||, p_j ∈ 非支配集合(正規化座標)。
        #  - 非支配かつ疎(CD>0.2): I_ds = I_l2 = 0(最高=最後まで残る)
        #  - 非支配だが密集/重複(CD≤0.2): I_ds = 2(I_l2 - c) = -2c(僅かに負=降格)
        #  - 被支配: I_ds = I_l2 = -距離(フロントに近いほど 0 に近い=多様性として残る)
        # 前版の「非支配=CD+1.0 / 被支配=一律0」は、被支配の中の淘汰が FIFO になり
        # 「フロント近傍の多様性を残す」という §4.4 の意図(too few V-values の回避)を
        # 半分しか実現していなかったため、論文式に置き換えた。
        _c = 1e-5
        nd_pts = valid_returns_norm[non_dominated_i]
        d2 = ((valid_returns_norm[:, None, :] - nd_pts[None, :, :]) ** 2).sum(-1)
        min_dist = np.sqrt(d2.min(axis=1))                      # 各点→最近傍ND距離
        priorities = -min_dist                                   # 被支配: I_l2(負)
        cd = crowding_distance(nd_pts)
        nd_arr = np.asarray(non_dominated_i)
        priorities[nd_arr[cd > 0.2]] = 0.0                       # 疎なND: 0
        priorities[nd_arr[cd <= 0.2]] = -2.0 * _c                # 密集ND: 降格
        for local_i, global_i in enumerate(valid_indices):
            _, step, episode = self.experience_replay[global_i]
            self.experience_replay[global_i] = (priorities[local_i], step, episode)
        heapq.heapify(self.experience_replay)
        return int(len(non_dominated_i))

    def _nlargest(self, n, threshold=0.2):
        """PF上の疎な点を優先して上位n個を選別する。

        PCNのコマンド生成は、既存の非支配解を少し改善した目標を作るため、
        まず非支配解を候補にし、その中で crowding distance が大きい
        （密集していない）点を優先する。
        """
        if len(self.experience_replay) == 0:
            print("警告: 経験再生バッファが空です。")
            return []

        valid_indices = [i for i, e in enumerate(self.experience_replay) if len(e[2]) > 0]
        if len(valid_indices) == 0:
            print("警告: 有効なエピソードが見つかりません。")
            return []

        valid_returns = np.array([self.experience_replay[i][2][0].reward for i in valid_indices])
        non_dominated_i = get_non_dominated_inds(valid_returns)
        if len(non_dominated_i) == 0:
            # 退化時フォールバック（論文実装の意図を崩さない範囲で安全化）
            take = min(n, len(valid_indices))
            return [self.experience_replay[i] for i in valid_indices[-take:]]

        selected_valid_local = []

        # 1. 非支配解の中から、密集していない点を優先して選ぶ。
        non_dominated = valid_returns[non_dominated_i]
        valid_returns_norm = self._normalize_points_for_selection(valid_returns)
        non_dominated_norm = valid_returns_norm[non_dominated_i]
        nd_distances = crowding_distance(non_dominated_norm)
        nd_order = np.argsort(nd_distances)[::-1]
        n_from_pf = min(n, len(nd_order))
        selected_valid_local.extend(non_dominated_i[nd_order[:n_from_pf]].tolist())

        # 2. PFだけで足りない場合のみ、PF近傍の点で補う。
        if len(selected_valid_local) < n:
            returns_exp = np.repeat(valid_returns_norm[:, None], len(non_dominated_norm), axis=1)
            nd_exp = np.repeat(non_dominated_norm[None], len(valid_returns_norm), axis=0)
            l2 = np.min(np.linalg.norm(returns_exp - nd_exp, axis=-1), axis=-1)
            selected_mask = np.zeros(len(valid_returns), dtype=bool)
            selected_mask[selected_valid_local] = True
            dominated_candidates = np.where(~selected_mask)[0]
            if len(dominated_candidates) > 0:
                # PFに近い点を優先し、同距離ではcrowdingが大きい点を残す。
                all_distances = crowding_distance(valid_returns_norm)
                order = sorted(
                    dominated_candidates.tolist(),
                    key=lambda idx: (l2[idx], -all_distances[idx]),
                )
                remaining = n - len(selected_valid_local)
                selected_valid_local.extend(order[:remaining])

        selected_global_indices = [valid_indices[i] for i in selected_valid_local]
        largest = [self.experience_replay[i] for i in selected_global_indices]

        # ヒープ優先度を更新: PF上の疎な点ほど残りやすくする。
        priorities = np.zeros(len(valid_returns), dtype=np.float64)
        if _REPLAY_REGIME_FAIR:
            # [PCN_REPLAY_REGIME_FAIR] 優先度をレジーム内で計算する。グローバル(cost,wait)座標の
            # ND/crowding は最空きレジームが独占し、評価レジームのPF見本が優先度0で淘汰される
            # (レジーム盲目)。各レジーム内のND点に「レジーム内crowding+1」を与え、レジームごとの
            # フロント構造を保護する。返り値(コマンド生成用の選抜)は従来どおりグローバル基準のまま。
            ep_scales = np.array(
                [episode_regime_scale(self.experience_replay[i][2]) for i in valid_indices]
            )
            for _scale in np.unique(ep_scales):
                g = np.where(ep_scales == _scale)[0]
                g_nd = get_non_dominated_inds(valid_returns[g])
                if len(g_nd) == 0:
                    continue
                g_cd = crowding_distance(valid_returns_norm[g][g_nd])
                priorities[g[g_nd]] = g_cd + 1.0
        else:
            priorities[non_dominated_i] = nd_distances + 1.0
        for local_i, global_i in enumerate(valid_indices):
            _, step, episode = self.experience_replay[global_i]
            self.experience_replay[global_i] = (priorities[local_i], step, episode)
        heapq.heapify(self.experience_replay)
        return largest

    def _local_step_pick(self, front: np.ndarray):
        """[PCN_CMD_LOCAL_STEP] cost昇順の前線 front (K×2, (cost,wait)) から局所ステップ注文を1つ作る。

        返り値: ((cost, wait), base_index)。
        - 内側ステップ: 土台 p_i 一様 → 隣接点 p_{i±1} 方向へ α~U(0.5,1.5) の両軸内挿/外挿
        - 端ステップ(確率 _CMD_LOCAL_EDGE_FRAC): 端点から内側隣接間隔と同じ幅で外側へ(前線を伸ばす)
        - クランプ: 達成済みレンジ±端の隣接間隔、かつ非負
        """
        front = np.asarray(front, dtype=np.float64)
        K = len(front)
        if K == 1:
            return (float(front[0, 0]), float(front[0, 1])), 0
        rng = self.np_random
        if rng.uniform() < _CMD_LOCAL_EDGE_FRAC:
            # 端点の外側への一歩: 内側の隣接点から遠ざかる方向(幅は内側間隔×U(0.5,1.5))
            i = 0 if int(rng.integers(0, 2)) == 0 else K - 1
            j = 1 if i == 0 else K - 2
            alpha = -float(rng.uniform(0.5, 1.5))
        else:
            i = int(rng.integers(0, K))
            if i == 0:
                j = 1
            elif i == K - 1:
                j = K - 2
            else:
                j = i + 1 if int(rng.integers(0, 2)) == 0 else i - 1
            alpha = float(rng.uniform(0.5, 1.5))
        p = front[i] + alpha * (front[j] - front[i])
        # クランプ: 各軸とも達成済みレンジを「その端の隣接間隔」以上は超えない
        d_lo = np.abs(front[1] - front[0])    # cost下端(=wait上端)側の隣接間隔
        d_hi = np.abs(front[-1] - front[-2])  # cost上端(=wait下端)側の隣接間隔
        c_min, c_max = float(front[0, 0]), float(front[-1, 0])
        w_min, w_max = float(front[:, 1].min()), float(front[:, 1].max())
        lo = np.array([c_min - d_lo[0], w_min - d_hi[1]])
        hi = np.array([c_max + d_hi[0], w_max + d_lo[1]])
        if _CMD_REACH_CLAMP > 1.0:
            # [PCN_CMD_REACH_CLAMP] 到達済み端×係数で外挿を頭打ちに(島ギャップ由来の暴走を防ぐ)
            f = _CMD_REACH_CLAMP
            hi[0] = min(hi[0], c_max * f)
            hi[1] = min(hi[1], w_max * f)
            lo[1] = max(lo[1], w_min / f)
        p = np.minimum(np.maximum(p, lo), hi)
        p = np.maximum(p, 0.0)
        return (float(p[0]), float(p[1])), int(i)

    def _local_step_choose_group(self, groups):
        """[PCN_CMD_LOCAL_STEP] 前線グループを点数比例で1つ選ぶ(=全点の一様選択と等価)。"""
        if len(groups) == 1:
            return 0
        sizes = np.array([len(g[0]) for g in groups], dtype=np.float64)
        return int(self.np_random.choice(len(groups), p=sizes / sizes.sum()))

    def _choose_commands_local_step(self, num_episodes: int, n_commands: int):
        """[PCN_CMD_LOCAL_STEP] 論文式(_choose_commands系)の代替: 局所ステップ注文をn個作る。

        replay の非支配前線を(レジーム標識があればレジーム別に)cost昇順に並べ、
        _local_step_pick で隣接間隔ベースの注文を作る。horizon は土台エピソードの
        horizon-2(現行と同じ規約)。返り値: [(desired_return, desired_horizon, base_return), ...]
        エピソードが無ければ None(呼び出し側で従来のデフォルト処理へ)。
        """
        episodes = self._nlargest(num_episodes)
        if len(episodes) == 0:
            return None
        returns = np.array([e[2][0].reward for e in episodes], dtype=np.float64)
        horizons = np.array([len(e[2]) for e in episodes], dtype=np.float64)
        regimes = np.array([episode_regime_scale(e[2]) for e in episodes], dtype=np.float64)
        groups = []  # (front K×2 (cost,total_wait) cost昇順, horizons K)
        for scale in np.unique(regimes):
            m = regimes == scale
            nd = get_non_dominated_inds(returns[m])
            if len(nd) == 0:
                continue
            r = returns[m][nd]
            h = horizons[m][nd]
            # (cost, total_wait) = (-r1, -r0) に変換
            pts = np.column_stack([-r[:, 1], -r[:, 0]])
            # 重複点を除去して cost 昇順へ(np.unique(axis=0)=辞書式ソート=cost昇順)。
            # 全オンプレ端などの同一達成点が複数あると隣接間隔=0になり、局所ステップが
            # 「土台と同一の注文」に退化+端を伸ばす一歩も0幅化するため(2026-08-06 実測)。
            _, ui = np.unique(np.round(pts, decimals=6), axis=0, return_index=True)
            groups.append((pts[ui], h[ui]))
        if not groups:
            return None
        results = []
        for _ in range(n_commands):
            front, hz = groups[self._local_step_choose_group(groups)]
            (c, w), bi = self._local_step_pick(front)
            desired_return = np.array([-w, -c], dtype=np.float32)
            desired_horizon = np.float32(hz[bi] - 2)
            base_return = np.array([-front[bi, 1], -front[bi, 0]], dtype=np.float32)
            results.append((desired_return, desired_horizon, base_return))
        return results

    def _local_step_picks_from_pf(self, pf_pool: np.ndarray, n_commands: int):
        """[PCN_CMD_LOCAL_STEP] 本番系(_choose_commands_batch_from_pf)の代替pick生成。

        レジーム標識が複数あれば archive をレジーム別 PF に分けて局所ステップ
        (ref_pf/anchor 混合済みの pf_pool はレジーム不明のため追加グループとして保持)。
        単一レジームなら pf_pool そのものを前線として使う。返り値: [(cost, wait), ...]
        """
        groups = []
        entries = self._valid_replay_entries()
        if entries:
            regimes = np.array([episode_regime_scale(e[2]) for e in entries])
            uniq = np.unique(regimes)
            if len(uniq) > 1:
                for scale in uniq:
                    sub = [e for e, r_ in zip(entries, regimes) if r_ == scale]
                    pf = self._archive_pf_objective_points(sub)
                    if pf.size:
                        o = np.argsort(pf[:, 0], kind="stable")
                        groups.append((pf[o], None))
        if not groups:
            pool = np.asarray(pf_pool, dtype=np.float64)
            pool = np.unique(np.round(pool, decimals=9), axis=0)
            o = np.argsort(pool[:, 0], kind="stable")
            groups.append((pool[o], None))
        picks = []
        for _ in range(n_commands):
            front, _ = groups[self._local_step_choose_group(groups)]
            (c, w), _bi = self._local_step_pick(front)
            picks.append((c, w))
        return picks

    def _choose_commands(self, num_episodes: int):
        """論文実装準拠のコマンド選択。"""
        if _CMD_LOCAL_STEP:
            cmds = self._choose_commands_local_step(num_episodes, 1)
            if cmds is not None:
                return cmds[0][0], cmds[0][1]
        episodes = self._nlargest(num_episodes)
        if len(episodes) == 0:
            print("警告: コマンド選択用のエピソードが見つかりませんでした。デフォルト値を返します。")
            return np.zeros(self.reward_dim, dtype=np.float32), np.float32(40)

        returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
        nd_i = get_non_dominated_inds(np.array(returns))
        returns = np.array(returns)[nd_i]
        horizons = np.array(horizons)[nd_i]
        if len(returns) == 0:
            return np.zeros(self.reward_dim, dtype=np.float32), np.float32(40)

        # ランダムな非支配解を1つ選ぶ
        r_i = self.np_random.integers(0, len(returns))
        desired_horizon = np.float32(horizons[r_i] - 2)

        # 論文実装どおり: 1目的のみ、標準偏差範囲で上乗せ
        _, s = np.mean(returns, axis=0), np.std(returns, axis=0)
        desired_return = returns[r_i].copy()
        r_obj = self.np_random.integers(0, len(desired_return))
        desired_return[r_obj] += self.np_random.uniform(high=s[r_obj] * _COMMAND_ALPHA)
        desired_return = np.float32(desired_return)
        return desired_return, desired_horizon

    def _archive_pf_objective_points(self, entries=None) -> np.ndarray:
        pts = self._archive_objective_points(entries)
        if pts.size == 0:
            return pts
        pf_i = get_non_dominated_inds_minimize(pts)
        if len(pf_i) == 0:
            return np.empty((0, 2), dtype=np.float64)
        pf = pts[pf_i]
        return np.unique(np.round(pf, decimals=3), axis=0)

    def _ref_pf_scale_guard(self, ref_pf: np.ndarray, ref_path: str):
        """[REF_PF] スケール不整合ガード: 参照前線の cost/wait レンジが自分の達成レンジと
        桁違い(中央値比 > PCN_REF_PF_GUARD_RATIO, 既定10x)なら警告してスキップする。

        別ワークロード(例: trace24)の桁違いに小さい前線が非支配比較で自分の達成点を
        押し出し、注文プールを乗っ取る事故(2026-08-06 確定バグ)の再発防止。
        自分の達成点(archive PF)が未形成の間は比較不能のためガードは保留する。
        PCN_REF_PF_GUARD_RATIO=0 でガード無効(旧挙動退避)。"""
        try:
            ratio_max = float(os.environ.get("PCN_REF_PF_GUARD_RATIO", "10") or "0")
        except ValueError:
            ratio_max = 10.0
        if ratio_max <= 0 or ref_pf is None or len(ref_pf) == 0:
            return ref_pf
        own = self._archive_pf_objective_points()
        if own is None or own.size == 0:
            return ref_pf
        eps = 1e-9
        bad_axes = []
        for ax, name in ((0, "cost"), (1, "wait")):
            m_own = max(float(np.median(own[:, ax])), eps)
            m_ref = max(float(np.median(np.asarray(ref_pf)[:, ax])), eps)
            r = max(m_own / m_ref, m_ref / m_own)
            if r > ratio_max:
                bad_axes.append(f"{name} 中央値比 {r:.1f}x (own={m_own:.3g} ref={m_ref:.3g})")
        if not bad_axes:
            return ref_pf
        if not getattr(self, "_ref_pf_guard_warned", False):
            self._ref_pf_guard_warned = True
            print(
                f"[REF_PF] スケール不整合ガード発動: {ref_path} をスキップ "
                f"({'; '.join(bad_axes)} > {ratio_max:g}x)。参照前線は command pool へ"
                "混ぜません(PCN_REF_PF_GUARD_RATIO=0 で無効化可)。",
                flush=True,
            )
        return None

    def _pf_command_horizon(self, cost: float, wait: float, entries=None) -> np.float32:
        """PF 点に近い archive エピソード長を horizon に使う。"""
        best_hz = None
        best_dist = float("inf")
        for entry in entries or self._valid_replay_entries():
            episode = entry[2]
            if not episode:
                continue
            first = episode[0]
            if not (hasattr(first, "objective_values") and first.objective_values is not None):
                continue
            obj = first.objective_values
            c, w = float(obj[0]), float(obj[2])
            d = abs(c - cost) + abs(w - wait) * 1e-3
            if d < best_dist:
                best_dist = d
                best_hz = float(max(1, len(episode) - 2))
        nj = max(1, self._policy_n_jobs())
        return np.float32(best_hz if best_hz is not None else nj)

    def _anchor_command_points(self, n_anchors: int) -> np.ndarray:
        """calibration 端点 (全OP, 全CL) を結ぶ線分上の固定 anchor (cost, wait)。

        archive PF が崩壊しても command pool が cost 全域を張れるよう毎回 pool へ混ぜる。
        端点は workload calibration が設定する env から読み、無ければ archive レンジへ
        フォールバックする。これは決定論的（再現性を壊さない）で、追加学習でもない。
        """
        if n_anchors <= 0:
            return np.empty((0, 2), dtype=np.float64)

        def _envf(k):
            v = os.environ.get(k)
            try:
                return float(v) if v is not None and v != "" else None
            except ValueError:
                return None

        cost_op = _envf("PCN_WORKLOAD_COST_OP")
        cost_cl = _envf("PCN_WORKLOAD_COST_CL")
        wait_op = _envf("PCN_WORKLOAD_WAIT_OP")
        wait_cl = _envf("PCN_WORKLOAD_WAIT_CL")
        if None in (cost_op, cost_cl, wait_op, wait_cl):
            pts = self._archive_objective_points()
            if pts.size == 0:
                return np.empty((0, 2), dtype=np.float64)
            i_lo = int(np.argmin(pts[:, 0]))
            i_hi = int(np.argmax(pts[:, 0]))
            cost_op, cost_cl = float(pts[i_lo, 0]), float(pts[i_hi, 0])
            wait_op, wait_cl = float(pts[i_lo, 1]), float(pts[i_hi, 1])
        if abs(cost_cl - cost_op) < 1e-9:
            return np.empty((0, 2), dtype=np.float64)
        n = max(2, int(n_anchors))
        fr = np.linspace(0.0, 1.0, n)
        costs = cost_op + fr * (cost_cl - cost_op)
        waits = wait_op + fr * (wait_cl - wait_op)
        return np.column_stack([
            np.maximum(costs, 0.0), np.maximum(waits, 0.0)
        ]).astype(np.float64)

    def _choose_commands_batch_from_pf(self, n_commands: int):
        """参照 PF + archive PF を cost 層化サンプルして command 化（batch 内で cost 被覆）。"""
        import os
        from pathlib import Path

        from src.utils.pf_command_eval import load_ref_pf, merge_pf_pools, stratified_sample_pf

        default_cmd = (
            np.zeros(self.reward_dim, dtype=np.float32),
            np.float32(40),
            np.zeros(self.reward_dim, dtype=np.float32),
        )
        nj = self._policy_n_jobs()
        jitter = float(os.environ.get("PCN_PF_COMMAND_JITTER", "0"))
        low_wait_frac = float(os.environ.get("PCN_PF_COMMAND_LOW_WAIT_FRAC", "0"))
        low_wait_quota = int(os.environ.get("PCN_PF_COMMAND_LOW_WAIT_QUOTA", "0"))
        low_wait_max = float(os.environ.get("PCN_PF_COMMAND_LOW_WAIT_MAX", "0"))
        include_extremes = os.environ.get("PCN_PF_COMMAND_INCLUDE_EXTREMES", "1") != "0"
        mode = os.environ.get("PCN_CHOOSE_COMMANDS_MODE", "paper").strip().lower()
        ref_path = os.environ.get("PCN_REF_PF_NPZ", "").strip()
        ref_pf = None
        if ref_path and os.path.isfile(ref_path):
            try:
                ref_pf = load_ref_pf(Path(ref_path))
            except Exception as e:
                print(f"警告: PCN_REF_PF_NPZ 読み込み失敗 ({ref_path}): {e}")
        if ref_pf is not None and not getattr(self, "_ref_pf_logged", False):
            # [REF_PF] オプトイン使用の明示ログ(起動後の初回のみ)。既定では REF_PF は
            # 未設定=混入なし(2026-08-06 バグ修正: workload profile の既定セットを廃止)。
            self._ref_pf_logged = True
            print(
                f"[REF_PF] 使用: {ref_path} 点数={len(ref_pf)} "
                f"cost=[{ref_pf[:, 0].min():.3g}, {ref_pf[:, 0].max():.3g}] "
                f"wait=[{ref_pf[:, 1].min():.3g}, {ref_pf[:, 1].max():.3g}]",
                flush=True,
            )
        if ref_pf is not None:
            ref_pf = self._ref_pf_scale_guard(ref_pf, ref_path)
        if mode == "pf_ref":
            pools = [ref_pf] if ref_pf is not None else []
        elif mode in ("pf_mixed", "pf_stratified", "pf_archive", "gap", "coverage"):
            pools = [self._archive_pf_objective_points()]
            if ref_pf is not None:
                pools.append(ref_pf)
        else:
            pools = [self._archive_pf_objective_points()]
        pf_pool = merge_pf_pools(*pools)
        if _PF_COMMAND_ANCHORS > 0:
            anchors = self._anchor_command_points(_PF_COMMAND_ANCHORS)
            if anchors.size:
                pf_pool = merge_pf_pools(pf_pool, anchors) if pf_pool.size else anchors
        if pf_pool.size == 0:
            print("警告: PF command pool が空です。デフォルト command を返します。")
            return [default_cmd] * n_commands

        if _CMD_LOCAL_STEP:
            # [PCN_CMD_LOCAL_STEP] pick生成を隣接間隔ベースの局所ステップへ置換(他は従来どおり)
            picks = self._local_step_picks_from_pf(pf_pool, n_commands)
        elif mode == "gap":
            picks = self._gap_directed_sample(pf_pool, n_commands)
        elif mode == "coverage":
            picks = self._coverage_directed_sample(
                pf_pool, n_commands,
                low_wait_frac=low_wait_frac, low_wait_quota=low_wait_quota,
                low_wait_max=low_wait_max, include_extremes=include_extremes)
        else:
            picks = stratified_sample_pf(
                pf_pool,
                n_commands,
                rng=self.np_random,
                low_wait_frac=low_wait_frac,
                low_wait_quota=low_wait_quota,
                low_wait_max=low_wait_max,
                include_extremes=include_extremes,
            )
        # [PCN_CMD_REACH_CLAMP] 注文を「到達済み(archive PF)の端 × 係数」で頭打ちにする(既定OFF)。
        # anchor(全クラウド端=真PF上限の2.3倍)や端外挿が実現不能域の注文を作ると、達成は頭打ちに
        # なるので注文と達成のズレだけが増える。到達幅(archive の端)は残したまま外側だけ削る。
        if _CMD_REACH_CLAMP > 1.0 and picks:
            _ach = np.asarray(pools[0], dtype=np.float64) if pools and pools[0] is not None else None
            if _ach is not None and _ach.size:
                _cmax = float(_ach[:, 0].max()) * _CMD_REACH_CLAMP
                _wmax = float(_ach[:, 1].max()) * _CMD_REACH_CLAMP
                if not getattr(self, "_reach_clamp_logged", False):
                    self._reach_clamp_logged = True
                    print(f"[CMD_REACH_CLAMP] x{_CMD_REACH_CLAMP:g} 注文上限 cost<={_cmax:.4g} "
                          f"wait<={_wmax:.4g} (到達済みarchive端基準)", flush=True)
                picks = [(min(float(c), _cmax), min(float(w), _wmax)) for c, w in picks]
        # [PCN_CMD_WAIT_ZERO] 廃止(2026-08-28・ユーザ指示)。注文の wait 成分を 0 にしていたが、
        # 待ち0は物理的に到達不能な注文なので方策がクラウド極へ倒れたまま戻らない。
        # 実証: main100 は100 iter すべてで誤差の100%が「達成>注文」の片側(飽和の署名)、
        # 正規化HVも 0.571 と v9(0.737)・fast100(0.778)に劣後した。
        # 経緯(残す): この挙動は n_jobs バグ修正(2026-08-27)以前の事実上の既定で、v9/main100 は
        # これで回っている。両runのログを読むときは「注文のwaitは常に0だった」前提で解釈すること。
        # 環境変数を立てても無視するが、黙って挙動が変わると事故なので警告を出す。
        if os.environ.get("PCN_CMD_WAIT_ZERO") == "1" and not getattr(self, "_cmd_wait_zero_warned", False):
            self._cmd_wait_zero_warned = True
            print("[PCN_CMD_WAIT_ZERO] 廃止済みのため無視する(待ち0は到達不能な注文)。"
                  "過去runの再現が目的ならこのコミットより前を使うこと。", flush=True)
        results = []
        for cost, wait in picks:
            c, w = float(cost), float(wait)
            if jitter > 0:
                c *= float(self.np_random.uniform(1.0 - jitter, 1.0 + jitter))
                w *= float(self.np_random.uniform(1.0 - jitter, 1.0 + jitter))
                c = max(0.0, c)
                w = max(0.0, w)
            dr = self._objectives_to_desired_return(c, w, nj)
            hz = self._pf_command_horizon(c, w)
            base = np.array([-w * nj, -c], dtype=np.float32)
            results.append((np.float32(dr), hz, np.float32(base)))
        return results

    def _coverage_directed_sample(self, pf_pool, n_commands, *, low_wait_frac=0.0,
                                  low_wait_quota=0, low_wait_max=0.0,
                                  include_extremes=True):
        """[PCN_CHOOSE_COMMANDS_MODE=coverage] カバレッジ駆動の注文選択。

        動機(2026-08-08 実測): 改善ループの replay は 2128 本のうち 439 本が「真の前線に
        一致するエピソード」だが、指している前線点は 4/76 種類しかない(同じ数点を数百回コピー)。
        一方 76 点を別々に持つ理想教師なら 66/76 ヒット・同コスト超過待ち 0 秒に到達する。
        つまり足りないのは表現力ではなく「まだ触っていない領域を注文すること」。

        方式:
          (a) 達成済み点(archive 全点, 重複除去)を cost 軸の格子に落として区画ごとの占有数を数える。
              格子は既定 log スケール(PCN_COVERAGE_SCALE=log|linear): 真の前線は安い側に密集する
              (18J では 76 点中 28 点が全 cost レンジの最初の 1/32)ので、線形格子だと安い側が
              1 区画に潰れて分解できない。
          (b) 空き/薄い区画ほど高い確率で選び(重み = 1/(1+占有数)^PCN_COVERAGE_ALPHA)、
              その区画内の cost を注文にする。wait は「今の到達前線を内挿した値 × (1-改善ナッジ)」。
              cost は到達済みレンジ [front_min, front_max] にクリップ = 外挿しすぎない。
          (c) PCN_COVERAGE_LEGACY_FRAC (既定 0.25) の割合は従来の stratified_sample_pf を混ぜ、
              既に取れている良い点の維持(安定性)を保つ。

        ノブ: PCN_COVERAGE_BINS(既定32) / PCN_COVERAGE_ALPHA(既定1.0) /
              PCN_COVERAGE_LEGACY_FRAC(既定0.25) / PCN_COVERAGE_IMPROVE(既定0.04) /
              PCN_COVERAGE_SCALE(既定log)。
        """
        from src.utils.pf_command_eval import stratified_sample_pf

        def _legacy(n):
            if n <= 0:
                return []
            return list(stratified_sample_pf(
                pf_pool, n, rng=self.np_random, low_wait_frac=low_wait_frac,
                low_wait_quota=low_wait_quota, low_wait_max=low_wait_max,
                include_extremes=include_extremes))

        pts = np.asarray(pf_pool, dtype=np.float64)
        if pts.shape[0] < 2:
            return _legacy(n_commands) or [(0.0, 0.0)]
        pts = pts[np.argsort(pts[:, 0])]
        c, w = pts[:, 0], pts[:, 1]
        lo, hi = float(c[0]), float(c[-1])
        nbins = max(2, int(os.environ.get("PCN_COVERAGE_BINS", "32")))
        alpha = max(0.0, float(os.environ.get("PCN_COVERAGE_ALPHA", "1.0")))
        legacy_frac = float(np.clip(float(os.environ.get("PCN_COVERAGE_LEGACY_FRAC", "0.25")), 0.0, 1.0))
        improve = float(os.environ.get("PCN_COVERAGE_IMPROVE", "0.04"))
        use_log = os.environ.get("PCN_COVERAGE_SCALE", "log").strip().lower() != "linear"
        if not (hi > lo):
            return _legacy(n_commands)

        def fwd(x):
            x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
            return np.log1p(x) if use_log else x

        def inv(y):
            return np.expm1(y) if use_log else y

        edges = np.linspace(float(fwd(lo)), float(fwd(hi)), nbins + 1)
        try:
            occ = self._archive_objective_points()
        except Exception:
            occ = np.empty((0, 2), dtype=np.float64)
        cnt = np.zeros(nbins, dtype=np.float64)
        if occ is not None and np.asarray(occ).size:
            uniq = np.unique(np.round(np.asarray(occ, dtype=np.float64), decimals=3), axis=0)
            b = np.clip(np.digitize(fwd(uniq[:, 0]), edges) - 1, 0, nbins - 1)
            cnt = np.bincount(b, minlength=nbins).astype(np.float64)
        wts = 1.0 / np.power(1.0 + cnt, alpha)
        s = float(wts.sum())
        wts = (wts / s) if (np.isfinite(s) and s > 0) else np.full(nbins, 1.0 / nbins)

        n_cov = int(round(n_commands * (1.0 - legacy_frac)))
        picks = []
        if n_cov > 0:
            alloc = self.np_random.multinomial(n_cov, wts)
            for bi, k in enumerate(alloc):
                for _ in range(int(k)):
                    t = float(self.np_random.uniform(0.0, 1.0))
                    cc = float(inv(edges[bi] + t * (edges[bi + 1] - edges[bi])))
                    cc = float(np.clip(cc, lo, hi))
                    ww = float(np.interp(cc, c, w)) * (1.0 - improve)
                    picks.append((max(0.0, cc), max(0.0, ww)))
        picks.extend(_legacy(n_commands - len(picks)))
        if not getattr(self, "_coverage_logged", False):
            self._coverage_logged = True
            print(f"[COVERAGE] カバレッジ駆動の注文を有効化: bins={nbins} "
                  f"scale={'log' if use_log else 'linear'} alpha={alpha:g} "
                  f"legacy_frac={legacy_frac:g} improve={improve:g} / "
                  f"到達cost=[{lo:.4g}, {hi:.4g}] 空き区画={int((cnt == 0).sum())}/{nbins}",
                  flush=True)
        return picks[:n_commands]

    def _gap_directed_sample(self, pf_pool, n_commands):
        """賢いコマンド選択(gap): PF の疎な帯（大きい cost 間隙＝「足りないところ」）に
        コマンドを多く割り当て、間隙の内挿点＋改善ナッジ（wait を少し下げた Pareto 方向の指令）を作る。
        ランダム選択・ランダム方向の代わりに「足りない所を重点的に・良い方向へ」改善させる。
        疎帯にコマンドが密集→sens/lowband の KL が近接コマンドを引き離す圧を強め、その帯の分解能を上げる狙い。
        ノブ: PCN_GAP_BOOST(間隙の効かせ方=指数), PCN_GAP_FRAC(gap割合), PCN_GAP_IMPROVE(ナッジ量)。"""
        pts = np.asarray(pf_pool, dtype=np.float64)
        if pts.shape[0] < 2:
            return [(float(p[0]), float(p[1])) for p in pts] or [(0.0, 0.0)]
        pts = pts[np.argsort(pts[:, 0])]
        c, w = pts[:, 0], pts[:, 1]
        seg = np.clip(np.diff(c), 1e-9, None)  # 隣接 cost 間隙
        boost = float(os.environ.get("PCN_GAP_BOOST", "1.5"))
        weights = seg ** max(0.0, boost)
        wsum = float(weights.sum())
        weights = (weights / wsum) if (np.isfinite(wsum) and wsum > 0) else np.full(len(seg), 1.0 / len(seg))
        gfrac = float(np.clip(float(os.environ.get("PCN_GAP_FRAC", "0.8")), 0.0, 1.0))
        n_gap = int(round(n_commands * gfrac))
        alloc = self.np_random.multinomial(n_gap, weights) if n_gap > 0 else np.zeros(len(seg), dtype=int)
        improve = float(os.environ.get("PCN_GAP_IMPROVE", "0.04"))
        picks = []
        for i, k in enumerate(alloc):
            for _ in range(int(k)):
                t = float(self.np_random.uniform(0.0, 1.0))
                cc = c[i] + t * (c[i + 1] - c[i])
                ww = (w[i] + t * (w[i + 1] - w[i])) * (1.0 - improve)  # 改善ナッジ
                picks.append((float(max(0.0, cc)), float(max(0.0, ww))))
        n_rest = n_commands - len(picks)  # 残りは既存 PF 点を維持
        if n_rest > 0:
            for i in self.np_random.choice(len(pts), size=n_rest, replace=True):
                picks.append((float(c[i]), float(w[i])))
        return picks

    def convergence_stats(self) -> dict:
        """PF収束の観測(原著PCNの自己焼きなましの見える化)。

        命令ナッジは ND 集合の std に比例(U(0, s·α))するため、フロントが改善を
        止めると std が縮み命令が達成点に漸近=「同じ点を支持し続ける」状態になる。
        ここでは毎 iteration:
        - n_nd: archive ND 点数(多様性・数)
        - n_new: 前回 ND に(正規化距離 eps 内で)存在しなかった新規点数
        - disp: 現 ND 各点→前回 ND への最近傍距離の平均(正規化)=フロントの動き
        - nd_std: ND 集合の正規化 std 平均(=ナッジ源の大きさ)
        を返し、「n_new==0 かつ disp<PCN_CONVERGE_EPS」の連続回数 streak を数える。
        streak>=PCN_CONVERGE_K で converged=True。
        """
        eps = float(os.environ.get("PCN_CONVERGE_EPS", "0.01"))
        k_need = int(os.environ.get("PCN_CONVERGE_K", "5"))
        pts = self._archive_pf_objective_points()
        if pts.size == 0:
            return {}
        prev = getattr(self, "_conv_prev_nd", None)
        lo = pts.min(axis=0)
        span = np.maximum(pts.max(axis=0) - lo, 1e-9)
        cur_n = (pts - lo) / span
        nd_std = float(cur_n.std(axis=0).mean())
        n_new = pts.shape[0]
        disp = 1.0
        if prev is not None and prev.size:
            prev_n = (prev - lo) / span
            d = np.linalg.norm(cur_n[:, None, :] - prev_n[None, :, :], axis=-1).min(axis=1)
            n_new = int((d > eps).sum())
            disp = float(d.mean())
        self._conv_prev_nd = pts.copy()
        stalled = (n_new == 0) and (disp < eps)
        self._conv_streak = getattr(self, "_conv_streak", 0) + 1 if stalled else 0
        converged = self._conv_streak >= k_need
        if converged:
            self._converged = True
        return {"n_nd": int(pts.shape[0]), "n_new": n_new, "disp": disp,
                "nd_std": nd_std, "streak": int(self._conv_streak), "converged": bool(converged)}

    def collapse_diag_stats(self, n_obs: int = 8, n_cmd: int = 5) -> str:
        """条件付け崩壊の診断(読み取り専用・学習非破壊)。

        1行のログ文字列を返す:
        - replay組成: episode先頭costの6帯ヒスト(0〜現在max) → データ劣化(H1)の観測
        - cmd_sens: 固定obs集合×PF両端+中間のn_cmd命令で action分布のペア平均L1/2(全変動)
                    → 「命令を変えると挙動が変わるか」の挙動レベル感度。崩壊=0に漸近
        - gamma_dev/beta_mag: FiLM変調量 |γ-1|,|β| の平均 → 命令経路の生死(H2)
        - wnorm: 全体/命令経路(film/c_emb)/s_emb/fc の重みL2 → ノルム膨張→飽和(H2)
        """
        import torch as th
        entries = self._valid_replay_entries()
        if not entries:
            return ""
        costs = []
        for e in entries:
            ov = getattr(e[2][0], "objective_values", None)
            if ov is not None:
                costs.append(float(ov[0]))
        costs = np.asarray(costs, dtype=np.float64)
        cmax = float(costs.max()) if costs.size else 1.0
        hist = np.histogram(costs, bins=6, range=(0.0, max(cmax, 1e-9)))[0].tolist() if costs.size else []

        # 命令感度プローブ: obs は replay 先頭 transition から等間隔に n_obs 個
        step = max(1, len(entries) // n_obs)
        obs_list = [entries[i][2][0].observation for i in range(0, len(entries), step)][:n_obs]
        pf = self._archive_pf_objective_points()
        if pf.size == 0 or len(obs_list) == 0:
            return f"replay_hist={hist} cmax={cmax:.3g} cmd_sens=NA"
        pf = pf[np.argsort(pf[:, 0])]
        nj = self._policy_n_jobs()
        fr = np.linspace(0.0, 1.0, n_cmd)
        cmds = []
        for t in fr:
            c = pf[0, 0] + t * (pf[-1, 0] - pf[0, 0])
            w = pf[0, 1] + t * (pf[-1, 1] - pf[0, 1])
            cmds.append(self._objectives_to_desired_return(c, w, nj))
        dev = next(self.model.parameters()).device
        obs_t = th.as_tensor(np.stack(obs_list), dtype=th.float32, device=dev)
        hz_t = th.full((len(obs_list), 1), float(nj), dtype=th.float32, device=dev)
        gamma_devs, beta_mags = [], []
        hooks = []
        if hasattr(self.model, "film_gamma"):
            hooks.append(self.model.film_gamma.register_forward_hook(
                lambda m, i, o: gamma_devs.append(float(o.abs().mean()))))
            hooks.append(self.model.film_beta.register_forward_hook(
                lambda m, i, o: beta_mags.append(float(o.abs().mean()))))
        was_training = self.model.training
        self.model.eval()
        probs = []
        try:
            with th.no_grad():
                for dr in cmds:
                    dr_t = th.as_tensor(np.tile(dr, (len(obs_list), 1)), dtype=th.float32, device=dev)
                    logp = self.model(obs_t, dr_t, hz_t)
                    probs.append(th.exp(logp).cpu().numpy())
        finally:
            if was_training:
                self.model.train()
            for h in hooks:
                h.remove()
        P = np.stack(probs)  # [n_cmd, n_obs, n_action]
        sens = []
        for i in range(len(cmds)):
            for j in range(i + 1, len(cmds)):
                sens.append(np.abs(P[i] - P[j]).sum(axis=-1).mean() / 2.0)
        cmd_sens = float(np.mean(sens)) if sens else 0.0

        def _wnorm(mods):
            s = 0.0
            for m in mods:
                if m is None:
                    continue
                for p in m.parameters():
                    s += float(p.detach().norm() ** 2)
            return s ** 0.5
        w_cmd = _wnorm([getattr(self.model, "film_gamma", None), getattr(self.model, "film_beta", None),
                        getattr(self.model, "c_emb", None)])
        w_s = _wnorm([getattr(self.model, "s_emb", None)])
        w_fc = _wnorm([getattr(self.model, "fc", None)])
        w_tot = _wnorm([self.model])
        g = float(np.mean(gamma_devs)) if gamma_devs else float("nan")
        b = float(np.mean(beta_mags)) if beta_mags else float("nan")
        return (f"replay_hist={hist} cmax={cmax:.3g} cmd_sens={cmd_sens:.4f} "
                f"gamma_dev={g:.4f} beta_mag={b:.4f} "
                f"wnorm_tot={w_tot:.2f} wnorm_cmd={w_cmd:.2f} wnorm_s={w_s:.2f} wnorm_fc={w_fc:.2f}")

    def _choose_commands_mpft(self, n_commands: int, iteration=None):
        """MPFT型 端→内側掃引の command 生成。

        杭 = 現 archive PF の両端点(+PCN_PF_COMMAND_ANCHORS の calibration 端点)。
        命令バッチの ENDPOINT_QUOTA を杭そのものに固定し、残りは cost 正規化位置
        p∈[0,r]∪[1-r,1] の PF 点から一様サンプル。r は iteration とともに
        START_FRAC→0.5 へ線形拡大＝端から内側へなぞる。掃引命令には Pareto 方向の
        微小ナッジ(wait×(1-IMPROVE))を掛け、杭は達成値そのまま(達成可能=RCSL安全)。
        """
        if iteration is not None:
            self._mpft_iter = int(iteration)
        else:
            self._mpft_iter = int(getattr(self, "_mpft_iter", 0)) + 1
        it = self._mpft_iter

        pool = self._archive_pf_objective_points()
        if _PF_COMMAND_ANCHORS > 0:
            from src.utils.pf_command_eval import merge_pf_pools
            anchors = self._anchor_command_points(_PF_COMMAND_ANCHORS)
            if anchors.size:
                pool = merge_pf_pools(pool, anchors) if pool.size else anchors
        if pool.size == 0 or pool.shape[0] < 2:
            # PF が育つ前(初期 iteration)は論文準拠の単発選択で場をつなぐ
            out = []
            for _ in range(n_commands):
                dr, hz = self._choose_commands(50)
                out.append((dr, hz, np.zeros(self.reward_dim, dtype=np.float32)))
            return out

        pts = pool[np.argsort(pool[:, 0])]
        c, w = pts[:, 0], pts[:, 1]
        span = max(float(c[-1] - c[0]), 1e-9)
        p = (c - c[0]) / span  # cost 正規化位置 0(安端)〜1(高端)

        if _MPFT_GATED:
            # 達成ゲート: reach は mpft_gate_update() が「マスターしたら」だけ広げる。
            reach = float(np.clip(getattr(self, "_mpft_reach", _MPFT_START_FRAC),
                                  _MPFT_START_FRAC, 0.5))
        else:
            reach = float(np.clip(
                _MPFT_START_FRAC + (0.5 - _MPFT_START_FRAC) * (it / max(_MPFT_FULL_ITER, 1.0)),
                _MPFT_START_FRAC, 0.5,
            ))
        lo_i = np.where(p <= reach)[0]
        hi_i = np.where(p >= 1.0 - reach)[0]
        if lo_i.size == 0:
            lo_i = np.array([0])
        if hi_i.size == 0:
            hi_i = np.array([len(pts) - 1])

        n_stake = max(2, int(round(n_commands * _MPFT_ENDPOINT_QUOTA)))
        picks = []  # (cost, wait, is_stake)
        for k in range(n_stake):
            i = 0 if k % 2 == 0 else len(pts) - 1  # 両端の杭を交互に
            picks.append((float(c[i]), float(w[i]), True))
        for k in range(n_commands - n_stake):
            seg = lo_i if k % 2 == 0 else hi_i  # 低cost側/高cost側を交互に掃引
            i = int(seg[self.np_random.integers(0, len(seg))])
            picks.append((float(c[i]), float(w[i]), False))

        if it % 10 == 1 or it <= 1:
            print(f"[MPFT] iter={it} reach={reach:.3f} pool={len(pts)} "
                  f"帯: 低側{len(lo_i)}点/高側{len(hi_i)}点 杭=({c[0]:.3g},{w[0]:.3g})/({c[-1]:.3g},{w[-1]:.3g}) "
                  f"命令 {n_commands} (杭{n_stake})", flush=True)

        nj = self._policy_n_jobs()
        results = []
        for cost, wait, is_stake in picks:
            ww = wait if is_stake else max(0.0, wait * (1.0 - _MPFT_IMPROVE))
            dr = self._objectives_to_desired_return(cost, ww, nj)
            hz = self._pf_command_horizon(cost, wait)
            base = np.array([-wait * nj, -cost], dtype=np.float32)
            results.append((np.float32(dr), hz, np.float32(base)))
        return results

    def mpft_gate_update(self, eval_pf, ref_pts) -> dict:
        """達成ゲート: 今の前線帯(cost正規化位置 [0,reach]∪[1-reach,1])を eval PF が
        archive PF に対しどれだけ一発再現できているか測り、gap<eps でマスター判定。
        PATIENCE 回連続でマスターしたら reach を STEP 広げる（=単調に前進）。

        driver が毎 eval で (eval_pf=到達非支配, ref_pts=archive PF) を渡す。返り値は診断用。
        """
        reach = float(getattr(self, "_mpft_reach", _MPFT_START_FRAC))
        ep = np.asarray(eval_pf, dtype=np.float64).reshape(-1, 2)
        rp = np.asarray(ref_pts, dtype=np.float64).reshape(-1, 2)
        if ep.shape[0] == 0 or rp.shape[0] < 2:
            return {"reach": reach, "gap": None, "mastered": False, "advanced": False}
        # cost 正規化位置で前線帯の archive PF 点を抽出
        cspan = max(float(rp[:, 0].max() - rp[:, 0].min()), 1e-9)
        wspan = max(float(rp[:, 1].max() - rp[:, 1].min()), 1e-9)
        p = (rp[:, 0] - rp[:, 0].min()) / cspan
        front = rp[(p <= reach) | (p >= 1.0 - reach)]
        if front.shape[0] == 0:
            front = rp
        # 各前線 archive 点に対する eval PF 最近傍の正規化距離 = 一発再現gap
        d = np.sqrt(((front[:, None, 0] - ep[None, :, 0]) / cspan) ** 2
                    + ((front[:, None, 1] - ep[None, :, 1]) / wspan) ** 2).min(axis=1)
        gap = float(d.mean())
        mastered = gap < _MPFT_GATE_EPS
        npass = int(getattr(self, "_mpft_gate_pass", 0))
        advanced = False
        if mastered:
            npass += 1
            if npass >= _MPFT_GATE_PATIENCE and reach < 0.5:
                reach = float(min(0.5, reach + _MPFT_GATE_STEP))
                npass = 0
                advanced = True
        else:
            npass = 0
        self._mpft_reach = reach
        self._mpft_gate_pass = npass

        # 学習量の適応: 改善(=前進 or 同一reachで gap 低下)が起きたら n_updates 倍率を上げ、
        # 伸び悩んだら 1.0 へ戻す。倍率は learn() が読んで n_updates に掛ける。
        mult = float(getattr(self, "_mpft_updates_mult", 1.0))
        if _MPFT_VOL_ADAPT:
            prev = getattr(self, "_mpft_prev_gap", None)
            improved = advanced or (prev is not None and (not advanced) and gap < prev - _MPFT_VOL_IMPROVE_EPS)
            if improved:
                mult = float(min(_MPFT_VOL_MAX, mult * _MPFT_VOL_RAMP))
            else:
                mult = float(max(1.0, mult * _MPFT_VOL_DECAY))
            self._mpft_updates_mult = mult
            # reach が進んだ直後は帯が変わり gap がジャンプするので基準をリセット
            self._mpft_prev_gap = None if advanced else gap

        print(f"[MPFT_GATE] front_gap={gap:.3f} eps={_MPFT_GATE_EPS} "
              f"mastered={mastered} pass={npass} reach={reach:.3f}"
              + (f" vol×{mult:.2f}" if _MPFT_VOL_ADAPT else "")
              + ("  → 前進" if advanced else ""), flush=True)
        return {"reach": reach, "gap": gap, "mastered": mastered, "advanced": advanced,
                "updates_mult": mult}

    @staticmethod
    def _hv2d_min(pf: np.ndarray, ref: np.ndarray) -> float:
        """2D最小化のhypervolume(ref=最悪角)。フロント品質の単一指標。"""
        pf = pf[(pf[:, 0] <= ref[0]) & (pf[:, 1] <= ref[1])]
        if pf.size == 0:
            return 0.0
        pf = pf[np.argsort(pf[:, 0])]
        hv = 0.0
        pw = float(ref[1])
        for c, w in pf:
            hv += (ref[0] - c) * (pw - w)
            pw = w
        return float(hv)

    def adapt_balance_power(self, eval_pf, ref_pts=None) -> dict:
        """command balance power を「到達PFのHVを最大化」する山登りで自己調整。
        left_frac は power に鈍感だったので、実目的HV(power)の山(≈0.5でピーク)を直接登る。
        HVが上がる限り同方向へ進み、下がったら反転してstepを半減=頂点に収束。手動powerを消す。
        nadir は archive(ref_pts, 全域を張り安定)の最悪角に固定 → 高cost到達でのHV水増しを防ぐ。"""
        if not (_COMMAND_BALANCE and _COMMAND_BALANCE_ADAPT):
            return {}
        pf_all = np.asarray(eval_pf, dtype=np.float64).reshape(-1, 2)
        if pf_all.shape[0] < 3:
            return {}
        nd_pf = pf_all[get_non_dominated_inds_minimize(pf_all)]
        # nadir(最悪角)は archive(ref_pts, 全域・power非依存)に固定 → 高cost到達でのHV水増しを防ぐ。
        nad = getattr(self, "_bal_nadir", None)
        rp = np.asarray(ref_pts, dtype=np.float64).reshape(-1, 2) if ref_pts is not None else None
        if nad is None:
            nad = (np.array([rp[:, 0].max(), rp[:, 1].max()]) if rp is not None and rp.shape[0] >= 2
                   else np.array([pf_all[:, 0].max(), pf_all[:, 1].max()], dtype=np.float64))
            self._bal_nadir = nad
        hv = self._hv2d_min(nd_pf, nad * 1.02)

        p = float(getattr(self, "_bal_power_cur", _COMMAND_BALANCE_POWER))
        last_hv = getattr(self, "_bal_last_hv", None)
        d = float(getattr(self, "_bal_dir", 1.0))
        step = float(getattr(self, "_bal_step", _COMMAND_BALANCE_STEP))
        if last_hv is None:
            p_new = p + d * step
        elif hv >= last_hv * 0.995:  # 改善→同方向
            p_new = p + d * step
        else:                         # 悪化→反転してstep半減(頂点へ収束)
            d = -d; step = max(step * 0.5, 0.03); p_new = p + d * step
        p_new = float(min(_COMMAND_BALANCE_PMAX, max(0.0, p_new)))
        self._bal_last_hv = hv; self._bal_dir = d; self._bal_step = step
        self._bal_power_cur = p_new
        self._apply_return_normalization_to_model()
        print(f"[BAL_ADAPT] hv={hv:.3e} power {p:.2f}->{p_new:.2f} step={step:.2f}", flush=True)
        return {"hv": hv, "power": p_new}

    def _choose_commands_batch(self, num_episodes: int, n_commands: int, iteration=None):
        """分散実行向け: 論文準拠の単発選択を複数回サンプリングする。"""
        if _MPFT_SWEEP:
            return self._choose_commands_mpft(n_commands, iteration)
        mode = os.environ.get("PCN_CHOOSE_COMMANDS_MODE", "paper").strip().lower()
        if mode in ("pf_archive", "pf_mixed", "pf_stratified", "pf_ref", "gap", "coverage"):
            return self._choose_commands_batch_from_pf(n_commands)
        # 戻り値要素:
        # (desired_return, desired_horizon, base_return)
        # base_return は「どの点から改善を狙ったか」の可視化/診断用
        default_cmd = (
            np.zeros(self.reward_dim, dtype=np.float32),
            np.float32(40),
            np.zeros(self.reward_dim, dtype=np.float32),
        )
        if _CMD_LOCAL_STEP:
            # [PCN_CMD_LOCAL_STEP] 論文式(σベース上乗せ)を隣接間隔ベースの局所ステップへ置換
            results = self._choose_commands_local_step(num_episodes, n_commands)
            if results is not None:
                return results
            print("警告: コマンド選択用のエピソードが見つかりませんでした。デフォルト値を返します。")
            return [default_cmd] * n_commands
        episodes = self._nlargest(num_episodes)
        if len(episodes) == 0:
            print("警告: コマンド選択用のエピソードが見つかりませんでした。デフォルト値を返します。")
            return [default_cmd] * n_commands

        returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
        nd_i = get_non_dominated_inds(np.array(returns))
        returns = np.array(returns)[nd_i]
        horizons = np.array(horizons)[nd_i]
        if len(returns) == 0:
            return [default_cmd] * n_commands

        _, s = np.mean(returns, axis=0), np.std(returns, axis=0)
        results = []
        for _ in range(n_commands):
            r_i = self.np_random.integers(0, len(returns))
            desired_horizon = np.float32(horizons[r_i] - 2)
            base_return = returns[r_i].copy()
            desired_return = base_return.copy()
            r_obj = self.np_random.integers(0, len(desired_return))
            desired_return[r_obj] += self.np_random.uniform(high=s[r_obj] * _COMMAND_ALPHA)
            results.append((np.float32(desired_return), desired_horizon, np.float32(base_return)))
        return results

    def _obs_for_policy(self, env, obs: np.ndarray) -> np.ndarray:
        """Actor が ReplayBuffer へ載せるのがイベント生ベクトルのとき、方策入力だけビットマップへ復元する。"""
        if not getattr(env, "_pcn_raw_event_obs_for_transfer", False):
            return obs
        from src.utils.event_obs_bitmap_adapter import event_obs_to_bitmap_observation

        ow = int(getattr(env, "obs_window_size", 10))
        return event_obs_to_bitmap_observation(
            obs,
            int(env.n_window),
            int(env.n_on_premise_node),
            int(env.n_cloud_node),
            ow,
        )

    def _act(self, obs: np.ndarray, desired_return, desired_horizon, eval_mode=False) -> int:
        obs_tensor = th.tensor(np.array([obs]), device=self.device).float()
        return_tensor = th.tensor(np.array([desired_return]), device=self.device).float()
        horizon_tensor = th.tensor([[desired_horizon]], device=self.device).float()

        # 推論のみ: 勾配グラフを作らない（出力は不変、毎ステップの dispatch/メモリを削減）。
        with th.no_grad():
            if self.use_enhanced_model:
                prediction_output = self.network(obs_tensor, return_tensor, horizon_tensor)
            elif _JIT_ACT and _S_EMB_DROPOUT <= 0.0:
                # TorchScript trace 経由(ビット一致・重み更新自動追従)。初回のみ実入力で trace。
                if getattr(self, "_jit_act_model", None) is None:
                    self._jit_act_model = th.jit.trace(
                        self.model, (obs_tensor, return_tensor, horizon_tensor)
                    )
                prediction_output = self._jit_act_model(obs_tensor, return_tensor, horizon_tensor)
            else:
                prediction_output = self.model(obs_tensor, return_tensor, horizon_tensor)

        if isinstance(prediction_output, tuple):
            prediction_scores = prediction_output[0]
        else:
            prediction_scores = prediction_output

        if self.continuous_action:
            action = prediction_scores.detach().cpu().numpy()[0]
            if not eval_mode:
                action = action + self.np_random.normal(0.0, self.noise, size=action.shape)
            return action
        else:
            scores = prediction_scores.detach()[0]
            if eval_mode:
                action = th.argmax(scores).item()
            else:
                if self.use_enhanced_model:
                    probs = F.softmax(scores, dim=-1)
                else:
                    probs = th.exp(scores)
                action = th.multinomial(probs, 1)[0].item()
            return action

    def _policy_logits_1d(
        self, obs: np.ndarray, desired_return, desired_horizon
    ) -> np.ndarray:
        """評価診断用: 1 ステップの方策出力（離散は LogSoftmax  logits、連続はそのまま）。"""
        obs_tensor = th.tensor(np.array([obs]), device=self.device).float()
        return_tensor = th.tensor(np.array([desired_return]), device=self.device).float()
        horizon_tensor = th.tensor([[desired_horizon]], device=self.device).float()
        with th.no_grad():
            if self.use_enhanced_model:
                prediction_output = self.network(obs_tensor, return_tensor, horizon_tensor)
            else:
                prediction_output = self.model(obs_tensor, return_tensor, horizon_tensor)
        if isinstance(prediction_output, tuple):
            prediction_scores = prediction_output[0]
        else:
            prediction_scores = prediction_output
        return prediction_scores.detach().cpu().numpy()[0]

    def _run_episode(self, env, desired_return, desired_horizon, max_return, eval_mode=False):
        transitions = []
        build_maps = schedule_maps_enabled()
        map_snapshots_on_premise = []
        map_snapshots_cloud = []
        
        obs = env.reset()
        done = False
        wt_sum = 0
        # アンカー残差: 指令から最近傍アンカー遺伝子を一度だけ選ぶ。env に渡す行動は
        # anchor_bit XOR policy_action(=残差)。eval は方策単独なので select は指令のみ依存(ずるなし)。
        _ar = get_anchor_set()
        _ar_gene = None
        _ar_job_idx = 0
        if _ar is not None:
            _, _ar_gene = _ar.select(desired_return)
        #
        # print("\n===== エピソード実行 =====")
        # print(f"目標: 報酬={desired_return}, ステップ数={desired_horizon}")

        _cost_hold_target = float(np.asarray(desired_return, dtype=np.float32)[1]) if _COST_HOLD else None
        # [SCHEDULER_OBS_BUDGET_RATIO] env が対応していれば、最初の行動選択前にも初期予算を
        # 反映しておく(既定OFFは _obs_budget_ratio=False のままなので下の分岐を一切通らず
        # ビット不変)。
        _obs_budget_ratio = getattr(env, "_obs_budget_ratio", False)
        if _obs_budget_ratio:
            env.set_remaining_budget(float(-np.asarray(desired_return, dtype=np.float32)[1]))
            obs = env.get_observation()
        while not done:
            policy_obs = self._obs_for_policy(env, obs)
            action = self._act(policy_obs, desired_return, desired_horizon, eval_mode)
            if _ar_gene is not None:
                _abit = int(_ar_gene[_ar_job_idx]) if _ar_job_idx < len(_ar_gene) else 0
                env_action = _abit ^ int(action)
            else:
                env_action = action
            n_obs, reward, scheduled, wt_step, done = env.step(env_action)
            if _ar_gene is not None and scheduled:
                _ar_job_idx += 1

            wt_sum += wt_step
            # 学習データ生成（Actor）はクリップなしで残り return を更新する。評価も同じ条件に揃える。
            desired_return = (desired_return - reward).astype(np.float32, copy=False)
            if _COST_HOLD:
                desired_return[1] = _cost_hold_target  # [anti-ration] cost目標を一定保持
            if not eval_mode and np.all(np.isfinite(max_return)):
                desired_return = np.clip(desired_return, None, max_return, dtype=np.float32)
            if _obs_budget_ratio:
                # [SCHEDULER_OBS_BUDGET_RATIO] n_obs(=次ステップの現ジョブ観測)が「このステップ
                # 支払い後の残り予算」を反映するよう、Transition構築前にenv側へ反映してから
                # n_obs を再構築する(get_observationは副作用なしの純関数)。
                env.set_remaining_budget(float(-desired_return[1]))
                n_obs = env.get_observation()
            if scheduled:
                desired_horizon = np.float32(max(desired_horizon - 1, 1.0))

            if done:
                fin = getattr(env, "finalize_window_history", None)
                if fin is not None:
                    import inspect
                    if "build_maps" in inspect.signature(fin).parameters:
                        fin(build_maps=build_maps)
                    else:
                        fin()

            transitions.append(
                Transition(
                    observation=obs,
                    action=action,
                    reward=np.float32(reward).copy(),
                    next_observation=n_obs,
                    terminal=done,
                )
            )
            obs = n_obs
        
        # エピソード完了後の結果表示
        # 注意: 累積報酬の計算は_add_episodeメソッドで行われるため、ここでは行わない
        episode_return = transitions[0].reward  # 即座の報酬（累積報酬ではない）
        
        final_return = episode_return
        onpre_final = (
            env.on_premise_window_history_full
            if build_maps and getattr(env, "on_premise_window_history_full", None) is not None
            else None
        )
        cloud_final = (
            env.cloud_window_history_full
            if build_maps and getattr(env, "cloud_window_history_full", None) is not None
            else None
        )
        value_cost, value_wt, value_avg_wt = env.calc_objective_values()
        
        # エピソード完了時の結果を表示
        if eval_mode:
            pass
            # print(f"エピソード完了: 長さ={len(transitions)}")
            # print(f"  累積報酬: {final_return}")
            # print(f"  実際のコスト: {value_cost}, 実際の待ち時間: {value_wt}")
            
            # # actionsetを表示
            # actions = [t.action for t in transitions]
            # print(f"  actionset: {actions}")
            
            # # マップ情報を表示
            # # print(f"  オンプレミスマップ: {onpre_final}")
            # # print(f"  クラウドマップ: {cloud_final}")
            # print("------------------------")
        
        # print(f"エピソード完了: 長さ={len(transitions)}")
        # print(f"最終報酬: {final_return}")
        # print(f"実際の値: コスト={value_cost}, 待ち時間={value_wt}")
        # print("=========================\n")
        
        return transitions, map_snapshots_on_premise, map_snapshots_cloud, wt_sum, [onpre_final, cloud_final], [value_cost, value_avg_wt]

    def set_desired_return_and_horizon(self, desired_return: np.ndarray, desired_horizon: int):
        """Set desired return and horizon for evaluation."""
        self.desired_return = desired_return
        self.desired_horizon = desired_horizon

    def eval(self, obs, w=None):
        """Evaluate policy action for a given observation."""
        return self._act(obs, self.desired_return, self.desired_horizon, eval_mode=True)
    
    def evaluate_and_execute_selected_policy(self, env, max_return, objective_index, n=10):
        """特定の目的関数の値が最大となるようなポリシーを評価して実行する"""
        n = min(n, len(self.experience_replay))
        # print("len(self.experience_replay)", len(self.experience_replay))
        episodes = self._nlargest(n)
        
        # 実際に取得されたエピソード数に基づいてnを調整
        actual_n = len(episodes)
        if actual_n == 0:
            print("警告: 評価用のエピソードが見つかりませんでした。")
            return None, [0, 0, 0]
        
        returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
        returns = np.float32(returns)
        horizons = np.float32(horizons)
        all_transitions = []
        e_returns = []
        
        # print(f"\n===== {actual_n}個のエピソード評価結果（選択実行版）=====")
        
        for i in range(actual_n):
            transitions, _, _, _, map_fin, value = self._run_episode(env, returns[i], np.float32(horizons[i]), max_return, eval_mode=True)
            all_transitions.append(transitions)
            # compute return
            for j in reversed(range(len(transitions) - 1)):
                transitions[j].reward += self.gamma * transitions[j + 1].reward
            e_returns.append(transitions[0].reward)
            
            # 各エピソードの結果を表示
            # print(f"エピソード {i+1}:")
            # print(f"  累積報酬: {transitions[0].reward}")
            # print(f"  実際のコスト: {value[0]}, 実際の待ち時間: {value[1]}")
            
            # actionsetを表示
            # actions = [t.action for t in transitions]
            # print(f"  actionset: {actions}")
            
            # マップ情報を表示
            # print(f"  オンプレミスマップ: {map_fin[0]}")
            # print(f"  クラウドマップ: {map_fin[1]}")
            # print()
            #やってみて、再現可能なデータを集める。

        # print("==========================================\n")

        # 非支配解の取得
        # print("e_returns", e_returns)
        e_returns_np = np.array(e_returns, dtype=np.float64)
        non_dominated_inds = get_non_dominated_inds(e_returns_np)
        pareto_front = e_returns_np[non_dominated_inds]
        if self.log:
            wandb.log({"pareto_front_eval_and_execute": wandb.Table(data=pareto_front, columns=["Objective1", "Objective2"])})


        # objective_indexが範囲内かチェック
        if objective_index >= len(e_returns[0]) if e_returns else 0:
            print(f"警告: objective_index {objective_index} が範囲外です。デフォルト値0を使用します。")
            objective_index = 0
        
        best_policy_index = np.argmax([e[objective_index] for e in e_returns])
        if objective_index == 9:
            "並び替えて，真ん中にあるものを選択する．"
            best_policy_index = len(e_returns) // 2
        # print("best_policy_index", best_policy_index)
        best_transitions = all_transitions[best_policy_index]
        # print("best_transitions", best_transitions[0].action)

        #execute best_transitions
        obs = env.reset()
        done = False
        step = 0
        wt_sum = 0
        culmulative_reward = np.zeros(self.reward_dim)
        while not done and step < len(best_transitions):
            action = best_transitions[step].action
            n_obs, reward, _, wt_step,_,done = env.step(action, exe_mode=1)
            if done:
                env.finalize_window_history()
            culmulative_reward += reward
            wt_sum += wt_step
            step += 1
        cost, mkspan = env.get_cost()
        print("culmulative_reward", culmulative_reward)

        return best_transitions, [wt_sum, mkspan, cost]

    def _select_eval_target_episodes(self, n: int):
        """評価用commandの元episodeを選ぶ。

        学習用のcommand選択とは分け、archive上のPFを端点込みで広く覆う。
        ここで選ぶのはあくまで「評価時にPCNへ投げる目標」であり、描画する点は
        この後に現在の方策をeval実行して得た到達点だけにする。
        """
        if n <= 0 or len(self.experience_replay) == 0:
            return []

        valid_entries = [entry for entry in self.experience_replay if len(entry[2]) > 0]
        if not valid_entries:
            return []

        returns = np.array([entry[2][0].reward for entry in valid_entries], dtype=np.float64)
        values = []
        has_values = True
        for _, _, episode in valid_entries:
            first = episode[0]
            if hasattr(first, "objective_values") and first.objective_values is not None:
                obj = first.objective_values
                values.append([obj[0], obj[2]])
            else:
                has_values = False
                break

        if has_values:
            value_arr = np.array(values, dtype=np.float64)
            nd_i = get_non_dominated_inds_minimize(value_arr)
            if len(nd_i) == 0:
                return self._nlargest(n)
            pf_points = value_arr[nd_i]
            sort_order = np.lexsort((pf_points[:, 1], pf_points[:, 0]))
            sorted_local = nd_i[sort_order]
            sorted_points = self._normalize_points_for_selection(pf_points)[sort_order]
        else:
            nd_i = get_non_dominated_inds(returns)
            if len(nd_i) == 0:
                return self._nlargest(n)
            pf_points = returns[nd_i]
            sort_order = np.lexsort((pf_points[:, 1], pf_points[:, 0]))
            sorted_local = nd_i[sort_order]
            sorted_points = self._normalize_points_for_selection(pf_points)[sort_order]

        selected_local: List[int] = []
        if len(sorted_local) <= n:
            selected_local = sorted_local.tolist()
        elif len(sorted_local) == 1:
            selected_local = [int(sorted_local[0])]
        else:
            # PF 点が密（replay が大きい）だと弧長 searchsorted が同じ index に潰れる。
            # 評価は index 上で均等サンプルし、端点を必ず含める。
            pick_pos = np.linspace(0, len(sorted_local) - 1, n)
            pick_i = np.unique(np.rint(pick_pos).astype(int))
            pick_i = np.clip(pick_i, 0, len(sorted_local) - 1)
            endpoint_i = np.array([0, len(sorted_local) - 1], dtype=int)
            pick_i = np.unique(np.concatenate([endpoint_i, pick_i]))
            if len(pick_i) < min(n, len(sorted_local)):
                diffs = np.diff(sorted_points, axis=0)
                seg_lengths = np.linalg.norm(diffs, axis=1)
                arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
                if arc[-1] > 0:
                    targets = np.linspace(0.0, arc[-1], n)
                    arc_pick = np.unique(
                        np.clip(np.searchsorted(arc, targets, side="left"), 0, len(sorted_local) - 1)
                    )
                    pick_i = np.unique(np.concatenate([pick_i, arc_pick]))
            selected_local = [int(sorted_local[i]) for i in pick_i[:n]]

        selected = [valid_entries[i] for i in selected_local]
        seen = {
            tuple(np.round(np.asarray(entry[2][0].reward, dtype=np.float64), 6).tolist() + [float(len(entry[2]))])
            for entry in selected
        }

        if len(selected) < n:
            for entry in self._nlargest(n):
                key = tuple(
                    np.round(np.asarray(entry[2][0].reward, dtype=np.float64), 6).tolist()
                    + [float(len(entry[2]))]
                )
                if key in seen:
                    continue
                selected.append(entry)
                seen.add(key)
                if len(selected) >= n:
                    break

        return selected[:n]

    def evaluate(
        self,
        env,
        max_return,
        n=10,
        save_history: bool = True,
        eval_diag: Optional[Dict[str, Any]] = None,
        return_command_outcomes: bool = False,
    ):
        """評価結果を履歴に保存し、優れた解を経験再生バッファに追加するよう拡張したevaluate"""
        n = min(n, len(self.experience_replay))
        episodes = self._select_eval_target_episodes(n)
        eval_command_outcomes: List[Dict[str, Any]] = []
        
        # 実際に取得されたエピソード数に基づいてnを調整
        actual_n = len(episodes)
        if actual_n == 0:
            print("警告: 評価用のエピソードが見つかりませんでした。")
            return [], [], [], None
        
        returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
        returns = np.float32(returns)
        horizons = np.float32(horizons)
        e_returns = []
        e_values = []
        all_transitions = []  # 全てのtransitionsを保存するリスト
        first_actions: List[int] = []
        episode_lens: List[int] = []
        trajectory_samples: List[Dict[str, Any]] = []
        sample_idx: set = set()
        eval_lightweight = bool(eval_diag and eval_diag.get("lightweight"))
        if eval_diag and eval_diag.get("path") and not eval_lightweight:
            sample_idx = {0, actual_n // 2, actual_n - 1}
            sample_idx = {j for j in sample_idx if 0 <= j < actual_n}
        elif eval_diag and eval_diag.get("path") and eval_lightweight:
            sample_idx = {0, actual_n - 1} if actual_n > 0 else set()
            sample_idx = {j for j in sample_idx if 0 <= j < actual_n}
        
        # print(f"\n===== {actual_n}個のエピソード評価結果 =====")
        
        for i in range(actual_n):
            logits0_list: Optional[List[float]] = None
            if i in sample_idx:
                obs_probe = env.reset()
                logits0 = self._policy_logits_1d(
                    obs_probe, returns[i], np.float32(horizons[i])
                )
                logits0_list = [float(round(x, 6)) for x in np.asarray(logits0).ravel()]
            transitions, _, _, _, map_fin, value = self._run_episode(env, returns[i], np.float32(horizons[i]), max_return, eval_mode=True)
            if i in sample_idx:
                sample_rec: Dict[str, Any] = {
                    "i": i,
                    "target_return": np.asarray(returns[i], dtype=np.float64).tolist(),
                    "target_horizon": float(horizons[i]),
                    "logits_step0": logits0_list,
                }
                if not eval_lightweight:
                    sample_rec["actions"] = [int(t.action) for t in transitions]
                else:
                    sample_rec["first_action"] = int(transitions[0].action) if transitions else -1
                    sample_rec["episode_len"] = len(transitions)
                trajectory_samples.append(sample_rec)
            
            # 累積報酬を計算（表示用のみ）
            transitions_copy = []
            for t in transitions:
                transitions_copy.append(Transition(
                    observation=t.observation,
                    action=t.action,
                    reward=np.array(t.reward, copy=True),
                    next_observation=t.next_observation,
                    terminal=t.terminal
                ))
            
            for j in reversed(range(len(transitions_copy) - 1)):
                transitions_copy[j].reward += self.gamma * transitions_copy[j + 1].reward
            
            e_returns.append(transitions_copy[0].reward)
            e_values.append(value)
            all_transitions.append(transitions)  # 元のtransitionsを保存
            first_actions.append(int(transitions[0].action) if transitions else -1)
            episode_lens.append(len(transitions))

            if return_command_outcomes:
                target_return = np.asarray(returns[i], dtype=np.float32)
                achieved_return = np.asarray(transitions_copy[0].reward, dtype=np.float32)
                source_first = episodes[i][2][0]
                if (
                    hasattr(source_first, "objective_values")
                    and source_first.objective_values is not None
                ):
                    obj = source_first.objective_values
                    command_values = np.array([obj[0], obj[2]], dtype=np.float32)
                else:
                    n_jobs = max(1, int(getattr(env, "n_jobs", 1) or 1))
                    if not hasattr(env, "n_jobs") and hasattr(env, "unwrapped"):
                        n_jobs = max(1, int(getattr(env.unwrapped, "n_jobs", n_jobs) or n_jobs))
                    command_values = np.array(
                        [-target_return[1], (-target_return[0] / n_jobs)],
                        dtype=np.float32,
                    )
                achieved_values = np.array(
                    [float(value[0]), float(value[1])],
                    dtype=np.float32,
                )
                eval_command_outcomes.append(
                    {
                        "command_return": target_return.tolist(),
                        "achieved_return": achieved_return.tolist(),
                        "command_values": command_values.tolist(),
                        "achieved_values": achieved_values.tolist(),
                        "target_horizon": float(horizons[i]),
                        "first_action": int(transitions[0].action) if transitions else -1,
                    }
                )
            
            # 各エピソードの結果を表示
            # print(f"エピソード {i+1}:")
            # print(f"  累積報酬: {transitions[0].reward}")
            # print(f"  実際のコスト: {value[0]}, 実際の待ち時間: {value[1]}")
            
            # actionsetを表示
            # actions = [t.action for t in transitions]
            # print(f"  actionset: {actions}")
            # print()
        
        # print("===============================\n")
        


        # 従来通りNumPyで計算（オーバーフロー防止のためfloat64を使用）
        distances = np.linalg.norm(
            np.array(returns, dtype=np.float64) - np.array(e_returns, dtype=np.float64), 
            axis=-1
        )

        # 非支配解を抽出（CPU上で実行）
        e_returns_np = np.array(e_returns, dtype=np.float64)  # float64でキャスト
        e_values_np = np.array(e_values, dtype=np.float64)    # float64でキャスト
        
        non_dominated_inds_reward = get_non_dominated_inds(e_returns_np)
        non_dominated_inds_values = get_non_dominated_inds_minimize(e_values_np)
        pareto_front_reward = e_returns_np[non_dominated_inds_reward]
        pareto_front_values = e_values_np[non_dominated_inds_values]
        
        # 履歴に保存（定期評価では save_history=False でメモリ・I/O を抑える）
        if save_history:
            self.evaluation_history.append({
                'all_returns': np.array(e_returns),
                'pareto_front_reward': pareto_front_reward,
                'pareto_front_values': pareto_front_values,
                'values': e_values
            })
            self.evaluation_timestamps.append("1")
            self.global_steps_at_evaluation.append(self.global_step)
            if hasattr(self, '_train_wall_t0'):
                self.wall_seconds_at_evaluation.append(
                    time.perf_counter() - self._train_wall_t0
                )
            else:
                self.wall_seconds_at_evaluation.append(time.perf_counter())
        
        if eval_diag and eval_diag.get("path"):
            from pathlib import Path

            from src.utils.pcn_eval_diag import append_jsonl, count_unique_targets, count_unique_values

            n_uni_t, _target_rows = count_unique_targets(returns, horizons)
            n_uni_v = count_unique_values(e_values)
            n_uni_er = len(
                {
                    tuple(np.round(np.asarray(r, dtype=np.float64).ravel(), 5))
                    for r in e_returns
                }
            )
            heap_ids = [str(ep[1]) if len(ep) > 1 else "" for ep in episodes]
            n_uni_seq = 0
            if all_transitions:
                n_uni_seq = len(
                    {tuple(int(t.action) for t in tr) for tr in all_transitions}
                )
            record: Dict[str, Any] = {
                "training_iteration": eval_diag.get("training_iteration"),
                "n_eval_episodes": actual_n,
                "experience_replay_size": len(self.experience_replay),
                "n_unique_targets_return_horizon": n_uni_t,
                "n_unique_e_values": n_uni_v,
                "n_unique_e_returns_vector": n_uni_er,
                "n_unique_action_sequences": n_uni_seq,
                "first_actions": first_actions[: min(16, len(first_actions))],
                "episode_lens": episode_lens[: min(16, len(episode_lens))],
                "heap_entry_step_ids": heap_ids[: min(16, len(heap_ids))],
                "trajectory_samples": trajectory_samples,
            }
            append_jsonl(Path(eval_diag["path"]), record)
        
        if return_command_outcomes:
            return e_returns, e_values, distances, map_fin, eval_command_outcomes
        return e_returns, e_values, distances, map_fin

    def save(self, filename: str = "PCN_model", savedir: str = "weights"):
        """保存時に一意のファイル名を生成して新規ファイルを作成"""
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        
        # 一意のIDを生成（タイムスタンプとランダム数字の組み合わせ）
        import datetime
        import random
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = f"{timestamp}_{random.randint(1000, 9999)}"
        
        # 一意のファイル名を作成
        unique_filename = f"{filename}_{unique_id}.pt"
        
        # モデルを保存
        model_path = f"{savedir}/{unique_filename}"

        if self.use_enhanced_model:
            th.save(self.network.state_dict(), model_path) # 拡張モデルのstate_dictを保存
        else:
            th.save(self.model.state_dict(), model_path) # 通常モデルのstate_dictを保存
        
        # 最新モデルとしてもコピーして保存（最新版へのアクセスを簡単にするため）
        latest_path = f"{savedir}/{filename}_latest.pt"
        import shutil
        shutil.copy2(model_path, latest_path)
        
        # print(f"モデルを保存しました: {model_path}")
        # print(f"最新モデルとしても保存: {latest_path}")

    def load(self, filename: str = "PCN_model", savedir: str = "weights"):
        """指定されたモデルを読み込み、拡張モデルか通常モデルかを自動判別"""
        model_path = f"{savedir}/{filename}.pt"
        if not os.path.exists(model_path):
            latest_path = f"{savedir}/{filename}_latest.pt"
            if os.path.exists(latest_path):
                model_path = latest_path
            else:
                print(f"モデルファイルが見つかりません: {model_path} および {latest_path}")
                return

        # state_dictまたはモデルオブジェクトを読み込む
        loaded = th.load(model_path, map_location=self.device, weights_only=False)

        # モデル全体が保存されている場合（legacy_validation/PCN_origin.py形式）はstate_dictを抽出
        if isinstance(loaded, th.nn.Module):
            state_dict = loaded.state_dict()
        else:
            state_dict = loaded

        try:
            if self.use_enhanced_model:
                # EnhancedPCNModel が __init__ で正しく初期化されている前提
                self.network.load_state_dict(state_dict, strict=False)
                self.target_network.load_state_dict(state_dict, strict=False) # ターゲットネットワークも同期
                print(f"拡張モデルを読み込みました: {model_path}")
            else:
                # BasePCNModel のサブクラスが __init__ で正しく初期化されている前提
                self.model.load_state_dict(state_dict, strict=False)
                print(f"通常モデルを読み込みました: {model_path}")
        except RuntimeError as e:
            print(f"モデルの読み込み中にエラーが発生しました（キーの不一致など）: {e}")
            print("モデルのアーキテクチャが保存時と異なる可能性があります。")
        except AttributeError as e:
            print(f"モデルの読み込み中にエラーが発生しました: {e}")
            print("`self.network` または `self.model` が正しく初期化されていません。")

    def clean_zero_crowding_points(self, threshold=0.001):
        """混雑度が閾値以下（実質的に重複）の点をバッファから削除する"""
        if len(self.experience_replay) <= 5:  # バッファが少なすぎる場合は何もしない
            return 0
            
        returns = np.array([e[2][0].reward for e in self.experience_replay], dtype=np.float64)
        cd_values = crowding_distance(returns)
        
        # 非支配解のインデックスを取得
        non_dom_indices = get_non_dominated_inds(returns)
        
        # 新しいバッファを準備
        new_buffer = []
        removed_count = 0
        
        for i, (priority, step, transitions) in enumerate(self.experience_replay):
            # 混雑度が閾値より大きいか、非支配解の場合は保持
            if cd_values[i] > threshold or i in non_dom_indices:
                new_buffer.append((priority, step, transitions))
            else:
                removed_count += 1
        
        if removed_count > 0:
            # バッファの更新
            self.experience_replay = new_buffer
            heapq.heapify(self.experience_replay)
            print(f"バッファクリーニング: 混雑度{threshold}以下の{removed_count}個のエピソードを除去。残り{len(self.experience_replay)}個")
        
        return removed_count

    def train(
        self,
        total_timesteps: int,
        eval_env: gym.Env,
        ref_point: np.ndarray,
        num_er_episodes: int,
        num_step_episodes: int,
        num_model_updates: int,
        max_buffer_size: int,
        num_eval_weights_for_eval: int = 25,
        max_return: np.ndarray = None,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_points_pf: int = 200,
        num_eval_episodes: int = 30,
        log_episode_only: bool = True,
        use_wandb: bool = True,
        reset_buffer: bool = False,  # バッファをリセットするかどうかの新しいパラメータ
    ):
        """Train PCN with support for safe termination."""
        # シグナルハンドラを登録
        self.register_signal_handlers()
        self._train_wall_t0 = time.perf_counter()
        self.evaluation_history = []
        self.global_steps_at_evaluation = []
        self.wall_seconds_at_evaluation = []
        
        try:
            # ユーザーに中断機能について通知
            print("\n=== PCN学習を開始します。Ctrl+Cで安全に終了できます ===\n")
            
            # 既存の初期化コード
            max_return = max_return if max_return is not None else np.full(self.reward_dim, np.inf, dtype=np.float32)

            if self.log:
                self.register_additional_config(
                    {
                        "total_timesteps": total_timesteps,
                        "ref_point": ref_point.tolist(),
                        "known_front": known_pareto_front,
                        "num_eval_weights_for_eval": num_eval_weights_for_eval,
                        "num_er_episodes": num_er_episodes,
                        "num_step_episodes": num_step_episodes,
                        "num_model_updates": num_model_updates,
                        "max_return": max_return.tolist(),
                        "max_buffer_size": max_buffer_size,
                        "num_points_pf": num_points_pf,
                        "log_episode_only": log_episode_only,
                    }
                )
                
            self.global_step = 0
            total_episodes = 0  # ヒューリスティックエピソードを含めないよう0から開始
            n_checkpoints = 0
            desired_return = np.zeros((2,), dtype=np.float32)
            desired_horizon = np.zeros((1,), dtype=np.float32)
            
            cumulative_rewards = []
            real_values = []
            episode_maps_on_premise = []
            episode_maps_cloud = []
            
            # クリーニングのカウンター
            cleaning_counter = 0
            
            # バッファにランダムエピソードを追加（既存のバッファ内容は保持）
            existing_buffer_size = len(self.experience_replay)
            
            # バッファがすでにヒューリスティックデータで初期化されているか確認
            if reset_buffer or existing_buffer_size == 0:
                # バッファを空にして新しく埋める
                self.experience_replay = []
                print(f"経験再生バッファをリセットし、{num_er_episodes}エピソードのランダムデータで埋めます...")
                
                num_count = 0
                for episode_idx in range(num_er_episodes):
                    # 中断要求を確認
                    if self.terminate_requested:
                        print("経験バッファ充填中に中断要求を受信しました。")
                        break
                        
                    transitions = []
                    obs = self.env.reset()
                    done = False

                    while not done:
                        # 中断要求を確認（長いエピソード内でも）
                        if self.terminate_requested:
                            break
                            
                        # より多様なランダムアクションを生成
                        # 各エピソードで異なるシードを使用
                        episode_seed = int(time.time() * 1000) + episode_idx + self.global_step
                        np.random.seed(episode_seed)
                        action = self.env.action_space.sample()
                        
                        n_obs, reward, scheduled, wt_step, done = self.env.step(action)
                        
                        if done:
                            self.env.finalize_window_history()
                            num_count += 1
                            
                            # 1000エピソードごとに進捗表示
                            if num_count % 1000 == 0:
                                print(f"Completed {num_count} episodes for experience replay buffer.")
                                
                                # log_episode_onlyがTrueの場合は、エピソード数のみをログに記録
                                if log_episode_only and self.log:
                                    wandb.log({"episodes": num_count})
                        
                        transitions.append(Transition(obs, action, np.float32(reward).copy(), n_obs, done))
                        obs = n_obs
                        self.global_step += 1

                    # 中断されていない場合のみエピソードを追加
                    if not self.terminate_requested:
                        self._add_episode(transitions, max_size=max_buffer_size, step=self.global_step)
                
                # バッファ充填の最終状態を表示
                if num_count % 1000 != 0:
                    print(f"Total of {num_count} episodes completed for experience replay buffer.")
                
                total_episodes = num_count
            else:
                # 既存のバッファを使用
                print(f"既存の経験再生バッファを使用します（サイズ: {existing_buffer_size}エピソード）")
                total_episodes = existing_buffer_size

            # メインの学習ループ
            while self.global_step < total_timesteps and not self.terminate_requested:
                loss = []
                entropy = []
                
                # 中断要求を確認
                if self.terminate_requested:
                    break
                    
                for update_idx in range(num_model_updates):
                    # 頻繁な中断チェック（更新が多い場合）
                    if update_idx % 100 == 0 and self.terminate_requested:
                        break
                        
                    l, lp = self.update()
                    loss.append(l.detach().cpu().numpy())
                    if not self.continuous_action:
                        lp = lp.detach().cpu().numpy()
                        ent = np.sum(-np.exp(lp) * lp)
                        entropy.append(ent)

                # モデル更新中に中断された場合
                if self.terminate_requested:
                    break

                desired_return, desired_horizon = self._choose_commands(num_er_episodes)

                if use_wandb and log_episode_only:
                    wandb.log({
                        "episodes": total_episodes,
                        "global_step": self.global_step,
                        "loss": np.mean(loss),
                    })

                # 既存のログコード
                leaves_r = np.array([e[2][0].reward for e in self.experience_replay[len(self.experience_replay) // 2 :]])
                if self.log and not log_episode_only:
                    hv = hypervolume(ref_point, leaves_r)
                    hv_est = hv
                    wandb.log(
                        {
                            "train/hypervolume": hv_est,
                            "train/loss": np.mean(loss), 
                            "global_step": self.global_step,
                        },
                    )

                returns = []
                horizons = []

                for episode_idx in range(num_step_episodes):
                    # 中断要求を確認
                    if self.terminate_requested:
                        break
                        
                    transitions, maps_on_pre, maps_cloud, wt_sum, map_fin, value = self._run_episode(
                        self.env, desired_return, desired_horizon, max_return
                    )
                    self.global_step += len(transitions)
                    self._add_episode(transitions, max_size=max_buffer_size, step=self.global_step)
                    del maps_on_pre, maps_cloud, map_fin
                    
                    # 累積報酬を計算（表示用のみ）
                    transitions_copy = []
                    for t in transitions:
                        transitions_copy.append(Transition(
                            observation=t.observation,
                            action=t.action,
                            reward=np.array(t.reward, copy=True),
                            next_observation=t.next_observation,
                            terminal=t.terminal
                        ))
                    
                    for j in reversed(range(len(transitions_copy) - 1)):
                        transitions_copy[j].reward += self.gamma * transitions_copy[j + 1].reward
                    
                    returns.append(transitions_copy[0].reward)
                    horizons.append(len(transitions))

                    # 各エピソードのstepごとの配置マップを累積
                    if schedule_maps_enabled():
                        episode_maps_on_premise.extend(maps_on_pre)
                        episode_maps_cloud.extend(maps_cloud)

                    # エピソードごとの待ち時間とコストを取得
                    _, total_cost = self.env.get_episode_metrics()

                    if self.log and not log_episode_only:
                        wandb.log(
                            {
                                "episode/wt_sum": wt_sum,
                                "episode/total_cost": total_cost,
                            },
                        )
                    # 累積報酬を計算してリストに追加
                    cumulative_rewards.append(transitions[0].reward)
                    real_values.append((wt_sum, total_cost))

                # エピソード実行中に中断された場合
                if self.terminate_requested:
                    break

                total_episodes += num_step_episodes
                
                # log_episode_onlyがTrueの場合は、エピソード数のみをログに記録
                if self.log and log_episode_only:
                    wandb.log({
                        "episodes": total_episodes,
                        "global_step": self.global_step,
                        "loss": np.mean(loss),
                    })

                # 既存のログ処理
                if self.log and not log_episode_only and len(returns) > 0:  # 中断時に空の場合に備えて確認
                    wandb.log(
                        {
                            "train/episode": total_episodes,
                            "train/horizon_desired": desired_horizon,
                            "train/mean_horizon_distance": np.linalg.norm(np.mean(horizons) - desired_horizon),
                        },
                    )

                    for i in range(self.reward_dim):
                        wandb.log(
                            {
                                f"train/desired_return_{i}": desired_return[i],
                                f"train/mean_return_{i}": np.mean(np.array(returns)[:, i]),
                                f"train/mean_return_distance_{i}": np.linalg.norm(
                                    np.mean(np.array(returns)[:, i]) - desired_return[i]
                                ),
                                "global_step": self.global_step,
                            },
                        )
                
                # 学習状況のコンソール出力
                leaves_r = np.array([e[2][0].reward for e in self.experience_replay[len(self.experience_replay) // 2 :]])
                # print("self.experience_replay",self.experience_replay)
                hv = hypervolume(ref_point, leaves_r)
                # print("leaves_r",leaves_r)
                
                # 各点のCrowding Distanceを計算して表示
                if len(leaves_r) > 1:  # 少なくとも2点必要
                    cd_values = crowding_distance(leaves_r)
                    # print("各点のCrowding Distance:")
                    # for i, (point, cd) in enumerate(zip(leaves_r, cd_values)):
                    #     print(f"  点{i+1}: {point} -> 混雑度: {cd:.4f}")
                    
                    # 非支配解を特定して表示
                    non_dom_indices = get_non_dominated_inds(leaves_r)
                    # print(f"非支配解の数: {len(non_dom_indices)}")
                    # if len(non_dom_indices) > 0:
                        # print("非支配解:")
                        # for i, idx in enumerate(non_dom_indices):
                            # print(f"  解{i+1}: {leaves_r[idx]} -> 混雑度: {cd_values[idx]:.4f}")
                
                hv_est = hv
                
                # 定期的に混雑度0の点をクリーニング（20エピソードごと）
                cleaning_counter += 1
                if cleaning_counter >= 20:
                    self.clean_zero_crowding_points(threshold=0.001)
                    cleaning_counter = 0
 
                if len(returns) > 0:  # 中断時に空の場合に備えて確認
                    print(
                        f"step {self.global_step}/{total_timesteps} ({self.global_step/total_timesteps*100:.1f}%) | "
                        f"episode {total_episodes} | "
                        f"return {np.mean(returns, axis=0)} | "
                        f"loss {np.mean(loss):.3E} | "
                        f"hv {hv_est}"
                    )



                # 前回評価時のエピソード数を記録
                if not hasattr(self, 'last_eval_episode'):
                    self.last_eval_episode = 0

                # より厳密な条件で評価実行
                if total_episodes >= self.last_eval_episode + num_eval_episodes:
                    print(f"エピソード{total_episodes}での評価を実行中...")
                    self.evaluate(eval_env, max_return, n=num_points_pf)
                    self.last_eval_episode = total_episodes
                    
                    # 中断処理のために短いスリープを入れる（Ctrl+Cを処理する時間）
                    time.sleep(0.1)

            # 学習ループ終了（通常完了または中断）
            training_status = "中断" if self.terminate_requested else "正常完了"
            print(f"\n=== 学習が{training_status}しました ===")
            
            # 終了処理（通常終了時も中断時も実行）
            if not self.terminate_requested:
                # 通常終了時の保存処理
                self.save()
                self.e_returns, _, _, self.mapmap = self.evaluate(eval_env, max_return, n=num_points_pf)
                self.save_pareto_solutions_to_txt(mode_name="training_complete")
                print("訓練結果を保存しました。")
            else:
                # 中断時は特別な終了処理を実行
                self.save_results_on_termination(eval_env, max_return, num_points_pf)
            
        except Exception as e:
            print(f"学習中に予期しないエラーが発生しました: {e}")
            traceback.print_exc()
            # エラー発生時も安全に終了処理を試みる
            self.save_results_on_termination(eval_env, max_return, num_points_pf)
        
        finally:
            # 元のシグナルハンドラを復元
            self.restore_signal_handlers()
        
        return self.e_returns

    def visualize_evaluation_history(self, save_dir="evaluation_history"):
        """評価履歴を可視化し、固定ファイル名で上書き保存。
        報酬（最大化目的）と実数値（最小化目的）の両方のグラフを別々に表示する。
        """
        if not self.evaluation_history:
            print("評価履歴がありません")
            return
        
        # ディレクトリ作成
        os.makedirs(save_dir, exist_ok=True)
        
        # 固定ファイル名を使用（更新時に上書き）
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ------ 実数値（最小化目的）のグラフ ------
        # 全データから適切な表示範囲を計算
        all_x_values = []
        all_y_values = []
        for history in self.evaluation_history:
            all_returns = history['values']
            all_x_values.extend([ret[0] for ret in all_returns])
            all_y_values.extend([ret[1] for ret in all_returns])
        
        # 表示範囲の計算（少しマージンを追加）
        x_min, x_max = min(all_x_values), max(all_x_values)
        y_min, y_max = min(all_y_values), max(all_y_values)
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        x_range = [x_min - x_margin, x_max + x_margin]
        y_range = [y_min - y_margin, y_max + y_margin]
        
        # パレートフロントの進化を可視化（実数値 - 最小化目的）
        plt.figure(figsize=(15, 10))
        
        # 各評価時点のパレートフロントをプロット
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.evaluation_history)))
        
        for i, (history, step) in enumerate(zip(self.evaluation_history, self.global_steps_at_evaluation)):
            pareto_front_values = history['pareto_front_values']
            plt.scatter(
                [ret[0] for ret in pareto_front_values], 
                [ret[1] for ret in pareto_front_values],
                color=colors[i], 
                label=f"Step {step}",
                alpha=0.7
            )
        
        plt.title("パレートフロントの進化（実数値 - 最小化目的）")
        plt.xlabel("コスト（実数）")
        plt.ylabel("makespan（実数）")
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        
        # 左下が最適な方向であることを示す矢印
        arrow_x = x_min + (x_max - x_min) * 0.85
        arrow_y = y_min + (y_max - y_min) * 0.85
        plt.annotate('最適方向', xy=(arrow_x - (x_max - x_min) * 0.2, arrow_y - (y_max - y_min) * 0.2), 
                    xytext=(arrow_x, arrow_y),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                    fontsize=12)
        
        # 固定ファイル名で保存（上書き）
        pareto_values_png_filename = f"{save_dir}/pareto_values_evolution.png"
        plt.savefig(pareto_values_png_filename)
        plt.close()
        
        # ------ 報酬（最大化目的）のグラフ ------
        # 全データから適切な表示範囲を計算
        all_x_values_reward = []
        all_y_values_reward = []
        for history in self.evaluation_history:
            all_returns = history['all_returns']
            all_x_values_reward.extend([ret[0] for ret in all_returns])
            all_y_values_reward.extend([ret[1] for ret in all_returns])
        
        # 表示範囲の計算（少しマージンを追加）
        x_min_reward, x_max_reward = min(all_x_values_reward), max(all_x_values_reward)
        y_min_reward, y_max_reward = min(all_y_values_reward), max(all_y_values_reward)
        x_margin_reward = (x_max_reward - x_min_reward) * 0.1
        y_margin_reward = (y_max_reward - y_min_reward) * 0.1
        x_range_reward = [x_min_reward - x_margin_reward, x_max_reward + x_margin_reward]
        y_range_reward = [y_min_reward - y_margin_reward, y_max_reward + y_margin_reward]
        
        # パレートフロントの進化を可視化（報酬 - 最大化目的）
        plt.figure(figsize=(15, 10))
        
        # 各評価時点のパレートフロントをプロット
        for i, (history, step) in enumerate(zip(self.evaluation_history, self.global_steps_at_evaluation)):
            pareto_front_reward = history['pareto_front_reward']
            plt.scatter(
                [ret[0] for ret in pareto_front_reward], 
                [ret[1] for ret in pareto_front_reward],
                color=colors[i], 
                label=f"Step {step}",
                alpha=0.7
            )
        
        plt.title("パレートフロントの進化（報酬 - 最大化目的）")
        plt.xlabel("報酬1")
        plt.ylabel("報酬2")
        plt.xlim(x_range_reward)
        plt.ylim(y_range_reward)
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        # 右上が最適な方向であることを示す矢印
        arrow_x_reward = x_min_reward + (x_max_reward - x_min_reward) * 0.15
        arrow_y_reward = y_min_reward + (y_max_reward - y_min_reward) * 0.15
        plt.annotate('最適方向', xy=(arrow_x_reward + (x_max_reward - x_min_reward) * 0.2, 
                                arrow_y_reward + (y_max_reward - y_min_reward) * 0.2), 
                    xytext=(arrow_x_reward, arrow_y_reward),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                    fontsize=12)
        
        # 固定ファイル名で保存（上書き）
        pareto_rewards_png_filename = f"{save_dir}/pareto_rewards_evolution.png"
        plt.savefig(pareto_rewards_png_filename)
        plt.close()
        
        # ------ アニメーション作成（実数値） ------
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def update_values(frame):
            ax.clear()
            history = self.evaluation_history[frame]
            pareto_front_values = history['pareto_front_values']
            all_returns = history['values']
            
            ax.scatter([ret[0] for ret in all_returns], [ret[1] for ret in all_returns], alpha=0.3, color='blue')
            ax.scatter([ret[0] for ret in pareto_front_values], [ret[1] for ret in pareto_front_values], color='red', s=80)
            
            ax.set_title(f"Step {self.global_steps_at_evaluation[frame]}でのパレートフロント（実数値 - 最小化目的）")
            ax.set_xlabel("コスト（実数）")
            ax.set_ylabel("makespan（実数）")
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.grid(True)
            
            # 左下が最適な方向であることを示す矢印
            arrow_x = x_min + (x_max - x_min) * 0.85
            arrow_y = y_min + (y_max - y_min) * 0.85
            ax.annotate('最適方向', xy=(arrow_x - (x_max - x_min) * 0.2, arrow_y - (y_max - y_min) * 0.2), 
                       xytext=(arrow_x, arrow_y),
                       arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                       fontsize=12)
        
        ani_values = FuncAnimation(fig, update_values, frames=len(self.evaluation_history), repeat=True)
        
        # 固定ファイル名でGIFを保存（上書き）
        pareto_values_gif_filename = f"{save_dir}/pareto_values_animation.gif"
        ani_values.save(pareto_values_gif_filename, writer='pillow', fps=2)
        plt.close()
        
        # ------ アニメーション作成（報酬） ------
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def update_rewards(frame):
            ax.clear()
            history = self.evaluation_history[frame]
            pareto_front_reward = history['pareto_front_reward']
            all_returns = history['all_returns']
            
            ax.scatter([ret[0] for ret in all_returns], [ret[1] for ret in all_returns], alpha=0.3, color='green')
            ax.scatter([ret[0] for ret in pareto_front_reward], [ret[1] for ret in pareto_front_reward], color='red', s=80)
            
            ax.set_title(f"Step {self.global_steps_at_evaluation[frame]}でのパレートフロント（報酬 - 最大化目的）")
            ax.set_xlabel("報酬1")
            ax.set_ylabel("報酬2")
            ax.set_xlim(x_range_reward)
            ax.set_ylim(y_range_reward)
            ax.grid(True)
            
            # 右上が最適な方向であることを示す矢印
            arrow_x_reward = x_min_reward + (x_max_reward - x_min_reward) * 0.15
            arrow_y_reward = y_min_reward + (y_max_reward - y_min_reward) * 0.15
            ax.annotate('最適方向', xy=(arrow_x_reward + (x_max_reward - x_min_reward) * 0.2, 
                                   arrow_y_reward + (y_max_reward - y_min_reward) * 0.2), 
                       xytext=(arrow_x_reward, arrow_y_reward),
                       arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                       fontsize=12)
        
        ani_rewards = FuncAnimation(fig, update_rewards, frames=len(self.evaluation_history), repeat=True)
        
        # 固定ファイル名でGIFを保存（上書き）
        pareto_rewards_gif_filename = f"{save_dir}/pareto_rewards_animation.gif"
        ani_rewards.save(pareto_rewards_gif_filename, writer='pillow', fps=2)
        plt.close()
        
        print(f"評価履歴の可視化を保存しました（最終更新: {timestamp}）:")
        print(f" - 実数値パレートフロント画像: {pareto_values_png_filename}")
        print(f" - 報酬パレートフロント画像: {pareto_rewards_png_filename}")
        print(f" - 実数値アニメーションGIF: {pareto_values_gif_filename}")
        print(f" - 報酬アニメーションGIF: {pareto_rewards_gif_filename}")

    def save_pareto_solutions_to_txt(self, mode_name="default"):
        """パレートフロントの解をテキストファイルに保存（単一ファイルで更新）"""
        if not self.evaluation_history:
            print("評価履歴がありません。ファイルは作成されませんでした。")
            return
        
        # 保存ディレクトリの作成
        save_dir = "pareto_solutions"
        os.makedirs(save_dir, exist_ok=True)
        
        # 固定ファイル名を使用（更新時に上書き）
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 最新の評価結果を取得
        latest_eval = self.evaluation_history[-1]
        
        # 結果をテキストファイルに書き込む（固定ファイル名で上書き）
        filename = f"{save_dir}/pareto_solutions_{mode_name}.txt"
        try:
            with open(filename, 'w') as f:
                # ヘッダー情報
                f.write(f"# パレートフロント解 - {mode_name}\n")
                f.write(f"# 最終更新日時: {timestamp}\n")
                f.write(f"# ステップ数: {self.global_step}\n")
                f.write(f"# 評価履歴数: {len(self.evaluation_history)}\n")
                f.write("\n")
                
                # パレートフロントデータ
                f.write("## パレートフロント\n")
                if 'pareto_front_values' in latest_eval:
                    pareto_front_values = latest_eval['pareto_front_values']
                    for i, solution in enumerate(pareto_front_values):
                        f.write(f"解 {i+1}: {solution}\n")
                f.write("\n")
                
                # 実際の評価値
                f.write("## 実際の評価値 (コスト, 実行時間)\n")
                if 'values' in latest_eval:
                    values = latest_eval['values']
                    for i, val in enumerate(values):
                        f.write(f"値 {i+1}: {val}\n")
                f.write("\n")
                
                # 全解のデータ
                f.write("## 全ての報酬\n")
                if 'all_returns' in latest_eval:
                    all_returns = latest_eval['all_returns']
                    for i, ret in enumerate(all_returns):
                        f.write(f"解 {i+1}: {ret}\n")
                
                # マップ情報はテキストでは表現しにくいので省略
                f.write("\n## マップデータは別途画像として保存されます\n")
            
            # マップデータの視覚化を別途保存（固定ファイル名で上書き）
            try:
                if 'maps' in latest_eval:
                    final_maps = latest_eval['maps']
                    map_image_path = f"{save_dir}/final_schedule_{mode_name}.png"
                    visualize_map(final_maps[0], final_maps[1], [], map_image_path)
                    print(f"スケジュールマップを保存しました: {map_image_path}")
            except Exception as map_err:
                print(f"マップ画像の保存中にエラーが発生しました: {map_err}")
            
            print(f"パレートフロントデータをテキストファイルに保存しました: {filename}")
            return filename
        except Exception as e:
            print(f"ファイルの保存中にエラーが発生しました: {e}")
            return None

    def extract_env_info(self, env: Optional[gym.Env]) -> None:
        """Extracts all the features of the environment: observation space, action space, ..."""
        if env is not None:
            self.env = env
            
            # 辞書型観測空間の処理
            if isinstance(self.env.unwrapped.observation_space, spaces.Dict):
                # 辞書型観測空間の場合
                self.observation_shape = None  # 形状は定義しない
                # 全サブスペースの要素数の合計を計算（より安全な方法）
                total_dim = 0
                for space in self.env.unwrapped.observation_space.spaces.values():
                    if hasattr(space, 'shape'):
                        total_dim += int(np.prod(space.shape))
                    elif hasattr(space, 'n'):
                        total_dim += space.n
                self.observation_dim = total_dim
                self.is_dict_obs = True
                
            # 離散型観測空間の処理
            elif isinstance(self.env.unwrapped.observation_space, spaces.Discrete):
                self.observation_shape = (1,)
                self.observation_dim = self.env.unwrapped.observation_space.n
                self.is_dict_obs = False
                
            # 連続型観測空間の処理
            else:
                self.observation_shape = self.env.unwrapped.observation_space.shape
                self.observation_dim = int(np.prod(self.observation_shape))
                self.is_dict_obs = False

            # 行動空間の処理
            self.action_space = env.unwrapped.action_space
            if isinstance(self.env.unwrapped.action_space, (spaces.Discrete, spaces.MultiBinary)):
                self.action_shape = (1,)
                self.action_dim = self.env.unwrapped.action_space.n
            else:
                self.action_shape = self.env.unwrapped.action_space.shape
                self.action_dim = int(np.prod(self.action_shape))
            
            # 報酬次元
            self.reward_dim = self.env.unwrapped.reward_space.shape[0]

    def cleanup_memory(self):
        """明示的にメモリをクリーンアップ"""
        import gc
        
        # 経験再生バッファの古いエピソードを削除
        if len(self.experience_replay) > 5000:  # バッファが大きすぎる場合
            # 古いエピソードを削除（下位50%を削除）
            sorted_buffer = sorted(self.experience_replay, key=lambda x: x[0])
            self.experience_replay = sorted_buffer[len(sorted_buffer)//2:]
            heapq.heapify(self.experience_replay)
            print(f"メモリクリーンアップ: 経験再生バッファを {len(sorted_buffer)} から {len(self.experience_replay)} に削減")
        
        # 評価履歴をクリア
        if len(self.evaluation_history) > 100:  # 履歴が多すぎる場合
            self.evaluation_history = self.evaluation_history[-50:]  # 最新50件のみ保持
            self.evaluation_timestamps = self.evaluation_timestamps[-50:]
            self.global_steps_at_evaluation = self.global_steps_at_evaluation[-50:]
            print(f"メモリクリーンアップ: 評価履歴を最新50件に削減")
        
        # ガベージコレクションを強制実行
        gc.collect()

    def __del__(self):
        """デストラクタ: メモリを確実に解放"""
        try:
            # 大きな配列を明示的に解放
            if hasattr(self, 'experience_replay'):
                del self.experience_replay
            if hasattr(self, 'evaluation_history'):
                del self.evaluation_history
            if hasattr(self, 'evaluation_timestamps'):
                del self.evaluation_timestamps
            if hasattr(self, 'global_steps_at_evaluation'):
                del self.global_steps_at_evaluation
        except:
            pass  # デストラクタでのエラーは無視

    def save_learning_data_to_file(self, filename="learning_data_debug.txt", sample_size=100):
        """学習データの詳細をファイルに書き込む
        
        Args:
            filename (str): 出力ファイル名
            sample_size (int): 分析するサンプル数
        """
        import os
        from datetime import datetime
        
        if len(self.experience_replay) == 0:
            print("⚠️  経験バッファが空です。データを収集してから実行してください。")
            return
        
        # ファイルパスを設定
        debug_dir = "debug_learning_data"
        os.makedirs(debug_dir, exist_ok=True)
        filepath = os.path.join(debug_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PCN学習データ詳細分析\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # 1. 全体統計
            f.write("1. 全体統計\n")
            f.write("-" * 40 + "\n")
            total_episodes = len(self.experience_replay)
            total_transitions = sum(len(episode[2]) for episode in self.experience_replay)
            f.write(f"総エピソード数: {total_episodes}\n")
            f.write(f"総遷移数: {total_transitions}\n")
            f.write(f"バッチサイズ: {self.batch_size}\n")
            f.write(f"割引率γ: {self.gamma}\n")
            f.write(f"学習率: {self.opt.param_groups[0]['lr']}\n")
            f.write(f"デバイス: {self.device}\n")
            f.write(f"モデルタイプ: {'EnhancedPCNModel' if self.use_enhanced_model else 'DefaultModel'}\n\n")
            
            # 2. サンプルデータの詳細分析
            f.write("2. サンプルデータ詳細分析\n")
            f.write("-" * 40 + "\n")
            
            # サンプルサイズを調整
            actual_sample_size = min(sample_size, total_episodes)
            sample_indices = np.random.choice(total_episodes, actual_sample_size, replace=False)
            
            all_observations = []
            all_actions = []
            all_desired_returns = []
            all_desired_horizons = []
            all_episode_lengths = []
            
            for i, idx in enumerate(sample_indices):
                episode = self.experience_replay[idx][2]
                episode_length = len(episode)
                all_episode_lengths.append(episode_length)
                
                # エピソード内のランダムなステップを選択
                t = np.random.randint(0, episode_length)
                transition = episode[t]
                
                # データを収集
                all_observations.append(transition.observation)
                all_actions.append(transition.action)
                
                # 論文通りの累積報酬計算
                remaining_return = 0.0
                for j in range(t, episode_length):
                    remaining_return += (self.gamma ** (j - t)) * episode[j].reward
                
                all_desired_returns.append(remaining_return)
                all_desired_horizons.append(episode_length - t)
                
                # 最初の10サンプルの詳細を記録
                if i < 10:
                    f.write(f"\nサンプル {i+1}:\n")
                    f.write(f"  エピソード長: {episode_length}\n")
                    f.write(f"  選択ステップ: {t}\n")
                    f.write(f"  観測形状: {transition.observation.shape}\n")
                    f.write(f"  観測値（最初の10要素）: {transition.observation}\n")
                    f.write(f"  行動: {transition.action}\n")
                    f.write(f"  即時報酬: {transition.reward}\n")
                    f.write(f"  累積報酬（論文通り）: {remaining_return}\n")
                    f.write(f"  残りステップ数: {episode_length - t}\n")
            
            # 3. 統計分析
            f.write("\n3. 統計分析\n")
            f.write("-" * 40 + "\n")
            
            # 観測データの統計
            obs_array = np.array(all_observations)
            f.write("観測データ統計:\n")
            f.write(f"  形状: {obs_array.shape}\n")
            f.write(f"  平均: {obs_array.mean():.6f}\n")
            f.write(f"  標準偏差: {obs_array.std():.6f}\n")
            f.write(f"  最小値: {obs_array.min():.6f}\n")
            f.write(f"  最大値: {obs_array.max():.6f}\n")
            f.write(f"  NaN数: {np.isnan(obs_array).sum()}\n")
            f.write(f"  Inf数: {np.isinf(obs_array).sum()}\n\n")
            
            # 行動分布
            actions_array = np.array(all_actions)
            unique_actions, action_counts = np.unique(actions_array, return_counts=True)
            f.write("行動分布:\n")
            for action, count in zip(unique_actions, action_counts):
                percentage = (count / len(actions_array)) * 100
                f.write(f"  行動{action}: {count}回 ({percentage:.1f}%)\n")
            
            # 不均衡チェック
            max_action_ratio = np.max(action_counts) / len(actions_array)
            f.write(f"  最大行動比率: {max_action_ratio:.3f}")
            if max_action_ratio > 0.8:
                f.write(" ⚠️  不均衡検出（80%以上が同じ行動）\n")
            else:
                f.write(" ✓ バランス良好\n")
            f.write("\n")
            
            # 累積報酬の統計
            returns_array = np.array(all_desired_returns)
            f.write("累積報酬統計（論文通り）:\n")
            f.write(f"  平均: {returns_array.mean():.6f}\n")
            f.write(f"  標準偏差: {returns_array.std():.6f}\n")
            f.write(f"  最小値: {returns_array.min():.6f}\n")
            f.write(f"  最大値: {returns_array.max():.6f}\n")
            f.write(f"  範囲: {returns_array.max() - returns_array.min():.6f}\n")
            f.write(f"  NaN数: {np.isnan(returns_array).sum()}\n")
            f.write(f"  Inf数: {np.isinf(returns_array).sum()}\n\n")
            
            # ホライゾンの統計
            horizons_array = np.array(all_desired_horizons)
            f.write("残りステップ数統計:\n")
            f.write(f"  平均: {horizons_array.mean():.1f}\n")
            f.write(f"  標準偏差: {horizons_array.std():.1f}\n")
            f.write(f"  最小値: {horizons_array.min()}\n")
            f.write(f"  最大値: {horizons_array.max()}\n\n")
            
            # エピソード長の統計
            episode_lengths_array = np.array(all_episode_lengths)
            f.write("エピソード長統計:\n")
            f.write(f"  平均: {episode_lengths_array.mean():.1f}\n")
            f.write(f"  標準偏差: {episode_lengths_array.std():.1f}\n")
            f.write(f"  最小値: {episode_lengths_array.min()}\n")
            f.write(f"  最大値: {episode_lengths_array.max()}\n\n")
            
            # 4. 学習データの品質チェック
            f.write("4. 学習データ品質チェック\n")
            f.write("-" * 40 + "\n")
            
            # データの有効性チェック
            issues = []
            if np.isnan(obs_array).any():
                issues.append("観測データにNaNが含まれています")
            if np.isinf(obs_array).any():
                issues.append("観測データにInfが含まれています")
            if np.isnan(returns_array).any():
                issues.append("累積報酬にNaNが含まれています")
            if np.isinf(returns_array).any():
                issues.append("累積報酬にInfが含まれています")
            if max_action_ratio > 0.9:
                issues.append("行動分布が極端に偏っています（90%以上が同じ行動）")
            if returns_array.std() < 1e-6:
                issues.append("累積報酬の分散が極めて小さいです")
            
            if issues:
                f.write("⚠️  検出された問題:\n")
                for issue in issues:
                    f.write(f"  - {issue}\n")
            else:
                f.write("✓ データ品質に問題は検出されませんでした\n")
            
            f.write("\n")
            
            # 5. 推奨事項
            f.write("5. 推奨事項\n")
            f.write("-" * 40 + "\n")
            
            if max_action_ratio > 0.8:
                f.write("- 行動分布の不均衡を解決するため、重み付き損失関数の使用を検討してください\n")
                f.write("- より多様な行動を生成するため、探索戦略の見直しを検討してください\n")
            
            if returns_array.std() < 1e-3:
                f.write("- 累積報酬の分散が小さいため、報酬設計の見直しを検討してください\n")
            
            if obs_array.std() > 100:
                f.write("- 観測データのスケールが大きいため、正規化の適用を検討してください\n")
            
            if len(unique_actions) < 2:
                f.write("- 行動の多様性が不足しています。環境設定の見直しを検討してください\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("分析完了\n")
            f.write("=" * 80 + "\n")
        
        print(f"✓ 学習データ分析結果を保存しました: {filepath}")
        return filepath

    def export_learning_samples_to_csv(self, filename="learning_samples.csv", num_samples=1000):
        """学習サンプルをCSVファイルにエクスポート
        
        Args:
            filename (str): 出力ファイル名
            num_samples (int): エクスポートするサンプル数
        """
        import pandas as pd
        import os
        from datetime import datetime
        
        if len(self.experience_replay) == 0:
            print("⚠️  経験バッファが空です。データを収集してから実行してください。")
            return
        
        # ファイルパスを設定
        debug_dir = "debug_learning_data"
        os.makedirs(debug_dir, exist_ok=True)
        filepath = os.path.join(debug_dir, filename)
        
        # サンプルデータを収集
        total_episodes = len(self.experience_replay)
        actual_samples = min(num_samples, total_episodes * 10)  # エピソードあたり最大10サンプル
        
        data_rows = []
        sample_count = 0
        
        for episode_idx in range(total_episodes):
            if sample_count >= actual_samples:
                break
                
            episode = self.experience_replay[episode_idx][2]
            episode_length = len(episode)
            
            # エピソードから複数サンプルを抽出
            num_episode_samples = min(10, episode_length)
            step_indices = np.random.choice(episode_length, num_episode_samples, replace=False)
            
            for t in step_indices:
                if sample_count >= actual_samples:
                    break
                    
                transition = episode[t]
                
                # 論文通りの累積報酬計算
                remaining_return = 0.0
                for j in range(t, episode_length):
                    remaining_return += (self.gamma ** (j - t)) * episode[j].reward
                
                # 観測データをフラット化
                obs_flat = transition.observation.flatten()
                
                # データ行を作成
                row = {
                    'sample_id': sample_count,
                    'episode_id': episode_idx,
                    'step_in_episode': t,
                    'episode_length': episode_length,
                    'action': transition.action,
                    'immediate_reward': transition.reward,
                    'cumulative_return': remaining_return,
                    'remaining_steps': episode_length - t,
                }
                
                # 観測データの各要素を追加
                for i, obs_val in enumerate(obs_flat):
                    row[f'observation_{i}'] = obs_val
                
                data_rows.append(row)
                sample_count += 1
        
        # DataFrameを作成してCSVに保存
        df = pd.DataFrame(data_rows)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"✓ 学習サンプルをCSVにエクスポートしました: {filepath}")
        print(f"  サンプル数: {len(df)}")
        print(f"  列数: {len(df.columns)}")
        
        return filepath
