"""フラグ台帳: 起動時に環境変数の整合を検査し、有効な構成を1箇所に印字する。

2026-08-25 のフラグ監査で、実害が出た不具合はすべて同じ形をしていた:
「フラグ A を立てるには B も必要だが、どこにもチェックが無く、黙って無効になる」。

- PCN_TEACH_FRONT_ONLY=1 は PCN_FROZEN_PF_CLONE=1 が前提(未設定→教師が replay 全件のまま)
- PCN_TRAIN_HEAD_STEP_WEIGHT はガード漏れで no-op
- PCN_EVAL_PF_GRID / PCN_EVAL_STOCHASTIC は読み手が存在しない
- XLA_PYTHON_CLIENT_MEM_FRACTION は PREALLOCATE=true でないと参照されない

ここで前提違反を検出したら「黙って無効」ではなく明示的に落とす(または警告する)。
既定は警告のみ(PCN_FLAG_AUDIT_STRICT=1 で例外送出)。検査自体を切るには
PCN_FLAG_AUDIT=0。
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

# 子フラグ -> 前提フラグ。前提が真でなければ子は無効(コードに到達しない/読まれない)。
REQUIRES: Dict[str, str] = {
    "PCN_TEACH_FRONT_ONLY": "PCN_FROZEN_PF_CLONE",   # pcn_agent.py:2324-2334 で早期return
    "PCN_FROZEN_PF_MAX": "PCN_FROZEN_PF_CLONE",      # 同上
    "PCN_COND_ADD_SCALE": "PCN_FILM",                # pcn_agent.py:819 の分岐内
    "PCN_FOURIER_BANDS": "PCN_FOURIER_CMD",
    "PCN_FOURIER_BANDS_COST": "PCN_FOURIER_CMD",
    "PCN_DEDUP_TRAIN_DECIMALS": "PCN_DEDUP_TRAIN_WEIGHT",
    "PCN_COMMAND_BALANCE_POWER": "PCN_COMMAND_BALANCE",
    "PCN_COMMAND_BALANCE_STEP": "PCN_COMMAND_BALANCE",
    "PCN_COMMAND_BALANCE_PMAX": "PCN_COMMAND_BALANCE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "XLA_PYTHON_CLIENT_PREALLOCATE",  # JAXはprealloc時のみ参照
    "PCN_CMD_TRACK_WAIT_WEIGHT": "PCN_CMD_TRACK_WEIGHT",  # cost側0だと関数ごと不発(pcn_agent)
    "PCN_COND_WAIT_Z0": "PCN_COND_WAIT_ROBUST",           # logexpandモード時のみ参照
}

# 読み手が存在しないフラグ(設定しても何も起きない)。
DEAD: Tuple[str, ...] = (
    "PCN_EVAL_PF_GRID",            # src/ に読み手なし
    "PCN_EVAL_STOCHASTIC",         # 同上
    "PCN_COMMAND_BALANCE_TARGET",  # 代入のみ・参照0件(HV山登り方式へ移行時の削除漏れ)
)

# ガード漏れ等で機能しないフラグ -> 説明。
BROKEN: Dict[str, str] = {
    "PCN_TRAIN_HEAD_STEP_WEIGHT":
        "_training_flat_step_weights の早期returnガードに HEAD が無く常に None を返す(no-op)。"
        "run_j20000_c3.sh / run_j50000_gpu.sh / run_jscale_c3.sh と v10 系の v9_env_export.sh は "
        "20 を設定しているが、これまでの全 run で効いていない。直すと学習レシピが変わるので "
        "整理整頓とは別コミットで(2026-08-27 調査)",
    "PCN_TRAIN_HEAD_STEP_FRAC":
        "PCN_TRAIN_HEAD_STEP_WEIGHT が no-op のため連鎖して無効",
}

# 明示しないとプロファイルの setdefault に取られる、影響の大きいフラグ -> 注意書き。
SILENT_DEFAULTS: Dict[str, str] = {
    "DISTRIBUTED_PCN_SUPERVISED_EPOCHS":
        "未設定だと workload_pcn_profile が 0 を入れ Phase2(教師あり)が全無効になる",
    "PCN_S_EMB_DROPOUT":
        "未設定だとプロファイルが 0.08 を入れる。Actorは model.eval() を呼ばないため "
        "学習中evalが非決定的になり、TorchScript高速化(PCN_JIT_ACT)も無効化される",
    "PCN_USE_AMP":
        "未設定だと AMP=ON。5万ジョブは報酬が 1e9 級で fp16 溢れの懸念があり、"
        "GradScaler の step スキップは検知しづらい",
}

_PREFIXES = ("PCN_", "SCHEDULER_", "DISTRIBUTED_PCN_", "XLA_")


def _truthy(name: str) -> bool:
    """フラグが「有効」か。数値なら 0/空以外、真偽なら 1/true を真とみなす。"""
    v = os.environ.get(name)
    if v is None:
        return False
    v = v.strip()
    if v == "" or v.lower() in ("0", "false", "off", "none"):
        return False
    try:
        return float(v) != 0.0
    except ValueError:
        return True


def audit_flags(strict: bool | None = None, echo: bool = True) -> List[str]:
    """整合を検査して問題のリストを返す。strict なら 1件でも例外を送出。"""
    if os.environ.get("PCN_FLAG_AUDIT", "1") != "1":
        return []
    if strict is None:
        strict = os.environ.get("PCN_FLAG_AUDIT_STRICT", "0") == "1"

    problems: List[str] = []
    for child, parent in REQUIRES.items():
        if _truthy(child) and not _truthy(parent):
            problems.append(
                f"{child}={os.environ.get(child)} は {parent} が有効であることが前提です"
                f"(現在 {parent}={os.environ.get(parent, '未設定')} → {child} は無効)")
    for name in DEAD:
        if name in os.environ:
            problems.append(f"{name} は読み手が存在しません(設定しても無効)")
    for name, why in BROKEN.items():
        if _truthy(name):
            problems.append(f"{name} は現状機能しません: {why}")

    if echo:
        active = {k: v for k, v in sorted(os.environ.items()) if k.startswith(_PREFIXES)}
        print(f"[FLAG_AUDIT] 有効フラグ {len(active)} 件:", flush=True)
        for k, v in active.items():
            print(f"    {k}={v}", flush=True)
        for name, why in SILENT_DEFAULTS.items():
            if name not in os.environ:
                print(f"[FLAG_AUDIT] ℹ️ {name} 未設定: {why}", flush=True)
        for p in problems:
            print(f"[FLAG_AUDIT] ⚠️ {p}", flush=True)
        if not problems:
            print("[FLAG_AUDIT] ✅ 前提フラグの不整合なし", flush=True)

    if strict and problems:
        raise ValueError("[FLAG_AUDIT] フラグ構成に不整合があります:\n  - "
                         + "\n  - ".join(problems))
    return problems
