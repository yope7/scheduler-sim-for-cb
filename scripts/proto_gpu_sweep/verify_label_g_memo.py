"""factory_episode の label_g メモ化修正がビット一致かを検証する。

背景: FactoryArrayEpisode._training_block() に "label_g" キーが無く、
pcn_agent._encode_episode_training_block のメモ化ヒット条件
(`_cached.get("label_g", False) == _LABEL_G`) が PCN_LABEL_G=1 で必ず外れていた。
結果、per-step Python ループ(1本 0.9 秒)に落ちていた。

本検証は「メモ化ブロック(factory 側で事前計算)」と「メモを無効化して encode 経路で
計算したブロック」を突き合わせ、observations/actions/desired_returns/desired_horizons
の全要素が**ビット一致**することを確認する。

usage:
  PCN_LABEL_G=1 PYTHONPATH=. .venv/bin/python scripts/proto_gpu_sweep/verify_label_g_memo.py
  PCN_LABEL_G=0 ... でも同様に一致すること(既定挙動の不変を確認)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main() -> int:
    from src.agents.pcn_agent import PCN, _LABEL_G
    from src.distributed.factory_episode import build_factory_array_episodes

    rng = np.random.default_rng(0)
    B, T, D, R = 3, 400, 8, 2

    # 工場出力を模した res(実データと同じ dtype/形状)
    res = {
        "obs": rng.normal(size=(B, T + 1, D)).astype(np.float32),
        "actions": rng.integers(0, 2, size=(B, T)).astype(np.int8),
        "rewards": rng.normal(size=(B, T, R)).astype(np.float32) * 1000.0,
        "achieved": np.stack([
            rng.integers(0, 10**9, size=B),
            rng.integers(0, 10**6, size=B),
            rng.normal(size=B) * 100,
        ], axis=1).astype(np.float64),
        "episode_ids": np.arange(B, dtype=np.int64),
        "seed0": 0,
    }
    eps = build_factory_array_episodes(res, uids=[f"verify:{i}" for i in range(B)])

    # PCN インスタンス(encode 経路を呼ぶためだけ。env は不要)
    agent = PCN.__new__(PCN)
    agent.gamma = 1.0

    ok = True
    for i, ep in enumerate(eps):
        memo = ep._training_block()                       # 工場が事前計算したブロック
        head0 = ep[0]
        setattr(head0, "_pcn_training_block", None)       # メモを外して encode 経路を強制
        enc = agent._encode_episode_training_block(ep)    # per-step ループで再計算

        for key in ("observations", "actions", "desired_returns", "desired_horizons"):
            a, b = np.asarray(memo[key]), np.asarray(enc[key])
            if a.shape != b.shape or a.dtype != b.dtype:
                print(f"  ep{i} {key}: 形状/型不一致 {a.shape}{a.dtype} vs {b.shape}{b.dtype}")
                ok = False
                continue
            n_diff = int((a != b).sum())
            if n_diff:
                d = np.abs(a.astype(np.float64) - b.astype(np.float64))
                print(f"  ep{i} {key}: 不一致 {n_diff}/{a.size} 要素 (max|Δ|={d.max():.3e})")
                ok = False
        if memo.get("label_g") != _LABEL_G:
            print(f"  ep{i} label_g キー不一致: {memo.get('label_g')} vs {_LABEL_G}")
            ok = False
        if memo.get("episode_length") != enc.get("episode_length"):
            print(f"  ep{i} episode_length 不一致")
            ok = False

    print(f"[verify_label_g_memo] PCN_LABEL_G={int(_LABEL_G)} B={B} T={T}: "
          f"{'✅ 全要素ビット一致' if ok else '❌ 不一致あり'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
