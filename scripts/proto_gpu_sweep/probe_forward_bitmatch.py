"""方策forwardのビット一致性を実証する予備実験(Phase 2 設計の分岐点)。

問い: モデル forward の出力(logits)は
  T1: CPU バッチ(B,221) と CPU 単発(1,221)ループ で行ごとにビット一致するか
  T2: GPU バッチ と GPU 単発 で一致するか
  T3: CPU と GPU で一致するか(まず無理と予想。argmax 分裂率を測る)
結果次第で「バッチ版forwardをどのデバイスで回せば参照(既存evalの_act=単発)と
完全一致するか」が決まる。

usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/proto_gpu_sweep/probe_forward_bitmatch.py
"""
import os

os.environ.setdefault("PCN_FILM", "0")
os.environ.setdefault("PCN_FOURIER_CMD", "0")
os.environ.setdefault("PCN_OBS_LOG", "1")
os.environ.setdefault("PCN_HIDDEN_DIM", "512")

import numpy as np
import torch as th

from src.agents.pcn_agent import DiscreteActionsDefaultModel

CKPT = os.environ.get(
    "CKPT",
    "/home/noguchi/scheduler-sim-for-cb/experiments/distributed_pcn/"
    "run_synth512_nded512_r1/20260708_030028/iteration_100/model_iter_100.pth",
)
NJ = int(os.environ.get("NJ", "512"))
B = int(os.environ.get("B", "372"))
NSTEP = int(os.environ.get("NSTEP", "64"))  # 疑似ステップ数(入力を変えて反復)


def build(device):
    st = th.load(CKPT, map_location="cpu", weights_only=False)
    sd = st.get("model_state_dict", st)
    m = DiscreteActionsDefaultModel(
        221, 2, 2, np.array([1.0, 1.0, 1.0 / NJ], dtype=np.float32), hidden_dim=512
    )
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m.to(device)


def main():
    print(f"torch {th.__version__} matmul.allow_tf32={th.backends.cuda.matmul.allow_tf32}")
    rng = np.random.default_rng(0)
    # 現実的な入力: obs は [0,1] 221次元(イベント正規化+jq生値だが log 圧縮はモデル内)、
    # jq 生値は大きい(到着時刻~数万)ので一部次元は大きくする
    obs = rng.random((NSTEP, B, 221)).astype(np.float32)
    obs[:, :, 180:220] *= 50000.0  # job_queue 生値相当
    dr = np.stack(
        [
            rng.uniform(-3.0e6, 0, (NSTEP, B)).astype(np.float32),
            rng.uniform(-3.3e6, 0, (NSTEP, B)).astype(np.float32),
        ],
        axis=2,
    )
    hz = rng.uniform(1, NJ, (NSTEP, B, 1)).astype(np.float32)

    m_cpu = build("cpu")
    m_gpu = build("cuda")

    def run_batch(m, device):
        outs = []
        with th.no_grad():
            for t in range(NSTEP):
                o = th.tensor(obs[t], device=device)
                d = th.tensor(dr[t], device=device)
                h = th.tensor(hz[t], device=device)
                outs.append(m(o, d, h).cpu().numpy())
        return np.stack(outs)

    def run_b1(m, device):
        outs = np.zeros((NSTEP, B, 2), dtype=np.float32)
        with th.no_grad():
            for t in range(NSTEP):
                for b in range(B):
                    o = th.tensor(obs[t, b : b + 1], device=device)
                    d = th.tensor(dr[t, b : b + 1], device=device)
                    h = th.tensor(hz[t, b : b + 1], device=device)
                    outs[t, b] = m(o, d, h).cpu().numpy()[0]
        return outs

    cpu_b = run_batch(m_cpu, "cpu")
    cpu_1 = run_b1(m_cpu, "cpu")
    gpu_b = run_batch(m_gpu, "cuda")
    gpu_1 = run_b1(m_gpu, "cuda")

    def cmp(name, a, b):
        eq = np.array_equal(a.view(np.uint32), b.view(np.uint32))
        mx = np.abs(a.astype(np.float64) - b.astype(np.float64)).max()
        aa = a.argmax(axis=2)
        ab = b.argmax(axis=2)
        nflip = int((aa != ab).sum())
        # argmax 近接度: 分裂した箇所の logit ギャップ
        print(f"{name}: bit_equal={eq} max|d|={mx:.3g} argmax_flips={nflip}/{aa.size}")

    cmp("T1 cpu_batch vs cpu_b1 ", cpu_b, cpu_1)
    cmp("T2 gpu_batch vs gpu_b1 ", gpu_b, gpu_1)
    cmp("T3 cpu_b1   vs gpu_b1 ", cpu_1, gpu_1)
    cmp("T3'cpu_batch vs gpu_batch", cpu_b, gpu_b)


if __name__ == "__main__":
    main()
