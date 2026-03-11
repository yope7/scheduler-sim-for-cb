#!/usr/bin/env python3
"""
PyTorch vs JAX の学習更新ベンチマーク

PCNのLearner updateに相当するワークロード（forward + backward + optimizer step）を
PyTorch と JAX で比較し、JAXへの移行が高速化に寄与するか検討する。

使用方法:
  uv run python scripts/benchmark_pytorch_vs_jax.py
  uv run python scripts/benchmark_pytorch_vs_jax.py --device cuda   # GPU使用時
  uv run python scripts/benchmark_pytorch_vs_jax.py --warmup 20 --iters 100
  uv run python scripts/benchmark_pytorch_vs_jax.py --profile       # プロファイル取得
"""
import argparse
import cProfile
import os
import pstats
import sys
import time

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def benchmark_pytorch(state_dim: int, batch_size: int, hidden_dim: int, n_iters: int, warmup: int, device: str):
    """PyTorch版: PCNと同様のMLP forward + backward + step"""
    import torch
    import torch.nn as nn

    torch.manual_seed(42)
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    class SimplePCNLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.s_emb = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Sigmoid())
            self.c_emb = nn.Sequential(nn.Linear(3, hidden_dim), nn.Sigmoid())  # reward_dim+1
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
                nn.LogSoftmax(dim=1),
            )

        def forward(self, obs, desired_return, desired_horizon):
            s = self.s_emb(obs)
            c = torch.cat([desired_return, desired_horizon], dim=1)
            cond = self.c_emb(c)
            x = s * cond
            return self.fc(x)

    model = SimplePCNLike().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ダミーデータ
    obs = torch.randn(batch_size, state_dim, dtype=torch.float32, device=dev)
    desired_return = torch.randn(batch_size, 2, dtype=torch.float32, device=dev)
    desired_horizon = torch.randn(batch_size, 1, dtype=torch.float32, device=dev)
    actions = torch.randint(0, 2, (batch_size,), device=dev)

    # Warmup
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        logits = model(obs, desired_return, desired_horizon)
        loss = nn.functional.nll_loss(logits, actions)
        loss.backward()
        opt.step()

    if dev.type == "cuda":
        torch.cuda.synchronize()

    # 計測
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        logits = model(obs, desired_return, desired_horizon)
        loss = nn.functional.nll_loss(logits, actions)
        loss.backward()
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return np.mean(times) * 1000, np.std(times) * 1000


def benchmark_jax(state_dim: int, batch_size: int, hidden_dim: int, n_iters: int, warmup: int, device: str):
    """JAX版: 同等のMLP forward + backward (grad) + optimizer step"""
    try:
        import jax
        import jax.numpy as jnp
        from jax import grad, jit, value_and_grad
    except ImportError:
        return None, None, "JAX not installed (uv add jax jaxlib)"

    def init_params(key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        return {
            "s_w": jax.random.normal(k1, (state_dim, hidden_dim)) * 0.02,
            "s_b": jnp.zeros(hidden_dim),
            "c_w": jax.random.normal(k2, (3, hidden_dim)) * 0.02,
            "c_b": jnp.zeros(hidden_dim),
            "fc1_w": jax.random.normal(k3, (hidden_dim, hidden_dim)) * 0.02,
            "fc1_b": jnp.zeros(hidden_dim),
            "fc2_w": jax.random.normal(k4, (hidden_dim, 2)) * 0.02,
            "fc2_b": jnp.zeros(2),
        }

    def forward(params, obs, desired_return, desired_horizon):
        s = jax.nn.sigmoid(jnp.dot(obs, params["s_w"]) + params["s_b"])
        c = jnp.concatenate([desired_return, desired_horizon], axis=1)
        cond = jax.nn.sigmoid(jnp.dot(c, params["c_w"]) + params["c_b"])
        x = s * cond
        h = jax.nn.relu(jnp.dot(x, params["fc1_w"]) + params["fc1_b"])
        logits = jnp.dot(h, params["fc2_w"]) + params["fc2_b"]
        return logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)

    def loss_fn(params, obs, desired_return, desired_horizon, actions):
        logits = forward(params, obs, desired_return, desired_horizon)
        return -jnp.mean(jnp.sum(jax.nn.one_hot(actions, 2) * logits, axis=1))

    @jit
    def step(params, opt_state, obs, desired_return, desired_horizon, actions):
        val, grads = value_and_grad(loss_fn)(params, obs, desired_return, desired_horizon, actions)
        updates, opt_state = opt_update(grads, opt_state)
        params = jax.tree_util.tree_map(lambda p, u: p - 1e-3 * u, params, updates)
        return params, opt_state, val

    # 簡易SGD（optaxを使うとより公平だが依存追加になる）
    def opt_update(grads, opt_state):
        return grads, opt_state  # 単純なSGDとして grads をそのまま返す

    key = jax.random.PRNGKey(42)
    params = init_params(key)
    opt_state = None

    obs = jax.random.normal(key, (batch_size, state_dim)).astype(jnp.float32)
    desired_return = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 2)).astype(jnp.float32)
    desired_horizon = jax.random.normal(jax.random.PRNGKey(2), (batch_size, 1)).astype(jnp.float32)
    actions = jax.random.randint(jax.random.PRNGKey(3), (batch_size,), 0, 2)

    # JIT compile (warmup)
    for _ in range(warmup):
        params, opt_state, _ = step(params, opt_state, obs, desired_return, desired_horizon, actions)

    # 計測（JAXは非同期のため block_until_ready で完了を待つ）
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        params, opt_state, val = step(params, opt_state, obs, desired_return, desired_horizon, actions)
        jax.block_until_ready(params)  # 計算完了を保証（params が最後に更新される）
        times.append(time.perf_counter() - t0)

    return np.mean(times) * 1000, np.std(times) * 1000, None


def main():
    parser = argparse.ArgumentParser(description="PyTorch vs JAX 学習更新ベンチマーク")
    parser.add_argument("--state_dim", type=int, default=38440, help="観測次元（PCN small: 38440）")
    parser.add_argument("--batch_size", type=int, default=1024, help="バッチサイズ")
    parser.add_argument("--hidden_dim", type=int, default=256, help="隠れ層次元")
    parser.add_argument("--warmup", type=int, default=10, help="ウォームアップ回数")
    parser.add_argument("--iters", type=int, default=50, help="計測反復回数")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="PyTorchデバイス")
    parser.add_argument("--profile", action="store_true", help="cProfileでプロファイル取得")
    args = parser.parse_args()

    print("=" * 70)
    print("PyTorch vs JAX 学習更新ベンチマーク")
    print(f"  state_dim={args.state_dim}, batch_size={args.batch_size}, hidden_dim={args.hidden_dim}")
    print(f"  warmup={args.warmup}, iters={args.iters}, device={args.device}")
    print("=" * 70)

    def run_benchmarks():
        print("\n[PyTorch] 計測中...")
        pt_mean, pt_std = benchmark_pytorch(
            args.state_dim, args.batch_size, args.hidden_dim,
            args.iters, args.warmup, args.device
        )
        print(f"  PyTorch: {pt_mean:.2f} ± {pt_std:.2f} ms/update")
        print("\n[JAX] 計測中...")
        jax_result = benchmark_jax(
            args.state_dim, args.batch_size, args.hidden_dim,
            args.iters, args.warmup, args.device
        )
        return pt_mean, pt_std, jax_result

    if args.profile:
        prof = cProfile.Profile()
        prof.enable()
        pt_mean, pt_std, jax_result = run_benchmarks()
        prof.disable()
        stats = pstats.Stats(prof)
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        print("\n--- プロファイル (cumulative time, top 30) ---")
        stats.print_stats(30)
    else:
        pt_mean, pt_std, jax_result = run_benchmarks()
    if len(jax_result) == 3 and jax_result[2] is not None:
        print(f"  JAX: {jax_result[2]}")
    else:
        jax_mean, jax_std, _ = jax_result
        print(f"  JAX: {jax_mean:.2f} ± {jax_std:.2f} ms/update")

        speedup = pt_mean / jax_mean if jax_mean > 0 else 0
        print(f"\n  → JAX が {speedup:.2f}x {'速い' if speedup > 1 else '遅い'}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
