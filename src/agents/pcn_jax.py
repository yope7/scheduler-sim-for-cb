"""
JAX/Flax 版 PCN モデル（DiscreteActionsDefaultModel と同等）

DISTRIBUTED_PCN_USE_JAX=1 で Learner が JAX を使用。
Actor は PyTorch のため、get_weights で JAX params を PyTorch state_dict に変換して返す。
"""
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    import optax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def _sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-jnp.clip(x, -20, 20)))


class PCNModelJAX(nn.Module):
    """DiscreteActionsDefaultModel の JAX/Flax 版"""
    state_dim: int
    action_dim: int
    reward_dim: int
    hidden_dim: int
    scaling_factor: np.ndarray

    @nn.compact
    def __call__(self, state, desired_return, desired_horizon):
        # クリッピング
        desired_return = jnp.clip(desired_return, -1000.0, 1000.0)
        desired_horizon = jnp.clip(desired_horizon, 0.0, 1000.0)
        state = jnp.clip(state.astype(jnp.float32), -1000.0, 1000.0)

        # 条件ベクトル: [desired_return (2), desired_horizon (1)] -> (3,)
        c = jnp.concatenate([desired_return, desired_horizon], axis=-1)
        c = c * jnp.array(self.scaling_factor, dtype=jnp.float32)

        # s_emb: Linear(state_dim, hidden_dim) + Sigmoid
        s = nn.Dense(self.hidden_dim, use_bias=True, name='s_emb')(state)
        s = _sigmoid(s)

        # c_emb: Linear(3, hidden_dim) + Sigmoid
        c_emb = nn.Dense(self.hidden_dim, use_bias=True, name='c_emb')(c)
        c_emb = _sigmoid(c_emb)

        # 要素積
        x = s * c_emb

        # fc: Linear -> ReLU -> Linear -> LogSoftmax
        x = nn.Dense(self.hidden_dim, use_bias=True, name='fc0')(x)
        x = jax.nn.relu(x)
        logits = nn.Dense(self.action_dim, use_bias=True, name='fc1')(x)
        return jax.nn.log_softmax(logits, axis=-1)


def init_model(state_dim, action_dim, reward_dim, hidden_dim, scaling_factor, key):
    """モデルを初期化"""
    model = PCNModelJAX(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        hidden_dim=hidden_dim,
        scaling_factor=np.array(scaling_factor, dtype=np.float32),
    )
    # ダミー入力で初期化
    key1, key2 = jax.random.split(key)
    dummy_state = jax.random.normal(key1, (2, state_dim))
    dummy_return = jax.random.normal(key2, (2, reward_dim))
    dummy_horizon = jnp.ones((2, 1))
    params = model.init(key, dummy_state, dummy_return, dummy_horizon)
    return model, params


def jax_params_to_pytorch_state_dict(params, scaling_factor=None):
    """
    Flax params を PyTorch state_dict 形式に変換。
    DiscreteActionsDefaultModel の構造に合わせる。
    """
    import torch as th
    if scaling_factor is None:
        scaling_factor = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    p = params.get('params', params)

    def to_pt(kernel, bias):
        return (
            th.from_numpy(np.array(kernel).T.copy()),
            th.from_numpy(np.array(bias).copy()),
        )

    state_dict = {}
    state_dict['scaling_factor'] = th.from_numpy(np.array(scaling_factor, dtype=np.float32))
    w, b = to_pt(p['s_emb']['kernel'], p['s_emb']['bias'])
    state_dict['s_emb.0.weight'] = w
    state_dict['s_emb.0.bias'] = b
    w, b = to_pt(p['c_emb']['kernel'], p['c_emb']['bias'])
    state_dict['c_emb.0.weight'] = w
    state_dict['c_emb.0.bias'] = b
    w, b = to_pt(p['fc0']['kernel'], p['fc0']['bias'])
    state_dict['fc.0.weight'] = w
    state_dict['fc.0.bias'] = b
    w, b = to_pt(p['fc1']['kernel'], p['fc1']['bias'])
    state_dict['fc.2.weight'] = w
    state_dict['fc.2.bias'] = b
    return state_dict


def pytorch_state_dict_to_jax_params(state_dict):
    """PyTorch state_dict を Flax params に変換"""
    return {
        'params': {
            's_emb': {
                'kernel': np.array(state_dict['s_emb.0.weight'].T.cpu().numpy()),
                'bias': np.array(state_dict['s_emb.0.bias'].cpu().numpy()),
            },
            'c_emb': {
                'kernel': np.array(state_dict['c_emb.0.weight'].T.cpu().numpy()),
                'bias': np.array(state_dict['c_emb.0.bias'].cpu().numpy()),
            },
            'fc0': {
                'kernel': np.array(state_dict['fc.0.weight'].T.cpu().numpy()),
                'bias': np.array(state_dict['fc.0.bias'].cpu().numpy()),
            },
            'fc1': {
                'kernel': np.array(state_dict['fc.2.weight'].T.cpu().numpy()),
                'bias': np.array(state_dict['fc.2.bias'].cpu().numpy()),
            },
        }
    }
