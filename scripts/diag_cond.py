"""Standalone: can DiscreteActionsDefaultModel learn return->action conditioning at 1024 scale?
Isolates the conditioning pathway with identical obs, only the command differs."""
import sys
import numpy as np
import torch as th
import torch.nn.functional as F
import src.agents.pcn_agent as m

th.manual_seed(0); np.random.seed(0)
OBS_DIM = 220

def run(nj, wait_scale, cost_scale, obs_max, label):
    model = m.DiscreteActionsDefaultModel(OBS_DIM, 2, 2, np.array([1.0, 1.0, 1.0/nj]), 512)
    # set normalization like the real learner (center 0, scale = reach range per objective)
    model.set_desired_return_normalization(np.zeros(2, dtype=np.float32),
                                           np.array([wait_scale, cost_scale], dtype=np.float32))
    model.train()
    opt = th.optim.Adam(model.parameters(), lr=1e-2)
    # realistic obs: mostly small (0..12), a few large (~obs_max). SAME obs for both classes.
    B = 256
    obs = th.zeros(B, OBS_DIM)
    obs[:, :200] = th.randint(0, 12, (B, 200)).float()
    obs[:, 200:205] = obs_max  # a few large processing-time-like features
    # two command classes: onprem (high wait, 0 cost)->action0 ; cloud (0 wait, high cost)->action1
    half = B // 2
    cmd = th.zeros(B, 2)
    cmd[:half, 0] = -wait_scale; cmd[:half, 1] = 0.0          # onprem -> action 0
    cmd[half:, 0] = 0.0;          cmd[half:, 1] = -cost_scale  # cloud  -> action 1
    act = th.zeros(B, dtype=th.long); act[half:] = 1
    hor = th.full((B, 1), float(nj))
    for step in range(800):
        opt.zero_grad()
        logits = model(obs, cmd, hor)
        loss = F.nll_loss(logits, act)
        loss.backward(); opt.step()
    with th.no_grad():
        logits = model(obs, cmd, hor)
        pred = logits.argmax(1)
        acc = (pred == act).float().mean().item()
        # inspect c_emb saturation
        cn = (cmd - model.desired_return_center) / model.desired_return_scale
        cfull = th.cat((cn, hor * (1.0/nj)), dim=-1)
        cemb = model.c_emb(cfull)
        print(f"[{label}] nj={nj} final loss={loss.item():.4f} acc={acc:.3f}  "
              f"norm_cmd[onprem]={cn[0].tolist()} [cloud]={cn[-1].tolist()}  "
              f"c_emb std across classes={float((model.c_emb(cfull[:half]).mean(0)-model.c_emb(cfull[half:]).mean(0)).abs().mean()):.4f}")

# 24-job-like (works in practice) vs 1024-job-like (stuck in practice)
run(24,   1581.0*24,   962664.0,    4332.0, "24J-like")
run(1024, 1.64e6*1024, 1.83e9,     14405.0, "1024-like")
