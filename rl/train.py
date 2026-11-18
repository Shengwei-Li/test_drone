"""用 PPO 训练四旋翼的端到端策略(状态 → 四个电机指令)。

用法:  python -m rl.train                     默认约 4000 万步
       python -m rl.train --iters 50          短训练,试跑用
       python -m rl.train --device cpu

PPO(近端策略优化)的流程,每一轮:
  1. 让当前策略在几千个并行环境里各飞 32 步,记下 观测/动作/奖励;
  2. 用价值网络估计"每一步比预期好多少"(优势,GAE);
  3. 调整策略,让"比预期好"的动作更可能出现,但每轮的改动幅度有上限(clip)。

训练完把策略网络的权重存到 rl/policy.npz,由 sim/rl_policy.py 用 numpy 加载。
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rl.env import QuadEnv
from sim.rl_policy import OBS_DIM

HERE = Path(__file__).parent


def mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.Tanh()]
    return nn.Sequential(*layers[:-1])


class ActorCritic(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.actor = mlp([OBS_DIM, hidden, hidden, 4])          # 小网络:要能放进单片机
        self.critic = mlp([OBS_DIM, 256, 256, 1])               # 价值网络只在训练时用,可以大一些
        self.log_std = nn.Parameter(torch.full((4,), -0.7))
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)        # 初始动作≈0,即悬停转速
        nn.init.zeros_(self.actor[-1].bias)

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


def export(model, path):
    lin = [m for m in model.actor if isinstance(m, nn.Linear)]
    np.savez(path, **{f"W{i}": m.weight.detach().cpu().numpy().astype(np.float64) for i, m in enumerate(lin)},
             **{f"b{i}": m.bias.detach().cpu().numpy().astype(np.float64) for i, m in enumerate(lin)})


def train(iters=300, n_envs=4096, horizon=32, epochs=4, minibatches=8, lr=3e-4, gamma=0.99, lam=0.95,
          clip=0.2, device=None, seed=0, out=HERE / "policy.npz", log=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    env = QuadEnv(n_envs, device, seed=seed)
    model = ActorCritic().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    obs = env.obs()
    shape = (horizon, n_envs)
    b_obs = torch.zeros(*shape, OBS_DIM, device=device)
    b_act = torch.zeros(*shape, 4, device=device)
    b_logp, b_rew, b_val, b_done = (torch.zeros(*shape, device=device) for _ in range(4))
    history, t0 = [], time.time()

    for it in range(iters):
        opt.param_groups[0]["lr"] = lr * (1.0 - 0.9 * it / iters)
        dist_sum = 0.0
        with torch.no_grad():
            for t in range(horizon):
                d = model.dist(obs)
                act = d.sample()
                b_obs[t], b_act[t], b_val[t] = obs, act, model.value(obs)
                b_logp[t] = d.log_prob(act).sum(-1)
                obs, rew, term, trunc, final_obs, dist = env.step(act)
                # 因为时间到而结束的回合,后面本来还有收益:用价值网络的估计补上
                rew = rew + gamma * model.value(final_obs) * trunc
                b_rew[t], b_done[t] = rew, (term | trunc).float()
                dist_sum += dist.mean().item()
            # 优势估计 (GAE)
            adv = torch.zeros_like(b_rew)
            last, next_val = torch.zeros(n_envs, device=device), model.value(obs)
            for t in reversed(range(horizon)):
                nonterminal = 1.0 - b_done[t]
                delta = b_rew[t] + gamma * next_val * nonterminal - b_val[t]
                last = delta + gamma * lam * nonterminal * last
                adv[t], next_val = last, b_val[t]
            ret = adv + b_val

        f_obs, f_act = b_obs.reshape(-1, OBS_DIM), b_act.reshape(-1, 4)
        f_logp, f_adv, f_ret = b_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        n = f_obs.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for idx in perm.chunk(minibatches):
                d = model.dist(f_obs[idx])
                logp = d.log_prob(f_act[idx]).sum(-1)
                ratio = (logp - f_logp[idx]).exp()
                a = f_adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                loss_pi = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
                loss_v = 0.5 * (model.value(f_obs[idx]) - f_ret[idx]).pow(2).mean()
                loss = loss_pi + 0.5 * loss_v - 0.001 * d.entropy().sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        row = dict(iter=it + 1, steps=(it + 1) * n, reward=b_rew.mean().item(),
                   dist_cm=100 * dist_sum / horizon, std=model.log_std.exp().mean().item(),
                   crash_per_1k=1000 * (b_done.sum().item() / n), sec=time.time() - t0)
        history.append(row)
        if log and (it % 10 == 9 or it == 0 or it == iters - 1):
            print(f"轮 {row['iter']:4d}  步数 {row['steps'] / 1e6:6.1f}M  奖励 {row['reward']:.3f}  "
                  f"平均距离 {row['dist_cm']:5.1f} cm  结束率 {row['crash_per_1k']:.2f}/千步  "
                  f"动作噪声 {row['std']:.2f}  用时 {row['sec']:.0f}s", flush=True)
        if it % 50 == 49 or it == iters - 1:
            export(model, out)
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "policy.npz"))
    args = ap.parse_args()
    _, hist = train(iters=args.iters, n_envs=args.envs, device=args.device, seed=args.seed, out=Path(args.out))
    import json
    (HERE / "train_log.json").write_text(json.dumps(hist), encoding="utf-8")
    print(f"策略已保存到 {args.out}")


if __name__ == "__main__":
    main()
