"""批量并行的四旋翼环境(PyTorch,可在 GPU 上同时跑几千架)。

动力学方程与 sim/dynamics.py 相同,只是:
- 所有量多了一个"第几架"的维度;
- 用 1 ms 的半隐式欧拉积分代替 RK4(便宜 4 倍,tests/test_rl.py 里与 RK4 对过);
- 每一架的质量、惯量、电机强度、电机响应、风都随机(域随机化),
  让策略学会应付一族机体,而不是只会飞标称模型。

任务:跟踪一个随机的参考点(静止,或沿随机的正弦轨迹运动)。
策略 100 Hz 输出四个电机的转速指令,观测见 sim/rl_policy.py。
"""
import math

import torch

from sim.params import QuadParams
from sim.rl_policy import (ACT_SCALE, INT_CLIP, INT_RADIUS, OBS_DIM, POLICY_DT, S_ACC, S_INT, S_OMEGA, S_POS,
                           S_VEL)


def quat_to_rot(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], -1)


class QuadEnv:
    def __init__(self, n: int, device="cpu", params: QuadParams | None = None, sim_dt: float = 0.001,
                 episode_s: float = 5.0, randomize: bool = True, seed: int = 0):
        self.n, self.dev = n, torch.device(device)
        self.p = params or QuadParams()
        self.sim_dt = sim_dt
        self.substeps = round(POLICY_DT / sim_dt)
        self.max_steps = round(episode_s / POLICY_DT)
        self.randomize = randomize
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        f = dict(device=self.dev, dtype=torch.float32)
        self.alloc = torch.tensor(self.p.alloc, **f)             # (4, 4)
        self.J0 = torch.tensor(self.p.inertia, **f)
        z = lambda *s: torch.zeros(n, *s, **f)                    # noqa: E731
        self.pos, self.vel, self.quat, self.omega, self.motor = z(3), z(3), z(4), z(3), z(4)
        self.mass, self.J, self.gain, self.tau, self.wind = z(), z(3), z(4), z(), z(3)
        self.amp, self.freq, self.phase = z(3), z(3), z(3)
        self.t, self.steps = z(), torch.zeros(n, dtype=torch.long, device=self.dev)
        self.prev_act, self.integ = z(4), z(3)
        self.reset_idx(torch.arange(n, device=self.dev))

    # ---- 随机数 ----
    def _u(self, k, lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(k, *shape, device=self.dev, generator=self.gen)

    def _n(self, k, *shape):
        return torch.randn(k, *shape, device=self.dev, generator=self.gen)

    # ---- 参考轨迹:每个轴一条正弦 ----
    def reference(self):
        arg = self.freq * self.t[:, None] + self.phase
        s = self.amp * torch.sin(arg)
        return s, self.amp * self.freq * torch.cos(arg), -self.freq ** 2 * s

    def reset_idx(self, idx):
        k, p = len(idx), self.p
        if k == 0:
            return
        r = self.randomize
        self.mass[idx] = p.mass * (self._u(k, 0.8, 1.2) if r else 1.0)
        self.J[idx] = self.J0 * (self._u(k, 0.8, 1.2, 3) if r else 1.0)
        self.gain[idx] = self._u(k, 0.85, 1.15, 4) if r else 1.0
        self.tau[idx] = p.motor_tau * (self._u(k, 0.7, 1.3) if r else 1.0)
        self.wind[idx] = (self._u(k, -0.04, 0.04, 3) * torch.tensor([1.0, 1.0, 0.3], device=self.dev)
                          if r else 0.0)
        # 参考:30% 静止,其余是随机正弦(幅值 ≤ 0.7 m,角频率 ≤ 3.5 rad/s)
        moving = (self._u(k, 0, 1) > 0.3).float()[:, None]
        self.amp[idx] = moving * self._u(k, 0.0, 0.7, 3) * torch.tensor([1.0, 1.0, 0.4], device=self.dev)
        self.freq[idx] = self._u(k, 0.3, 3.5, 3)
        self.phase[idx] = self._u(k, 0.0, 2 * math.pi, 3)
        self.t[idx] = 0.0
        self.steps[idx] = 0
        # 初始状态:离参考点最远 1.2 m,有初速度,姿态最多倾斜 60°(10% 是任意姿态,模拟被抛出)
        d = self._n(k, 3)
        d = d / d.norm(dim=-1, keepdim=True) * self._u(k, 0.0, 1.2)[:, None]
        self.pos[idx] = self.amp[idx] * torch.sin(self.phase[idx]) + d
        self.vel[idx] = self._u(k, -1.0, 1.0, 3)
        axis = self._n(k, 3)
        axis = axis / axis.norm(dim=-1, keepdim=True)
        wild = self._u(k, 0, 1) < 0.1
        ang = torch.where(wild, self._u(k, 0.0, math.pi), self._u(k, 0.0, math.pi / 3))
        self.quat[idx] = torch.cat([torch.cos(ang / 2)[:, None], axis * torch.sin(ang / 2)[:, None]], -1)
        self.omega[idx] = self._u(k, -3.0, 3.0, 3) * torch.where(wild, 3.0, 1.0)[:, None]
        self.motor[idx] = p.w_hover * self._u(k, 0.5, 1.2, 4)
        self.prev_act[idx] = 0.0
        self.integ[idx] = 0.0

    # ---- 物理 ----
    def _physics(self, w_cmd):
        p, dt = self.p, self.sim_dt
        for _ in range(self.substeps):
            self.motor += (w_cmd - self.motor) * (dt / self.tau[:, None])
            f = p.kf * self.gain * self.motor ** 2
            wrench = f @ self.alloc.T                              # [总推力, τx, τy, τz]
            Jw = self.J * self.omega
            self.omega += dt * (wrench[:, 1:] - torch.linalg.cross(self.omega, Jw)) / self.J
            dq = quat_mul(self.quat, torch.cat([torch.zeros_like(self.t)[:, None], self.omega], -1))
            self.quat += 0.5 * dt * dq
            self.quat /= self.quat.norm(dim=-1, keepdim=True)
            zb = quat_to_rot(self.quat)[:, :, 2]
            force = zb * wrench[:, :1] - p.drag * self.vel + self.wind
            self.vel += dt * force / self.mass[:, None]
            self.vel[:, 2] -= dt * p.g
            self.pos += dt * self.vel

    def obs(self, noise: bool = True):
        p_ref, v_ref, a_ref = self.reference()
        e_p, e_v, omega = p_ref - self.pos, v_ref - self.vel, self.omega
        if noise and self.randomize:
            e_p = e_p + 0.005 * self._n(self.n, 3)
            e_v = e_v + 0.05 * self._n(self.n, 3)
            omega = omega + 0.03 * self._n(self.n, 3)
        R = quat_to_rot(self.quat).reshape(self.n, 9)
        return torch.cat([(e_p * S_POS).clamp(-2, 2), (e_v * S_VEL).clamp(-2, 2), (a_ref * S_ACC).clamp(-2, 2), R,
                          (omega * S_OMEGA).clamp(-3, 3), self.prev_act, self.integ * S_INT], -1)

    def step(self, action):
        """返回 (观测, 奖励, 终止, 截断, 重置前的观测)。终止/截断的环境已自动重置。"""
        p = self.p
        a = action.clamp(-1.0, 1.0)
        w_cmd = (p.w_hover * (1.0 + ACT_SCALE * a)).clamp(0.0, p.w_max)
        self._physics(w_cmd)
        self.t += POLICY_DT
        self.steps += 1

        p_ref, v_ref, _ = self.reference()
        e_p, e_v = p_ref - self.pos, v_ref - self.vel
        dist = e_p.norm(dim=-1)
        near = (dist < INT_RADIUS).float()[:, None]                # 只在目标附近积分,避免远距离时积分饱和
        self.integ = (self.integ + near * e_p * POLICY_DT).clamp(-INT_CLIP, INT_CLIP)

        R = quat_to_rot(self.quat)
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0]).abs()
        # 奖励:离目标越近越高(一个宽的项负责"飞过去",一个窄的项负责"停得准");
        # 惩罚项合计封顶 0.5,保证活着的奖励永远不低于 0,策略不会"故意坠毁"来止损。
        r_pos = 0.6 * torch.exp(-dist / 0.5) + 0.4 * torch.exp(-dist / 0.05)
        pen = (0.05 * e_v.norm(dim=-1) + 0.01 * self.omega.norm(dim=-1) + 0.1 * yaw
               + 0.03 * (a - self.prev_act).norm(dim=-1) + 0.05 * (1.0 - R[:, 2, 2]))
        reward = 0.5 + r_pos - pen.clamp(max=0.5)
        self.prev_act = a

        terminated = dist > 2.0
        truncated = (self.steps >= self.max_steps) & ~terminated
        final_obs = self.obs()
        self.reset_idx(torch.nonzero(terminated | truncated).squeeze(-1))
        obs = self.obs()
        return obs, reward, terminated, truncated, final_obs, dist


assert OBS_DIM == 28
