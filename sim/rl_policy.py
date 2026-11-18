"""强化学习策略的部署端:只用 numpy,不依赖 PyTorch。

训练(rl/train.py)把网络权重存成 .npz,这里读进来做前向计算。
网络是两层隐藏层的小型全连接网络,单片机上就是三次矩阵乘向量加 tanh。

观测(28 维),训练环境 rl/env.py 用的是同一套定义和缩放系数:
    位置误差 (3)      参考位置 − 当前位置,世界系
    速度误差 (3)      参考速度 − 当前速度,世界系
    参考加速度 (3)    前馈:让策略提前倾斜,而不是等误差出现
    旋转矩阵 (9)      机体系到世界系,按行展开
    角速度 (3)        机体系
    上一次的动作 (4)
    位置误差的积分 (3) 只在离目标 0.3 m 以内时累加,让策略能消除恒定扰动造成的稳态误差
动作(4 维,−1..1):电机转速指令 = 悬停转速 × (1 + 0.5 × 动作)。
"""
from pathlib import Path

import numpy as np

from .dynamics import P, Q, V, W, quat_to_rot
from .params import QuadParams

POLICY_DT = 0.01            # 策略 100 Hz
OBS_DIM = 28
S_POS, S_VEL, S_ACC, S_OMEGA, S_INT = 1.0, 0.5, 0.2, 0.2, 4.0   # 观测缩放
ACT_SCALE = 0.5
INT_RADIUS, INT_CLIP = 0.3, 0.5
DEFAULT_POLICY = Path(__file__).resolve().parents[1] / "rl" / "policy.npz"


def make_obs(x, ref_pos, ref_vel, ref_acc, prev_act, integ) -> np.ndarray:
    e_p, e_v = ref_pos - x[P], ref_vel - x[V]
    return np.concatenate([
        np.clip(e_p * S_POS, -2, 2), np.clip(e_v * S_VEL, -2, 2), np.clip(ref_acc * S_ACC, -2, 2),
        quat_to_rot(x[Q]).ravel(),
        np.clip(x[W] * S_OMEGA, -3, 3), prev_act, integ * S_INT])


class RlController:
    """把训练好的策略包装成与其他控制器相同的接口(update 返回电机转速指令)。"""

    def __init__(self, params: QuadParams, dt: float, path=None):
        self.p, self.dt = params, dt
        self.every = max(1, round(POLICY_DT / dt))
        data = np.load(path or DEFAULT_POLICY)
        self.layers = [(data[f"W{i}"], data[f"b{i}"]) for i in range(3)]
        self.reset()

    def reset(self):
        self._k = 0
        self._act = np.zeros(4)
        self._integ = np.zeros(3)
        self._w_cmd = np.full(4, self.p.w_hover)

    def policy(self, obs: np.ndarray) -> np.ndarray:
        h = obs
        for i, (Wm, b) in enumerate(self.layers):
            h = Wm @ h + b
            if i < 2:
                h = np.tanh(h)
        return np.clip(h, -1.0, 1.0)

    def update(self, x: np.ndarray, ref, acc=None) -> np.ndarray:
        if self._k % self.every == 0:
            e_p = ref.pos - x[P]
            if np.linalg.norm(e_p) < INT_RADIUS:
                self._integ = np.clip(self._integ + e_p * POLICY_DT, -INT_CLIP, INT_CLIP)
            self._act = self.policy(make_obs(x, ref.pos, ref.vel, ref.acc, self._act, self._integ))
            self._w_cmd = np.clip(self.p.w_hover * (1.0 + ACT_SCALE * self._act), 0.0, self.p.w_max)
        self._k += 1
        return self._w_cmd
