""""抛飞恢复"测试:从随机的大角度姿态和翻滚角速度开始,看各控制器能否稳回悬停点。

用法:  python recovery.py [--n 100]

每次试验:机体在 3 m 高度,姿态随机倾斜 90°–180°(侧翻到完全倒置),角速度最高约 10 rad/s,
电机从怠速起转,目标是原地悬停。3 秒内没有掉落超过 2.5 m、最后 0.5 秒离目标不超过 20 cm,算成功。
"""
import argparse

import numpy as np

from sim.controller import CONTROLLERS, Reference
from sim.dynamics import M, P, Q, V, W, initial_state, step
from sim.params import QuadParams

NAMES = {"pid": "串级 PID", "geometric": "几何控制", "indi": "INDI", "mpc": "MPC+INDI", "rl": "强化学习"}
TARGET = np.array([0.0, 0.0, 3.0])


def trial(controller: str, rng, params: QuadParams, t_end: float = 3.0):
    x = initial_state(params, pos=TARGET, hover=False)
    axis = rng.normal(size=3)
    axis[2] *= 0.2                                     # 以横滚/俯仰方向的翻转为主
    axis /= np.linalg.norm(axis)
    ang = rng.uniform(np.pi / 2, np.pi)
    x[Q] = [np.cos(ang / 2), *(axis * np.sin(ang / 2))]
    x[W] = rng.uniform(-6.0, 6.0, size=3)
    x[V] = rng.uniform(-1.0, 1.0, size=3)
    x[M] = params.w_hover * 0.5
    ctrl = CONTROLLERS[controller](params, 0.002)
    ref = Reference(pos=TARGET.copy())
    lowest, errs = 0.0, []
    w_cmd = np.zeros(4)
    for i in range(int(t_end / 0.001)):
        if i % 2 == 0:
            ref.t = i * 0.001
            w_cmd = ctrl.update(x, ref)
        x = step(x, w_cmd, params, 0.001)
        if not np.all(np.isfinite(x)):
            return False, np.inf
        lowest = min(lowest, x[P][2] - TARGET[2])
        if i * 0.001 > t_end - 0.5:
            errs.append(np.linalg.norm(x[P] - TARGET))
    return bool(lowest > -2.5 and max(errs) < 0.2), -lowest


def run(n: int = 100, seed: int = 0, controllers=None):
    params = QuadParams()
    out = {}
    for name in controllers or CONTROLLERS:
        rng = np.random.default_rng(seed)              # 每个控制器面对同一批初始状态
        res = [trial(name, rng, params) for _ in range(n)]
        ok = [r for r in res if r[0]]
        out[name] = (len(ok) / n, float(np.mean([r[1] for r in ok])) if ok else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    print(f"{'控制器':<10}{'成功率':>8}{'成功时平均掉高 m':>14}")
    for name, (rate, drop) in run(args.n).items():
        print(f"{NAMES[name]:<10}{rate * 100:>7.0f}%{drop:>14.2f}")


if __name__ == "__main__":
    main()
