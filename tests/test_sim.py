import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.controller import CascadedController, Reference  # noqa: E402
from sim.dynamics import (M, P, Q, V, W, initial_state, quat_to_rot,  # noqa: E402
                          rot_to_quat, step)
from sim.params import QuadParams  # noqa: E402

DT = 0.001


def run(x, w_cmd, params, seconds):
    for _ in range(int(round(seconds / DT))):
        x = step(x, w_cmd, params, DT)
    return x


def test_free_fall():
    p = replace(QuadParams(), drag=0.0)
    x = run(initial_state(p, hover=False), np.zeros(4), p, 1.0)
    assert np.isclose(x[P][2], -0.5 * p.g, atol=1e-6)
    assert np.isclose(x[V][2], -p.g, atol=1e-6)


def test_hover_equilibrium():
    p = QuadParams()
    x = run(initial_state(p, pos=(0, 0, 1)), np.full(4, p.w_hover), p, 2.0)
    assert np.allclose(x[P], [0, 0, 1], atol=1e-6)
    assert np.allclose(x[W], 0, atol=1e-9)


def test_ground_holds_until_thrust_exceeds_weight():
    p = QuadParams()
    x = initial_state(p, hover=False)
    for _ in range(500):                              # 电机不转:留在地面
        x = step(x, np.zeros(4), p, DT, ground=True)
    assert x[P][2] == 0.0 and not np.any(x[V])
    for _ in range(500):                              # 1.2 倍悬停转速:离地
        x = step(x, np.full(4, 1.2 * p.w_hover), p, DT, ground=True)
    assert x[P][2] > 0.05 and x[V][2] > 0


def test_torque_signs():
    """加大某一侧电机的转速,机体应朝预期方向转。"""
    p = QuadParams()
    xy, d = p.motor_xy, p.motor_dir

    def rates(mask):
        x = initial_state(p)
        x[M] = p.w_hover * np.where(mask, 1.05, 1.0)   # 直接设转速,绕开电机滞后
        return step(x, x[M], p, DT)[W]

    assert rates(xy[:, 1] > 0)[0] > 0      # 左侧加力 → 正滚转
    assert rates(xy[:, 0] > 0)[1] < 0      # 前侧加力 → 抬头(负俯仰角速度)
    assert rates(d > 0)[2] > 0             # 正反扭矩组加力 → 正偏航


def test_quaternion_roundtrip_and_norm():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q2 = rot_to_quat(quat_to_rot(q))
        assert np.allclose(q, q2, atol=1e-9) or np.allclose(q, -q2, atol=1e-9)
    p = QuadParams()
    x = initial_state(p)
    x[W] = [3.0, -2.0, 1.0]
    x = run(x, np.full(4, p.w_hover), p, 1.0)
    assert np.isclose(np.linalg.norm(x[Q]), 1.0, atol=1e-12)


def test_closed_loop_reaches_setpoint():
    p = QuadParams()
    ctrl = CascadedController(p, dt=2 * DT)
    ref = Reference(pos=np.array([0.5, -0.3, 1.0]), yaw=0.5)
    x = initial_state(p)
    w_cmd = np.full(4, p.w_hover)
    for i in range(6000):
        if i % 2 == 0:
            w_cmd = ctrl.update(x, ref)
        x = step(x, w_cmd, p, DT)
    assert np.linalg.norm(x[P] - ref.pos) < 0.02
    assert np.linalg.norm(x[V]) < 0.02
    R = quat_to_rot(x[Q])
    assert np.isclose(np.arctan2(R[1, 0], R[0, 0]), 0.5, atol=0.02)
