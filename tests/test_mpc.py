"""MPC:求解器对照通用优化器,闭环检查约束与跟踪。"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_demo import T_CALIB, T_HOVER, simulate  # noqa: E402
from sim.mpc import AdmmMpc  # noqa: E402

H = 0.05
A = np.array([[1.0, H, 0.5 * H * H], [0.0, 1.0, H], [0.0, 0.0, 1.0]])
B = np.array([[H ** 3 / 6.0], [0.5 * H * H], [H]])
Q = np.diag([100.0, 0.0, 0.05])
R = np.array([[0.005]])
N = 12


def make(v_max=np.inf, u_max=np.inf):
    x_max = np.array([np.inf, v_max, np.inf])
    return AdmmMpc(A, B, Q, R, N, [-u_max], [u_max], -x_max, x_max, rho_u=0.005, rho_x=10.0)


def rollout(x0, u):
    xs = [x0]
    for k in range(N):
        xs.append(A @ xs[-1] + B[:, 0] * u[k])
    return np.array(xs)


def cost(mpc, x0, x_ref, u):
    xs = rollout(x0, u)
    e = xs - x_ref
    return (0.5 * np.einsum("ki,ij,kj->", e[:N], Q, e[:N]) + 0.5 * R[0, 0] * u @ u
            + 0.5 * e[N] @ mpc.P @ e[N])


def test_unconstrained_is_lqr():
    """没有约束时,一次迭代就应得到 LQR 的解 u = −K x(参考为 0)。"""
    mpc = make()
    x0 = np.array([0.3, -0.2, 0.1])
    u0, _ = mpc.solve(x0, np.zeros((N + 1, 3)), iters=1)
    assert np.allclose(u0, -mpc.K_lqr @ x0, atol=1e-6)


def test_matches_generic_solver():
    """有输入和速度约束时,ADMM 的解应与通用优化器 (SLSQP) 一致。"""
    v_max, u_max = 0.8, 20.0
    mpc = make(v_max, u_max)
    x0 = np.zeros(3)
    x_ref = np.zeros((N + 1, 3))
    x_ref[:, 0] = 1.0                                   # 1 m 阶跃
    _, xs = mpc.solve(x0, x_ref, iters=3000)
    u_admm = mpc.z[:, 0]

    # 代价是 u 的二次函数、速度是 u 的线性函数:用单位向量把它们的矩阵"量"出来,给优化器精确梯度
    eye = np.eye(N)
    c0 = cost(mpc, x0, x_ref, np.zeros(N))
    f = np.array([(cost(mpc, x0, x_ref, e) - cost(mpc, x0, x_ref, -e)) / 2 for e in eye])
    Hm = np.array([[cost(mpc, x0, x_ref, a + b) - cost(mpc, x0, x_ref, a)
                    - cost(mpc, x0, x_ref, b) + c0 for b in eye] for a in eye])
    Sv = np.array([rollout(x0, e)[1:N, 1] for e in eye]).T      # x0 = 0,所以没有常数项
    cons = [{"type": "ineq", "fun": lambda u: v_max - Sv @ u, "jac": lambda u: -Sv},
            {"type": "ineq", "fun": lambda u: v_max + Sv @ u, "jac": lambda u: Sv}]
    res = minimize(lambda u: 0.5 * u @ Hm @ u + f @ u + c0, np.zeros(N), method="SLSQP",
                   jac=lambda u: Hm @ u + f, bounds=[(-u_max, u_max)] * N, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    assert res.success, res.message
    assert np.max(np.abs(xs[1:N, 1])) <= v_max + 5e-3        # 约束确实起作用且被满足
    assert np.max(np.abs(rollout(x0, res.x)[1:N, 1])) > v_max - 1e-3
    assert abs(cost(mpc, x0, x_ref, u_admm) - res.fun) < 1e-3 * res.fun
    assert np.allclose(u_admm, res.x, atol=0.2)


def test_mpc_tracks_eight_without_feedforward():
    """MPC 只拿到未来的参考位置,8 字跟踪误差也应在 2 cm 以内。"""
    log, _ = simulate(controller="mpc")
    m = log["t"] > T_CALIB + T_HOVER + 2.0
    err = np.linalg.norm(log["pos"] - log["ref"], axis=1)[m]
    assert np.sqrt(np.mean(err ** 2)) < 0.02


def test_mpc_waypoints_beats_indi_and_respects_speed_limit():
    """航点跳变:MPC 能提前动作,误差应明显小于 INDI;速度不应明显超过 2 m/s 的限制。"""
    rms = {}
    for c in ("mpc", "indi"):
        log, _ = simulate(controller=c, trajectory="waypoints", disturbed=True)
        m = log["t"] > T_CALIB + T_HOVER + 2.0
        err = np.linalg.norm(log["pos"] - log["ref"], axis=1)[m]
        rms[c] = np.sqrt(np.mean(err ** 2))
        if c == "mpc":
            assert np.abs(log["vel"][m]).max() < 2.2
    assert rms["mpc"] < 0.5 * rms["indi"]
