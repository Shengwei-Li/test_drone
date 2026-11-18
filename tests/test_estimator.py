import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_demo import estimation_errors, rms, simulate  # noqa: E402
from sim.dynamics import (M, P, Q, V, W, initial_state, quat_conj,  # noqa: E402
                          quat_mul)
from sim.estimator import ESKF, NE, quat_from_rotvec  # noqa: E402
from sim.params import QuadParams  # noqa: E402
from sim.sensors import SensorConfig, SensorSuite  # noqa: E402


def random_estimator(rng):
    """一个处于一般姿态、一般运动状态的估计器。"""
    est = ESKF(QuadParams())
    est.p = np.array([0.3, -0.2, 0.9])
    est.v = rng.normal(0, 0.5, 3)
    est.q = quat_from_rotvec(rng.normal(0, 0.3, 3))
    est.bg = rng.normal(0, 0.02, 3)
    est.ba = rng.normal(0, 0.1, 3)
    est.gyro = rng.normal(0, 0.5, 3)
    return est


def boxminus(b: ESKF, a: ESKF) -> np.ndarray:
    """b 相对 a 的误差状态。"""
    dq = quat_mul(quat_conj(a.q), b.q)
    return np.concatenate([b.p - a.p, b.v - a.v, 2 * dq[1:] * np.sign(dq[0]),
                           b.bg - a.bg, b.ba - a.ba])


def test_measurement_jacobians_match_finite_differences():
    rng = np.random.default_rng(1)
    eps = 1e-6
    for _ in range(5):
        est = random_estimator(rng)
        for h in ("h_tof", "h_flow"):
            pred, H = getattr(est, h)()
            num = np.zeros_like(H)
            for k in range(NE):
                pert = copy.deepcopy(est)
                d = np.zeros(NE)
                d[k] = eps
                pert.inject(d)
                num[:, k] = (getattr(pert, h)()[0] - pred) / eps
            assert np.allclose(H, num, atol=1e-4), h


def test_predict_jacobian_matches_finite_differences():
    rng = np.random.default_rng(2)
    dt, eps = 0.002, 1e-5
    for _ in range(5):
        a = random_estimator(rng)
        gyro, acc = rng.normal(0, 1, 3), np.array([0.5, -0.3, 9.5])
        for k in range(NE):
            a0, b0 = copy.deepcopy(a), copy.deepcopy(a)
            d = np.zeros(NE)
            d[k] = eps
            b0.inject(d)
            a0.predict(gyro, acc, dt)
            b0.predict(gyro, acc, dt)
            assert np.allclose(boxminus(b0, a0), a0.last_Phi @ d, atol=1e-3 * eps), k


def test_ideal_sensors_match_ground_truth():
    """传感器无噪声时,悬停时的读数应符合物理直觉。"""
    p = QuadParams()
    s = SensorSuite(p, SensorConfig.ideal())
    x = initial_state(p, pos=(0, 0, 1.0))
    gyro, acc = s.imu(x, 0.002)
    assert np.allclose(gyro, 0) and np.allclose(acc, [0, 0, p.g])
    assert np.isclose(s.tof(x), 1.0)
    x[V] = [0.5, 0, 0]
    assert np.allclose(s.flow(x), [0.5, 0])
    x[P][2] = 0.01
    assert s.tof(x) is None and s.flow(x) is None


def test_noise_free_estimate_tracks_truth():
    log, _ = simulate(estimate=True, sensor_cfg=SensorConfig.ideal(), t_end=8.0)
    e = estimation_errors(log)
    assert np.abs(e["pos"]).max() < 0.02
    assert np.abs(e["vel"]).max() < 0.02
    assert e["tilt_deg"].max() < 0.2


def test_imu_reads_gravity_on_ground_and_calibration_recovers_bias():
    p = QuadParams()
    s = SensorSuite(p, rng=np.random.default_rng(3))
    x = initial_state(p, hover=False)                 # 静止在地面,电机不转
    samples = np.array([s.imu(x, 0.002) for _ in range(500)])
    assert np.allclose(samples[:, 1].mean(axis=0) - s.acc_bias, [0, 0, p.g], atol=0.1)
    est = ESKF(p)
    assert est.calibrate_gyro(samples[:, 0])
    assert np.allclose(est.bg, s.gyro_bias, atol=3e-3)
    # 机体在动时应拒绝校准
    moving = samples[:, 0] + np.linspace(0, 1, 500)[:, None]
    assert not ESKF(p).calibrate_gyro(moving)


def test_gyro_calibration_reduces_yaw_drift():
    for seed in range(3):
        yaw = {}
        for calib in (False, True):
            log, _ = simulate(estimate=True, seed=seed, calibrate=calib)
            yaw[calib] = abs(estimation_errors(log)["yaw_deg"][-1])
        assert yaw[True] < 3.0, (seed, yaw)
    # 种子 2 的 z 轴零偏较大,校准前后差别明显
    assert yaw[True] < 0.5 * yaw[False]


def test_closed_loop_with_noisy_sensors():
    """带噪声闭环飞完整条轨迹:不发散,估计和跟踪误差都在合理范围内。"""
    for seed in range(3):
        log, _ = simulate(estimate=True, seed=seed)
        e = estimation_errors(log)
        late = log["t"] > 5.0
        assert rms(e["pos"][late, 2]) < 0.02, seed
        assert rms(np.linalg.norm(e["vel"][late], axis=1)) < 0.15, seed
        assert rms(e["tilt_deg"][len(e["tilt_deg"]) // 3:]) < 2.0, seed
        track = np.linalg.norm(log["pos"] - log["ref"], axis=1)[late]
        assert rms(track) < 0.30, seed
        assert log["tilt"].max() < 45.0, seed
