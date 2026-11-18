"""固件核心库(C)与 Python 版本的对照。需要 C 编译器;没有时跳过。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_demo import simulate  # noqa: E402
from sim.estimator import ESKF  # noqa: E402
from sim.params import QuadParams  # noqa: E402

try:
    from firmware.build import ensure
    ensure("f32"), ensure("f64")
    from firmware.sil import CEskf
except Exception as exc:                                    # noqa: BLE001
    pytest.skip(f"无法编译固件库: {exc}", allow_module_level=True)


def _feed(e, seed=0, n=1500):
    rng = np.random.default_rng(seed)
    e.calibrate_gyro(rng.normal(0.01, 0.02, size=(200, 3)))
    for i in range(n):
        e.predict(rng.normal(0, 0.3, 3), np.array([0, 0, 9.81]) + rng.normal(0, 0.5, 3), 0.002)
        if i % 5 == 0:
            e.update_flow(rng.normal(0, 0.2, 2))
        if i % 10 == 0:
            e.update_tof(0.5 + 0.01 * np.sin(i / 50))
    return e


def test_eskf_same_inputs_same_outputs():
    """同一串随机的 IMU/测距/光流数据喂给两边:双精度应一致到 1e-9;
    单精度状态偏差 < 2e-3,协方差对角偏差 < 1% 且保持对称正定。"""
    p = QuadParams()
    py = _feed(ESKF(p, pos0=(0, 0, 0.5)))
    c64 = _feed(CEskf(p, pos0=(0, 0, 0.5), precision="f64"))
    for name in ("p", "v", "q", "bg", "ba"):
        assert np.allclose(getattr(c64, name), getattr(py, name), atol=1e-9), name
    assert np.allclose(c64.Pm, py.Pm, rtol=1e-8, atol=1e-14)
    c32 = _feed(CEskf(p, pos0=(0, 0, 0.5), precision="f32"))
    for name in ("p", "v", "q", "bg", "ba"):
        assert np.allclose(getattr(c32, name), getattr(py, name), atol=2e-3), name
    Pm = c32.Pm
    assert np.allclose(np.diag(Pm), np.diag(py.Pm), rtol=0.01)          # 单精度下协方差没有跑偏
    assert np.linalg.eigvalsh(0.5 * (Pm + Pm.T)).min() > 0               # 仍然正定
    assert np.abs(Pm - Pm.T).max() < 1e-6


@pytest.mark.parametrize("ctrl,estimate", [("indi", True), ("geometric", False), ("rl", True)])
def test_closed_loop_flight_matches_python(ctrl, estimate):
    """整段闭环飞行(扰动场景):C 双精度与 Python 的轨迹差 < 1 纳米,单精度 < 0.1 毫米。"""
    ref = simulate(controller=ctrl, estimate=estimate, disturbed=True)[0]
    for prec, tol in (("f64", 1e-9), ("f32", 1e-4)):
        log = simulate(controller=ctrl, estimate=estimate, disturbed=True, firmware=prec)[0]
        assert np.abs(log["pos"] - ref["pos"]).max() < tol, prec
