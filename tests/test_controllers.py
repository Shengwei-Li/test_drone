import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_controllers import tracking  # noqa: E402
from run_demo import simulate  # noqa: E402


def test_nominal_tracking_improves_with_feedforward():
    """模型准确、无扰动:带前馈的几何控制和 INDI 应明显好于串级 PID。"""
    pid = tracking("pid", False, False, [0])[0]
    geo = tracking("geometric", False, False, [0])[0]
    indi = tracking("indi", False, False, [0])[0]
    assert 7 < pid < 11
    assert geo < 5
    assert indi < 2


def test_indi_rejects_model_mismatch_and_wind():
    """机体重 10%、一个电机弱 10%、有侧风:INDI 仍应保持厘米级跟踪。"""
    pid = tracking("pid", True, False, [0])[0]
    geo = tracking("geometric", True, False, [0])[0]
    indi, indi_max = tracking("indi", True, False, [0])[:2]
    assert indi < 5 and indi_max < 12
    assert indi < pid / 3
    assert geo > pid          # 几何控制没有积分项,扛不住恒定扰动


def test_indi_stable_with_noisy_estimate():
    """INDI 要对陀螺求导,噪声下不能发散或让电机抖到饱和。"""
    for seed in range(2):
        log, params = simulate(controller="indi", disturbed=True, estimate=True, seed=seed)
        late = log["t"] > 6.0
        rel = np.linalg.norm(log["est_pos"] - log["ref"], axis=1)[late]
        assert np.sqrt(np.mean(rel ** 2)) < 0.08, seed
        assert log["tilt"].max() < 45.0, seed
        assert np.mean(log["motors"][late] > 0.98 * params.w_max) < 0.01, seed
