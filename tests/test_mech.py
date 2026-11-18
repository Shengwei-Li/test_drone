"""机架模型:几何与惯量公式的基本检查,以及换成新机架参数后控制器仍然可用。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mech.model import (ARM, Part, balanced, board_outline, design_params, mass_properties,  # noqa: E402
                        polygon_moments)
from run_demo import T_CALIB, T_HOVER, simulate  # noqa: E402


def test_polygon_moments_rectangle():
    """偏心放置的 4×2 矩形:面积、形心、绕形心的二次矩与公式一致。"""
    a, c, (ixx, iyy, ixy) = polygon_moments([(1, 1), (5, 1), (5, 3), (1, 3)])
    assert np.isclose(a, 8) and np.allclose(c, (3, 2))
    assert np.isclose(ixx, 4 * 2**3 / 12) and np.isclose(iyy, 2 * 4**3 / 12) and abs(ixy) < 1e-9


def test_plate_matches_box():
    """矩形薄板按"多边形"算和按"长方体"算,惯量应相同。"""
    box = Part("b", 3.0, "box", (4, 2, 0.5))
    plate = Part("p", 3.0, "plate", (0.5,), outline=[(-2, -1), (2, -1), (2, 1), (-2, 1)])
    assert np.allclose(box.inertia(), plate.inertia())


def test_parallel_axis():
    """两个质点状的小方块相距 2d:绕中点的惯量应为 2·m·d²。"""
    parts = [Part("a", 1.0, "box", (1e-6,) * 3, (10, 0, 0)), Part("b", 1.0, "box", (1e-6,) * 3, (-10, 0, 0))]
    m, cg, inertia = mass_properties(parts)
    assert np.isclose(m, 2) and np.allclose(cg, 0)
    assert np.allclose(np.diag(inertia), [0, 200, 200], atol=1e-6)


def test_design_is_balanced_and_plausible():
    batt_x, parts = balanced()
    m, cg, inertia = mass_properties(parts)
    assert abs(cg[0]) < 1e-6 and abs(cg[1]) < 0.5            # 前后配平,左右偏差 < 0.5 mm
    assert 25 < m < 40                                        # g
    p = design_params()
    assert p.thrust_to_weight > 1.7      # 加了外接模块和长脚架后是 1.78;再往下就该换电机或减重了
    assert np.isclose(np.hypot(*p.motor_xy[0]), ARM * 1e-3)
    outline = np.array(board_outline())
    assert np.all(np.isfinite(outline))
    assert np.allclose(outline, -np.roll(outline, len(outline) // 2, axis=0), atol=1e-3)       # 中心对称


def test_controllers_work_on_design_airframe():
    """把机体参数整个换成新机架的:INDI 不重调,8 字跟踪仍在 2 cm 以内;
    强化学习策略没有重训,也应保持在 8 cm 以内。"""
    p = design_params()
    for ctrl, limit in (("indi", 0.02), ("rl", 0.08)):
        log, _ = simulate(params=p, controller=ctrl)
        m = log["t"] > T_CALIB + T_HOVER + 2.0
        err = np.linalg.norm(log["pos"] - log["ref"], axis=1)[m]
        assert np.sqrt(np.mean(err ** 2)) < limit
