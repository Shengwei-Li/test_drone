"""强化学习:批量环境与原仿真的一致性,以及训练好的策略(如果有)的基本表现。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.dynamics import M, P, Q, V, W, initial_state, step  # noqa: E402
from sim.params import QuadParams  # noqa: E402
from sim.rl_policy import ACT_SCALE, DEFAULT_POLICY, POLICY_DT, make_obs  # noqa: E402

torch = pytest.importorskip("torch")
from rl.env import QuadEnv  # noqa: E402


def test_batched_env_matches_reference_sim():
    """同样的初始状态和动作序列,批量环境与原仿真 (RK4) 飞 0.5 s 后应重合。

    这里把批量环境的步长缩到 0.1 ms 来核对方程本身;训练用的 1 ms 欧拉积分
    误差约为它的 10 倍(位置 1 cm、速度 8 cm/s),随步长线性缩小。
    """
    p = QuadParams()
    env = QuadEnv(3, "cpu", params=p, randomize=False, seed=1, sim_dt=1e-4)
    env.amp[:] = 0.0                                   # 参考点固定在原点
    rng = np.random.default_rng(0)
    actions = rng.uniform(-0.3, 0.3, size=(50, 4))
    xs = []
    for i in range(env.n):
        x = initial_state(p)
        x[P], x[V] = env.pos[i].numpy(), env.vel[i].numpy()
        x[Q], x[W], x[M] = env.quat[i].numpy(), env.omega[i].numpy(), env.motor[i].numpy()
        xs.append(x.astype(float))
    obs0 = env.obs(noise=False)[0].numpy()
    assert np.allclose(obs0, make_obs(xs[0], np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(4), np.zeros(3)), atol=1e-5)
    for a in actions:
        env._physics(torch.tensor(p.w_hover * (1 + ACT_SCALE * a), dtype=torch.float32).expand(env.n, 4))
        for i in range(env.n):
            for _ in range(round(POLICY_DT / 0.001)):
                xs[i] = step(xs[i], p.w_hover * (1 + ACT_SCALE * a), p, 0.001)
    for i in range(env.n):
        assert np.linalg.norm(env.pos[i].numpy() - xs[i][P]) < 0.003
        assert np.linalg.norm(env.vel[i].numpy() - xs[i][V]) < 0.02
        assert abs(abs(env.quat[i].numpy() @ xs[i][Q]) - 1.0) < 1e-3


def test_short_training_improves():
    """跑 40 轮 PPO(小规模),平均奖励应明显上升 —— 检查训练流程本身没坏。

    开头十几轮奖励会先降(策略开始乱试),所以不能只看前几轮。
    """
    from rl.train import train
    tmp = Path(__file__).parent / "_tmp_policy.npz"
    _, hist = train(iters=40, n_envs=1024, out=tmp, log=False)
    tmp.unlink(missing_ok=True)
    assert hist[-1]["reward"] > hist[0]["reward"] + 0.1


@pytest.mark.skipif(not DEFAULT_POLICY.exists(), reason="还没有训练好的策略")
def test_trained_policy_flies_in_reference_sim():
    """训练好的策略放回原仿真(RK4、有地面、从地面起飞):能悬停、能跟 8 字、抗扰动。"""
    from run_demo import T_CALIB, T_HOVER, simulate
    for disturbed in (False, True):
        log, _ = simulate(controller="rl", disturbed=disturbed)
        m = log["t"] > T_CALIB + T_HOVER + 2.0
        err = np.linalg.norm(log["pos"] - log["ref"], axis=1)[m]
        assert np.sqrt(np.mean(err ** 2)) < 0.08


@pytest.mark.skipif(not DEFAULT_POLICY.exists(), reason="还没有训练好的策略")
def test_trained_policy_recovers_from_throw():
    """大角度翻滚的初始状态:策略的恢复成功率应高于 INDI。"""
    from recovery import run
    res = run(n=30, controllers=["rl", "indi"])
    assert res["rl"][0] >= 0.6 and res["rl"][0] > res["indi"][0]
