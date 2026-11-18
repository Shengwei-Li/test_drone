"""对比几种控制器的跟踪误差。

用法:  python compare_controllers.py          控制器用状态真值
       python compare_controllers.py --est    控制器用 ESKF 估计值(对 3 个随机种子取平均)

两条轨迹:
  8 字 —— 光滑轨迹,PID/几何/INDI 能拿到解析的速度、加速度、jerk 前馈;MPC 只看未来的位置。
  航点 —— 参考位置每 2.5 s 跳 1 m,没有任何前馈。
两种场景:
  标称 —— 真实机体与控制器的模型完全一致,没有外力。
  扰动 —— 机体重 10%、1 号电机弱 10%,飞到一半起侧风(约 11% 机重)。
"""
import argparse

import numpy as np

from run_demo import T_CALIB, T_HOVER, simulate
from sim.controller import CONTROLLERS

NAMES = {"pid": "串级 PID", "geometric": "几何控制", "indi": "INDI", "mpc": "MPC+INDI", "rl": "强化学习"}
TRAJ_NAMES = {"eight": "8 字", "waypoints": "航点"}


def tracking(controller: str, disturbed: bool, estimate: bool, seeds, trajectory="eight", params=None):
    """返回 [跟踪误差 RMS cm, 最大值 cm, 最大速度 m/s(, 相对"估计位置"的 RMS cm)]。"""
    rms_true, max_true, v_max, rms_est = [], [], [], []
    for seed in seeds:
        log, _ = simulate(params=params, controller=controller, disturbed=disturbed,
                          estimate=estimate, seed=seed, trajectory=trajectory)
        m = log["t"] > T_CALIB + T_HOVER + 2.0
        err = np.linalg.norm(log["pos"] - log["ref"], axis=1)[m]
        rms_true.append(np.sqrt(np.mean(err ** 2)))
        max_true.append(err.max())
        v_max.append(np.linalg.norm(log["vel"], axis=1)[m].max())
        if estimate:
            e = np.linalg.norm(log["est_pos"] - log["ref"], axis=1)[m]
            rms_est.append(np.sqrt(np.mean(e ** 2)))
    out = [np.mean(rms_true) * 100, np.max(max_true) * 100, np.max(v_max)]
    if estimate:
        out.append(np.mean(rms_est) * 100)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", action="store_true")
    ap.add_argument("--design", action="store_true", help="用 mech/model.py 算出的机架参数")
    args = ap.parse_args()
    seeds = range(3) if args.est else [0]
    params = None
    if args.design:
        from mech.model import design_params
        params = design_params()
        print(f"机架参数:质量 {params.mass * 1000:.1f} g,推重比 {params.thrust_to_weight:.2f}")
    print("状态来源:", "ESKF 估计值(3 个种子平均)" if args.est else "真值")
    head = f"{'轨迹':<6}{'控制器':<10}{'场景':<6}{'RMS cm':>9}{'最大 cm':>9}{'最大速度':>9}"
    if args.est:
        head += f"{'相对估计位置 RMS cm':>18}"
    print(head)
    from sim.rl_policy import DEFAULT_POLICY
    names = [c for c in CONTROLLERS if c != "rl" or DEFAULT_POLICY.exists()]
    for traj in TRAJ_NAMES:
        for name in names:
            for disturbed in (False, True):
                r = tracking(name, disturbed, args.est, seeds, traj, params)
                line = (f"{TRAJ_NAMES[traj]:<6}{NAMES[name]:<10}{'扰动' if disturbed else '标称':<6}"
                        f"{r[0]:>9.1f}{r[1]:>9.1f}{r[2]:>9.2f}")
                if args.est:
                    line += f"{r[3]:>18.1f}"
                print(line)


if __name__ == "__main__":
    main()
