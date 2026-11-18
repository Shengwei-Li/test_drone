"""跑一遍演示仿真,生成可在浏览器里打开的 3D 回放 out/replay.html。

用法:  python visualize.py           控制器用状态真值
       python visualize.py --est     控制器用状态估计,橙线为估计出的轨迹
页面从 CDN 加载 three.js,打开时需要联网。
"""
import argparse
import json
from pathlib import Path

import numpy as np

from run_demo import SIM_DT, simulate
from sim.controller import CONTROLLERS

FPS = 50
SCALE = 2.0     # 机体显示放大倍数(真机只有 9 cm,按原尺寸太小)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", action="store_true", help="控制器使用状态估计,并画出估计轨迹")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctrl", choices=list(CONTROLLERS), default="pid")
    ap.add_argument("--disturb", action="store_true", help="扰动场景:模型失配 + 侧风")
    ap.add_argument("--traj", choices=["eight", "waypoints"], default="eight", help="参考轨迹")
    ap.add_argument("--design", action="store_true", help="用 mech/model.py 的机架参数和外形")
    args = ap.parse_args()
    design = None
    if args.design:
        from mech.model import balanced, design_params, mass_properties, viz_model
        design = design_params()
    log, params = simulate(params=design, estimate=args.est, seed=args.seed,
                           controller=args.ctrl, disturbed=args.disturb,
                           trajectory=args.traj)
    stride = int(round(1.0 / FPS / SIM_DT))
    sl = slice(None, None, stride)
    data = {
        "dt": SIM_DT * stride,
        "scale": SCALE,
        "motor_xy": params.motor_xy.tolist(),
        "pos": np.round(log["pos"][sl], 4).tolist(),
        "quat": np.round(log["quat"][sl], 5).tolist(),
        "ref": np.round(log["ref"][sl], 4).tolist(),
        "thrust": np.round((log["motors"][sl] / params.w_max) ** 2, 3).tolist(),
    }
    if args.design:
        parts = balanced()[1]
        data["model"] = viz_model(parts, mass_properties(parts)[1])
    if args.est:
        data["est"] = np.round(log["est_pos"][sl], 4).tolist()
    root = Path(__file__).parent
    html = (root / "viz" / "replay_template.html").read_text(encoding="utf-8")
    out = root / "out"
    out.mkdir(exist_ok=True)
    (out / "replay.html").write_text(
        html.replace("/*DATA*/", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
    print(f"回放已生成: {out / 'replay.html'}")


if __name__ == "__main__":
    main()
