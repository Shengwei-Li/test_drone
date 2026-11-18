"""演示:从原点起飞到 1 m 悬停,然后跟踪 8 字轨迹。

用法:  python run_demo.py            控制器用状态真值(理想情况)
       python run_demo.py --est      控制器用 ESKF 的估计值(带传感器噪声)
装了 matplotlib 时会把曲线存到 out/demo.png。
"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from sim.controller import CONTROLLERS, Reference
from sim.dynamics import M, P, Q, V, initial_state, quat_to_rot, step
from sim.estimator import ESKF
from sim.params import QuadParams
from sim.sensors import SensorConfig, SensorSuite, specific_force

SIM_DT = 0.001      # 物理步长 1 kHz
CTRL_EVERY = 2      # IMU、姿态/角速度环 500 Hz,位置环 100 Hz
FLOW_EVERY = 10     # 光流 100 Hz
TOF_EVERY = 20      # ToF 50 Hz
T_CALIB = 1.0       # 起飞前静止在地面上的时间,用来校准陀螺零偏
T_HOVER = 3.0       # 起飞后先悬停 3 s
T_END = 16.0        # 总时长(含地面静止)
T_WIND = 8.0        # 扰动场景:从这一刻起有侧风
WIND_FORCE = np.array([0.03, 0.0, 0.0])   # N,约为机体重量的 11%


def reference(t: float) -> Reference:
    t = t - T_CALIB                     # 轨迹从起飞时刻开始计时
    if t < T_HOVER:
        return Reference(pos=np.array([0.0, 0.0, 1.0]))
    s, amp, om = t - T_HOVER, 0.5, 1.5
    return Reference(
        pos=np.array([amp * np.sin(om * s), 0.5 * amp * np.sin(2 * om * s), 1.0]),
        vel=np.array([amp * om * np.cos(om * s), amp * om * np.cos(2 * om * s), 0.0]),
        acc=np.array([-amp * om**2 * np.sin(om * s),
                      -2 * amp * om**2 * np.sin(2 * om * s), 0.0]),
        jerk=np.array([-amp * om**3 * np.cos(om * s),
                       -4 * amp * om**3 * np.cos(2 * om * s), 0.0]),
    )


WAYPOINTS = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
T_LEG = 2.5         # 每个航点停留的时间


def waypoints(t: float) -> Reference:
    """航点轨迹:参考位置每 T_LEG 秒跳到下一个角点,不给速度/加速度前馈。"""
    t = t - T_CALIB
    if t < T_HOVER:
        return Reference(pos=np.array([0.0, 0.0, 1.0]))
    return Reference(pos=WAYPOINTS[int((t - T_HOVER) / T_LEG) % len(WAYPOINTS)].copy())


TRAJECTORIES = {"eight": reference, "waypoints": waypoints}


def disturbed_plant(params: QuadParams) -> QuadParams:
    """"扰动场景"里的真实机体:比模型重 10%,1 号电机推力弱 10%。

    电机能力不变,所以推重比相应下降。
    """
    return replace(params, mass=params.mass * 1.1,
                   thrust_to_weight=params.thrust_to_weight / 1.1,
                   motor_gain=(0.9, 1.0, 1.0, 1.0))


def simulate(params: QuadParams | None = None, estimate: bool = False, seed: int = 0,
             sensor_cfg: SensorConfig | None = None, t_end: float = T_END,
             calibrate: bool = True, controller: str = "pid", disturbed: bool = False,
             trajectory: str = "eight", firmware: str | None = None):
    """先在地面静止 T_CALIB 秒(电机不转),然后起飞。

    estimate=False:控制器直接用状态真值。
    estimate=True:控制器只能看到 ESKF 根据带噪声传感器给出的估计值。
    calibrate=True:用地面静止这段时间的陀螺读数校准零偏。
    controller:"pid" / "geometric" / "indi"。
    disturbed=True:真实机体与控制器手里的模型不一致(见 disturbed_plant),
                   并且从 T_WIND 起有恒定侧风。控制器和估计器始终只知道 params。
    """
    params = params or QuadParams()
    plant = disturbed_plant(params) if disturbed else params
    dt_ctrl = SIM_DT * CTRL_EVERY
    if firmware:                       # 软件在环:估计器和控制器换成编译好的 C 代码("f32" 或 "f64")
        from firmware.sil import CEskf, make_controller
        ctrl = make_controller(controller, params, dt_ctrl, firmware)
    else:
        ctrl = CONTROLLERS[controller](params, dt=dt_ctrl)
    traj = TRAJECTORIES[trajectory]
    if hasattr(ctrl, "set_trajectory"):
        ctrl.set_trajectory(traj)
    x = initial_state(plant, hover=False)
    n_calib = int(T_CALIB / SIM_DT)
    calib_samples = []
    sensors = SensorSuite(plant, sensor_cfg, np.random.default_rng(seed))
    est = CEskf(params, pos0=x[P], precision=firmware) if firmware else ESKF(params, pos0=x[P])
    n = int(t_end / SIM_DT)
    keys = ["t", "pos", "vel", "quat", "ref", "tilt", "motors"]
    if estimate:
        keys += ["est_pos", "est_vel", "est_quat"]
    log = {k: [] for k in keys}
    w_cmd = np.zeros(4)
    for i in range(n):
        t = i * SIM_DT
        ref = traj(t)
        ref.t = t
        wind = WIND_FORCE if disturbed and t >= T_WIND else None
        if i < n_calib:
            if estimate and i % CTRL_EVERY == 0:
                calib_samples.append(sensors.imu(x, dt_ctrl)[0])
        elif i % CTRL_EVERY == 0:
            if estimate and calibrate and calib_samples:
                est.calibrate_gyro(np.array(calib_samples))
                calib_samples = []
            if estimate:
                gyro, acc = sensors.imu(x, dt_ctrl, wind)
                est.predict(gyro, acc, dt_ctrl)
                if i % FLOW_EVERY == 0 and (flow := sensors.flow(x)) is not None:
                    est.update_flow(flow)
                if i % TOF_EVERY == 0 and (d := sensors.tof(x)) is not None:
                    est.update_tof(d)
                w_cmd = ctrl.update(est.state_vector(), ref, acc=acc - est.ba)
            else:
                w_cmd = ctrl.update(x, ref, acc=specific_force(x, plant, wind))
        x = step(x, w_cmd, plant, SIM_DT, ground=True, ext_force=wind)
        log["t"].append(t)
        log["pos"].append(x[P].copy())
        log["vel"].append(x[V].copy())
        log["quat"].append(x[Q].copy())
        log["ref"].append(ref.pos)
        log["tilt"].append(np.degrees(np.arccos(np.clip(quat_to_rot(x[Q])[2, 2], -1, 1))))
        log["motors"].append(x[M].copy())
        if estimate:
            log["est_pos"].append(est.p.copy())
            log["est_vel"].append(est.v.copy())
            log["est_quat"].append(est.q.copy())
    return {k: np.array(v) for k, v in log.items()}, params


def estimation_errors(log):
    """估计误差:位置、速度 (m, m/s),倾斜方向误差和偏航误差 (度)。"""
    tilt, yaw = [], []
    for q, qe in zip(log["quat"][::10], log["est_quat"][::10]):
        R, Re = quat_to_rot(q), quat_to_rot(qe)
        tilt.append(np.degrees(np.arccos(np.clip(R[:, 2] @ Re[:, 2], -1, 1))))
        dyaw = np.arctan2(Re[1, 0], Re[0, 0]) - np.arctan2(R[1, 0], R[0, 0])
        yaw.append(np.degrees((dyaw + np.pi) % (2 * np.pi) - np.pi))
    return {
        "pos": log["est_pos"] - log["pos"],
        "vel": log["est_vel"] - log["vel"],
        "tilt_deg": np.array(tilt),
        "yaw_deg": np.array(yaw),
    }


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", action="store_true", help="控制器使用状态估计而不是真值")
    ap.add_argument("--seed", type=int, default=0, help="传感器噪声的随机种子")
    ap.add_argument("--no-calib", action="store_true", help="跳过上电陀螺零偏校准(对比用)")
    ap.add_argument("--ctrl", choices=list(CONTROLLERS), default="pid", help="控制器")
    ap.add_argument("--disturb", action="store_true", help="扰动场景:模型失配 + 侧风")
    ap.add_argument("--traj", choices=list(TRAJECTORIES), default="eight", help="参考轨迹")
    ap.add_argument("--design", action="store_true", help="用 mech/model.py 算出的机架参数")
    ap.add_argument("--fw", choices=["f32", "f64"], default=None,
                    help="软件在环:估计器和控制器用编译好的固件 C 代码(单精度/双精度)")
    args = ap.parse_args()
    design = None
    if args.design:
        from mech.model import design_params
        design = design_params()
    log, params = simulate(params=design, estimate=args.est, seed=args.seed, calibrate=not args.no_calib,
                           controller=args.ctrl, disturbed=args.disturb, trajectory=args.traj,
                           firmware=args.fw)
    if args.est:
        e = estimation_errors(log)
        print("--- 状态估计误差(估计值 - 真值)---")
        print(f"高度 RMS: {rms(e['pos'][:, 2]) * 100:.2f} cm"
              f"   水平位置漂移(结束时): {np.linalg.norm(e['pos'][-1, :2]) * 100:.1f} cm")
        print(f"速度 RMS: {rms(np.linalg.norm(e['vel'], axis=1)) * 100:.1f} cm/s")
        print(f"倾斜方向 RMS: {rms(e['tilt_deg']):.2f} 度"
              f"   偏航漂移(结束时): {e['yaw_deg'][-1]:.2f} 度")
        print("--- 控制效果(真实位置 vs 参考)---")
    t, err = log["t"], np.linalg.norm(log["pos"] - log["ref"], axis=1)
    i_hover = np.searchsorted(t, T_CALIB + T_HOVER) - 1
    track = t > T_CALIB + T_HOVER + 2.0
    print(f"起飞 {T_HOVER:.0f} s 后的悬停位置误差: {err[i_hover] * 100:.2f} cm")
    print(f"8 字跟踪 RMS 误差: {np.sqrt(np.mean(err[track] ** 2)) * 100:.2f} cm"
          f"   最大: {err[track].max() * 100:.2f} cm")
    print(f"最大倾角: {log['tilt'].max():.1f} 度")
    sat = np.mean(log["motors"] > 0.98 * params.w_max)
    print(f"电机接近饱和的时间占比: {sat * 100:.1f} %")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib,跳过画图。")
        return
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].plot(log["ref"][:, 0], log["ref"][:, 1], "--", label="reference")
    ax[0, 0].plot(log["pos"][:, 0], log["pos"][:, 1], label="actual")
    ax[0, 0].set(xlabel="x (m)", ylabel="y (m)", title="XY path")
    ax[0, 0].axis("equal")
    ax[0, 0].legend()
    for k, name in enumerate("xyz"):
        ax[0, 1].plot(t, log["pos"][:, k], label=name)
        ax[0, 1].plot(t, log["ref"][:, k], "k:", linewidth=0.8)
    ax[0, 1].set(xlabel="t (s)", ylabel="position (m)", title="Position vs reference")
    ax[0, 1].legend()
    ax[1, 0].plot(t, err * 100)
    ax[1, 0].set(xlabel="t (s)", ylabel="error (cm)", title="Position error")
    ax[1, 1].plot(t, log["motors"])
    ax[1, 1].axhline(params.w_max, color="k", linestyle=":")
    ax[1, 1].set(xlabel="t (s)", ylabel="rad/s", title="Motor speeds")
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "demo.png", dpi=120)
    print(f"曲线已保存到 {out / 'demo.png'}")


if __name__ == "__main__":
    main()
