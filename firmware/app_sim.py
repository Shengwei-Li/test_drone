"""整机软件在环:完整的固件主循环 + 地面站,通过串口协议对话。

用法:  python -m firmware.app_sim [--scenario normal|link_loss|low_battery|noisy_link] [--policy]

与 run_demo.py 的区别:那里是仿真主循环直接调用估计器和控制器;
这里仿真只扮演"物理世界 + 传感器 + 地面站",解锁、目标点、降落都靠协议帧发给固件,
固件自己决定什么时候校准、什么时候起飞、出问题时怎么办。
"""
import argparse

import numpy as np

from firmware import proto
from firmware.sil_app import FirmwareApp
from run_demo import CTRL_EVERY, FLOW_EVERY, SIM_DT, T_CALIB, T_HOVER, TOF_EVERY, reference
from sim.dynamics import P, initial_state, step
from sim.params import QuadParams
from sim.sensors import SensorSuite

GROUND_EVERY = 20           # 地面站每 20 ms 发一次(50 Hz)
SCENARIOS = ("normal", "link_loss", "low_battery", "noisy_link")


def run(scenario: str = "normal", precision: str = "f32", use_policy: bool = False, seed: int = 0,
        t_end: float = 22.0, t_land: float = 14.0, t_event: float = 8.0):
    """返回日志字典。t_event:链路丢失/电压跌落发生的时刻。"""
    assert scenario in SCENARIOS
    p = QuadParams()
    dt = SIM_DT * CTRL_EVERY
    app = FirmwareApp(p, dt, precision, use_policy)
    sensors = SensorSuite(p, None, np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 100)
    ground = proto.Parser()
    x = initial_state(p, hover=False)
    out = dict(motor=np.zeros(4), mode=0, reason=0, pos=np.zeros(3), calibrated=False)
    log = {k: [] for k in ("t", "pos", "ref", "mode", "reason", "est_pos", "tele_pos", "motor")}
    tele, armed_sent, land_sent, bad_injected = None, False, False, 0
    for i in range(int(t_end / SIM_DT)):
        t = i * SIM_DT
        ref = reference(t)
        if i % CTRL_EVERY == 0:
            # ---- 地面站 → 飞控 ----
            rx = b""
            link_up = not (scenario == "link_loss" and t >= t_event)
            if link_up and i % GROUND_EVERY == 0:
                rx += proto.encode(proto.HEARTBEAT) + proto.setpoint(ref.pos, ref.vel, ref.acc)
                if not armed_sent and t >= T_CALIB and out["calibrated"]:
                    rx += proto.encode(proto.ARM)
                    armed_sent = True
                if not land_sent and t >= t_land:
                    rx += proto.encode(proto.LAND)
                    land_sent = True
                if scenario == "noisy_link":                  # 干扰:随机翻转一些位,再夹杂几个乱码字节
                    b = bytearray(rx)
                    for k in range(len(b)):
                        if rng.random() < 0.01:
                            b[k] ^= 1 << int(rng.integers(8))
                            bad_injected += 1
                    rx = bytes(rng.integers(0, 256, int(rng.integers(0, 4)), dtype=np.uint8)) + bytes(b)
            # ---- 传感器 ----
            gyro, acc = sensors.imu(x, dt)
            flow = sensors.flow(x) if i % FLOW_EVERY == 0 else None
            tof = sensors.tof(x) if i % TOF_EVERY == 0 else None
            vbat = 3.2 if scenario == "low_battery" and t >= t_event else 3.9
            out = app.step(gyro, acc, flow, tof, vbat, rx)
            # ---- 飞控 → 地面站 ----
            for mtype, payload in ground.feed(out["tx"]):
                if mtype == proto.TELEMETRY:
                    tele = proto.parse_telemetry(payload)
        x = step(x, out["motor"], p, SIM_DT, ground=True)
        if i % 10 == 0:
            log["t"].append(t)
            log["pos"].append(x[P].copy())
            log["ref"].append(ref.pos)
            log["mode"].append(out["mode"])
            log["reason"].append(out["reason"])
            log["est_pos"].append(out["pos"].copy())
            log["tele_pos"].append(tele.pos.copy() if tele else np.zeros(3))
            log["motor"].append(out["motor"].copy())
    log = {k: np.array(v) for k, v in log.items()}
    log["bad_injected"] = bad_injected
    return log


def summarize(log):
    t, mode = log["t"], log["mode"]
    lines = []
    for k in np.flatnonzero(np.diff(mode)) + 1:
        lines.append(f"  t = {t[k]:5.2f} s  {proto.MODES[mode[k - 1]]} → {proto.MODES[mode[k]]}"
                     f"(原因:{proto.REASONS[log['reason'][k]]})")
    fly = (mode == 1) & (t > T_CALIB + T_HOVER + 2.0)
    if fly.any():
        err = np.linalg.norm(log["pos"] - log["ref"], axis=1)[fly]
        lines.append(f"  飞行段跟踪 RMS 误差 {np.sqrt(np.mean(err ** 2)) * 100:.1f} cm")
    lines.append(f"  结束时:{proto.MODES[mode[-1]]},高度 {log['pos'][-1, 2] * 100:.1f} cm,"
                 f"电机指令 {log['motor'][-1].max():.0f} rad/s")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=SCENARIOS, default=None)
    ap.add_argument("--policy", action="store_true", help="飞行时用强化学习策略代替 INDI")
    args = ap.parse_args()
    for sc in ([args.scenario] if args.scenario else SCENARIOS):
        print(f"场景 {sc}:")
        print(summarize(run(sc, use_policy=args.policy)))


if __name__ == "__main__":
    main()
