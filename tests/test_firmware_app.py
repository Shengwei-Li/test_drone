"""固件的通信协议、状态机、MPC 和整机主循环。需要 C 编译器;没有时跳过。"""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firmware import proto  # noqa: E402
from run_demo import simulate  # noqa: E402
from sim.params import QuadParams  # noqa: E402

try:
    from firmware.build import ensure
    ensure("f32"), ensure("f64")
    from firmware.app_sim import run
    from firmware.sil_app import FirmwareApp, c_encode, c_parse
except Exception as exc:                                    # noqa: BLE001
    pytest.skip(f"无法编译固件库: {exc}", allow_module_level=True)

DISARMED, FLYING, LANDING, FAULT = 0, 1, 2, 3


# ---------------- 通信协议 ----------------

def test_protocol_c_and_python_agree():
    """两份独立实现:Python 组的帧 C 能解,C 组的帧 Python 能解,字节完全相同。"""
    payload = struct.pack("<10f", *np.arange(10) * 0.25)
    for mtype, pl in ((proto.HEARTBEAT, b""), (proto.SETPOINT, payload), (proto.ARM, b"")):
        frame = proto.encode(mtype, pl)
        assert c_encode(mtype, pl) == frame
        assert c_parse(frame) == ([(mtype, pl)], 0)
        assert proto.Parser().feed(frame) == [(mtype, pl)]


def test_protocol_survives_garbage_and_corruption():
    """帧之间夹杂乱码、某一帧被改坏一位:坏帧被丢弃,其余帧照常解出。"""
    rng = np.random.default_rng(0)
    frames = [proto.setpoint(rng.normal(size=3)) for _ in range(20)]
    bad = bytearray(frames[7])
    bad[10] ^= 0x04
    stream = b""
    for i, f in enumerate(frames):
        stream += bytes(rng.integers(0, 256, 5, dtype=np.uint8)) + (bytes(bad) if i == 7 else f)
    msgs, n_bad = c_parse(stream)
    good = [(proto.SETPOINT, f[3:-2]) for i, f in enumerate(frames) if i != 7]
    assert [m for m in msgs if m in good] == good            # 好帧一个不少、顺序不变
    assert (proto.SETPOINT, frames[7][3:-2]) not in msgs     # 坏帧没有被当成好帧
    assert n_bad >= 1
    assert [m for m in proto.Parser().feed(stream) if m in good] == good


# ---------------- 状态机 ----------------

def _still(app, n, rx=b"", vbat=3.9):
    out = None
    for k in range(n):
        out = app.step(np.zeros(3), np.array([0.0, 0.0, 9.81]), None, None, vbat, rx if k == 0 else b"")
    return out


def test_arm_refused_until_ready():
    """没校准完、电压不足时解锁都应被拒绝;条件满足后才能解锁。"""
    app = FirmwareApp(QuadParams(), 0.002)
    out = _still(app, 5, proto.encode(proto.ARM))
    assert out["mode"] == DISARMED and out["reason"] == 7 and not out["calibrated"]
    out = _still(app, 500)
    assert out["calibrated"]
    out = _still(app, 5, proto.encode(proto.ARM), vbat=3.0)
    assert out["mode"] == DISARMED and out["reason"] == 7
    out = _still(app, 1, proto.encode(proto.ARM))
    assert out["mode"] == FLYING and out["motor"].max() > 0


def test_flip_cuts_motors_and_disarm_clears_fault():
    """解锁后机体翻过去超过 0.5 s:进入故障、电机停转;收到上锁指令后回到上锁状态。"""
    app = FirmwareApp(QuadParams(), 0.002)
    _still(app, 500)
    assert _still(app, 1, proto.encode(proto.ARM))["mode"] == FLYING
    out = None
    for k in range(600):                                     # 绕 x 轴 4 rad/s 转 1.2 s,约 275°
        rx = proto.encode(proto.HEARTBEAT) if k % 10 == 0 else b""
        out = app.step(np.array([4.0, 0, 0]), np.array([0.0, 0.0, 9.81]), None, None, 3.9, rx)
        if out["mode"] == FAULT:
            break
    assert out["mode"] == FAULT and out["reason"] == 5 and out["motor"].max() == 0
    out = _still(app, 1, proto.encode(proto.DISARM))
    assert out["mode"] == DISARMED


# ---------------- MPC ----------------

@pytest.mark.parametrize("traj", ["eight", "waypoints"])
def test_mpc_matches_python(traj):
    ref = simulate(controller="mpc", disturbed=True, trajectory=traj)[0]
    for prec, tol in (("f64", 1e-9), ("f32", 1e-4)):
        log = simulate(controller="mpc", disturbed=True, trajectory=traj, firmware=prec)[0]
        assert np.abs(log["pos"] - ref["pos"]).max() < tol, prec


# ---------------- 整机 ----------------

def _transitions(log):
    k = np.flatnonzero(np.diff(log["mode"])) + 1
    return [(round(float(log["t"][i]), 2), int(log["mode"][i]), int(log["reason"][i])) for i in k]


def _on_ground_disarmed(log):
    return log["mode"][-1] == DISARMED and log["pos"][-1, 2] < 0.01 and log["motor"][-1].max() == 0


@pytest.mark.parametrize("policy", [False, True])
def test_full_flight_through_protocol(policy):
    """整个固件主循环:校准 → 解锁 → 跟 8 字 → 收到降落指令 → 触地上锁,指令全部经协议帧下发。"""
    log = run("normal", use_policy=policy)
    tr = _transitions(log)
    assert [m for _, m, _ in tr] == [FLYING, LANDING, DISARMED]
    assert tr[1][0] == pytest.approx(14.0, abs=0.05) and tr[2][2] == 8
    fly = (log["mode"] == FLYING) & (log["t"] > 6.0)
    err = np.linalg.norm(log["pos"] - log["ref"], axis=1)[fly]
    assert np.sqrt(np.mean(err ** 2)) < 0.25
    assert _on_ground_disarmed(log)
    tele = np.linalg.norm(log["tele_pos"] - log["est_pos"], axis=1)[fly]
    assert tele.max() < 0.08                                 # 地面收到的遥测与机上估计一致(最多滞后一帧)


def test_link_loss_lands():
    log = run("link_loss", t_end=14.0)
    tr = _transitions(log)
    assert tr[1][1:] == (LANDING, 2) and 8.4 < tr[1][0] < 8.7          # 丢失 0.5 s 后判定
    assert _on_ground_disarmed(log)
    i = np.searchsorted(log["t"], tr[1][0])
    assert np.linalg.norm(log["pos"][-1, :2] - log["pos"][i, :2]) < 0.4  # 基本在原地降落


def test_low_battery_lands():
    log = run("low_battery", t_end=14.0)
    tr = _transitions(log)
    assert tr[1][1:] == (LANDING, 3) and 8.9 < tr[1][0] < 9.2          # 电压低于阈值持续 1 s
    assert _on_ground_disarmed(log)


def test_noisy_link_still_flies():
    """链路上有 1% 的位错误和乱码:坏帧被丢弃,飞行不受影响。"""
    log = run("noisy_link")
    assert log["bad_injected"] > 100
    assert [m for _, m, _ in _transitions(log)] == [FLYING, LANDING, DISARMED]
    assert _on_ground_disarmed(log)
