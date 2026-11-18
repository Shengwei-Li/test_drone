"""软件在环的第二部分:MPC、通信协议、整机主循环的 Python 接口(基础部分见 sil.py)。"""
import ctypes
from dataclasses import astuple

import numpy as np

from firmware.sil import CPolicy, _Base, _state13, load
from sim.estimator import EstimatorConfig
from sim.params import QuadParams

_u8p = ctypes.POINTER(ctypes.c_uint8)


def _bytes(data: bytes):
    return (ctypes.c_uint8 * max(len(data), 1)).from_buffer_copy(data or bytes(1))


def _veh(params: QuadParams) -> np.ndarray:
    return np.concatenate([[params.mass, params.g, params.kf, params.f_max, params.motor_tau], params.inertia,
                           params.alloc.ravel(), np.linalg.inv(params.alloc).ravel()])


def mpc_tables(params: QuadParams, dt: float):
    """从 Python 版 MPC 里取出一个轴的增益表,按 fc_sil_mpc_init 要求的顺序排成一维数组。"""
    from sim.controller import MpcController
    py = MpcController(params, dt)
    m = py.mpc
    x_max = np.where(np.isfinite(m.x_max), m.x_max, 1e30).reshape(3, 3)
    tab = np.concatenate([
        m.A[:3, :3].ravel(), m.B[:3, 0], np.diag(m.Q)[:3], m.P[:3, :3].ravel(),
        m.K[:, 0, :3].ravel(), m.C1[:, 0, 0], m.AmBKt[:, :3, :3].ravel(),
        [m.rho_u[0]], m.rho_x[:3], [m.u_max[0]], x_max.ravel(), [py.dt_mpc, m.iters, py.solve_every]])
    return tab, m.N, py.dt_mpc


class CMpc(_Base):
    """C 版 MPC(位置层)+ INDI(内环)。"""

    def __init__(self, params: QuadParams, dt: float, precision: str = "f32"):
        super().__init__(precision)
        tab, self.N, self.dt_mpc = mpc_tables(params, dt)
        ft = ctypes.c_float if precision == "f32" else ctypes.c_double
        self.lib.fc_sil_mpc_table_len.restype = ctypes.c_int
        assert len(tab) == self.lib.fc_sil_mpc_table_len()
        self.lib.fc_sil_mpc_init.argtypes = [self._ptr, self._ptr, ft]
        self.lib.fc_sil_mpc_update.argtypes = [self._ptr, self._ptr, ft, self._ptr, self._ptr]
        self.lib.fc_sil_mpc_init.restype = self.lib.fc_sil_mpc_update.restype = None
        t, v = self._a(tab), self._a(_veh(params))
        self.lib.fc_sil_mpc_init(self._p(t), self._p(v), dt)
        self._traj = None
        self._w = np.zeros(4, dtype=self.dtype)
        self._preview = np.zeros((self.N + 1, 3), dtype=self.dtype)

    def set_trajectory(self, fn):
        self._traj = fn

    def update(self, x, ref, acc=None) -> np.ndarray:
        if self.lib.fc_sil_mpc_needs_preview():
            for i in range(self.N + 1):
                r = self._traj(ref.t + i * self.dt_mpc) if self._traj else ref
                self._preview[i] = r.pos
        s = self._a(_state13(x))
        a = None if acc is None else self._a(acc)
        self.lib.fc_sil_mpc_update(self._p(s), self._p(self._preview), ref.yaw,
                                   None if a is None else self._p(a), self._p(self._w))
        return self._w.astype(float)


def c_parse(data: bytes, precision: str = "f32"):
    """用 C 解析器解析一串字节,返回 ([(类型, 载荷)], 坏帧数)。"""
    lib = load(precision)[0]
    lib.fc_sil_proto_reset()
    out = (ctypes.c_uint8 * (len(data) + 16))()
    bad = ctypes.c_int(0)
    n = lib.fc_sil_proto_parse(_bytes(data), len(data), out, ctypes.byref(bad))
    msgs, w = [], 0
    for _ in range(n):
        t, ln = out[w], out[w + 1]
        msgs.append((t, bytes(out[w + 2:w + 2 + ln])))
        w += 2 + ln
    return msgs, bad.value


def c_encode(msg_type: int, payload: bytes = b"", precision: str = "f32") -> bytes:
    lib = load(precision)[0]
    out = (ctypes.c_uint8 * 80)()
    n = lib.fc_sil_proto_encode(msg_type, _bytes(payload), len(payload), out)
    return bytes(out[:n])


class FirmwareApp(_Base):
    """整个固件主循环 (fc_app_step):喂传感器读数和串口字节,拿回电机指令和要发出的字节。"""

    def __init__(self, params: QuadParams, dt: float, precision: str = "f32", use_policy: bool = False,
                 cfg: EstimatorConfig | None = None):
        super().__init__(precision)
        ft = ctypes.c_float if precision == "f32" else ctypes.c_double
        if use_policy:
            CPolicy(params, dt, precision=precision)          # 把权重装进库里
        self.lib.fc_sil_app_init.argtypes = [self._ptr, self._ptr, ft, ft, ft, ctypes.c_int]
        self.lib.fc_sil_app_init.restype = None
        self.lib.fc_sil_app_step.argtypes = [self._ptr, _u8p, ctypes.c_int, self._ptr, _u8p]
        self.lib.fc_sil_app_step.restype = ctypes.c_int
        c, v = self._a(astuple(cfg or EstimatorConfig())), self._a(_veh(params))
        self.lib.fc_sil_app_init(self._p(c), self._p(v), dt, params.w_hover, params.w_max, int(use_policy))
        self._out = np.zeros(17, dtype=self.dtype)
        self._tx = (ctypes.c_uint8 * 80)()

    def step(self, gyro, acc, flow, tof, vbat: float, rx: bytes = b""):
        sens = self._a([*gyro, *acc, flow is not None, *(flow if flow is not None else (0, 0)),
                        tof is not None, tof if tof is not None else 0.0, vbat])
        n = self.lib.fc_sil_app_step(self._p(sens), _bytes(rx), len(rx), self._p(self._out), self._tx)
        o = self._out.astype(float)
        return dict(motor=o[0:4], mode=int(o[4]), reason=int(o[5]), calibrated=bool(o[6]),
                    pos=o[7:10], vel=o[10:13], quat=o[13:17], tx=bytes(self._tx[:n]))
