"""软件在环:用 ctypes 调用编译好的固件核心库,包装成与 Python 版本相同的接口。

CEskf 对应 sim.estimator.ESKF,CController 对应 sim.controller 里的几何/INDI 控制器,
CPolicy 对应 sim.rl_policy.RlController。仿真主循环(run_demo.simulate)不用改,
只是把估计器和控制器换成这些类 —— 飞行时实际执行的就是将来要烧进单片机的那份 C 代码。

precision:"f32" 单精度(与 STM32F4 上一致)或 "f64" 双精度(用来和 Python 逐位对照)。
注意:库里的状态是全局的,同一精度同一时刻只能有一个估计器和一个控制器实例。
"""
import ctypes
from dataclasses import astuple

import numpy as np

from firmware.build import ensure
from sim.dynamics import NX, P, Q, V, W
from sim.estimator import EstimatorConfig
from sim.params import QuadParams
from sim.rl_policy import DEFAULT_POLICY, POLICY_DT

_libs = {}


def load(precision: str):
    if precision not in _libs:
        lib = ctypes.CDLL(str(ensure(precision)))
        ft = ctypes.c_float if precision == "f32" else ctypes.c_double
        assert lib.fc_sil_float_size() == ctypes.sizeof(ft)
        ptr = ctypes.POINTER(ft)
        lib.fc_sil_eskf_init.argtypes = [ptr, ft, ptr]
        lib.fc_sil_eskf_calibrate.argtypes = [ptr, ctypes.c_int, ft]
        lib.fc_sil_eskf_predict.argtypes = [ptr, ptr, ft]
        lib.fc_sil_eskf_update_tof.argtypes = [ft]
        lib.fc_sil_eskf_update_flow.argtypes = [ptr]
        lib.fc_sil_eskf_get.argtypes = [ptr, ptr]
        lib.fc_sil_ctrl_init.argtypes = [ptr, ft, ctypes.c_int]
        lib.fc_sil_ctrl_update.argtypes = [ptr, ptr, ptr, ptr]
        lib.fc_sil_policy_load.argtypes = [ptr] * 6 + [ft, ft]
        lib.fc_sil_policy_update.argtypes = [ptr, ptr, ptr]
        for f in ("eskf_init", "eskf_predict", "eskf_update_tof", "eskf_update_flow", "eskf_get",
                  "ctrl_init", "ctrl_update", "policy_load", "policy_update"):
            getattr(lib, "fc_sil_" + f).restype = None
        _libs[precision] = (lib, np.float32 if precision == "f32" else np.float64, ptr)
    return _libs[precision]


class _Base:
    def __init__(self, precision: str):
        self.lib, self.dtype, self._ptr = load(precision)

    def _a(self, x):
        """转成库用的精度的连续数组(返回数组本身,调用期间要保持引用)。"""
        return np.ascontiguousarray(x, dtype=self.dtype)

    def _p(self, arr):
        return arr.ctypes.data_as(self._ptr)


class CEskf(_Base):
    def __init__(self, params: QuadParams, cfg: EstimatorConfig | None = None, pos0=(0.0, 0.0, 0.0),
                 precision: str = "f32"):
        super().__init__(precision)
        self.cfg = cfg or EstimatorConfig()
        c, p0 = self._a(astuple(self.cfg)), self._a(pos0)
        self.lib.fc_sil_eskf_init(self._p(c), params.g, self._p(p0))
        self._out = np.zeros(19, dtype=self.dtype)
        self._refresh()

    def _refresh(self):
        self.lib.fc_sil_eskf_get(self._p(self._out), None)
        o = self._out.astype(float)
        self.p, self.v, self.q, self.bg, self.ba, self.gyro = o[0:3], o[3:6], o[6:10], o[10:13], o[13:16], o[16:19]

    @property
    def Pm(self):
        buf = np.zeros(225, dtype=self.dtype)
        self.lib.fc_sil_eskf_get(self._p(self._out), self._p(buf))
        return buf.astype(float).reshape(15, 15)

    def calibrate_gyro(self, samples, max_std: float = 0.1) -> bool:
        s = self._a(samples)
        ok = bool(self.lib.fc_sil_eskf_calibrate(self._p(s), len(s), max_std))
        self._refresh()
        return ok

    def predict(self, gyro, acc, dt: float):
        g, a = self._a(gyro), self._a(acc)
        self.lib.fc_sil_eskf_predict(self._p(g), self._p(a), dt)
        self._refresh()

    def update_tof(self, d: float):
        self.lib.fc_sil_eskf_update_tof(float(d))
        self._refresh()

    def update_flow(self, flow):
        f = self._a(flow)
        self.lib.fc_sil_eskf_update_flow(self._p(f))
        self._refresh()

    def state_vector(self) -> np.ndarray:
        x = np.zeros(NX)
        x[P], x[V], x[Q] = self.p, self.v, self.q
        x[W] = self.gyro - self.bg
        return x


def _state13(x):
    return np.concatenate([x[P], x[V], x[Q], x[W]])


class CController(_Base):
    """C 版几何控制 (use_indi=False) 或 INDI (use_indi=True)。"""

    def __init__(self, params: QuadParams, dt: float, use_indi: bool = True, precision: str = "f32"):
        super().__init__(precision)
        veh = self._a(np.concatenate([
            [params.mass, params.g, params.kf, params.f_max, params.motor_tau], params.inertia,
            params.alloc.ravel(), np.linalg.inv(params.alloc).ravel()]))
        self.lib.fc_sil_ctrl_init(self._p(veh), dt, int(use_indi))
        self._w = np.zeros(4, dtype=self.dtype)

    def update(self, x, ref, acc=None) -> np.ndarray:
        s = self._a(_state13(x))
        r = self._a(np.concatenate([ref.pos, ref.vel, ref.acc, ref.jerk, [ref.yaw]]))
        a = None if acc is None else self._a(acc)
        self.lib.fc_sil_ctrl_update(self._p(s), self._p(r), None if a is None else self._p(a), self._p(self._w))
        return self._w.astype(float)


class CPolicy(_Base):
    """C 版强化学习策略推理。"""

    def __init__(self, params: QuadParams, dt: float, path=None, precision: str = "f32"):
        super().__init__(precision)
        d = np.load(path or DEFAULT_POLICY)
        arrs = [self._a(d[k]) for k in ("W0", "b0", "W1", "b1", "W2", "b2")]
        self.lib.fc_sil_policy_load(*[self._p(a) for a in arrs], params.w_hover, params.w_max)
        self.every = max(1, round(POLICY_DT / dt))
        self._k = 0
        self._w = np.full(4, params.w_hover, dtype=self.dtype)

    def update(self, x, ref, acc=None) -> np.ndarray:
        if self._k % self.every == 0:
            s = self._a(_state13(x))
            r = self._a(np.concatenate([ref.pos, ref.vel, ref.acc]))
            self.lib.fc_sil_policy_update(self._p(s), self._p(r), self._p(self._w))
        self._k += 1
        return self._w.astype(float)


def make_controller(name: str, params: QuadParams, dt: float, precision: str):
    if name == "indi":
        return CController(params, dt, True, precision)
    if name == "geometric":
        return CController(params, dt, False, precision)
    if name == "rl":
        return CPolicy(params, dt, precision=precision)
    if name == "mpc":
        from firmware.sil_app import CMpc
        return CMpc(params, dt, precision)
    raise ValueError(f"固件里还没有 {name} 控制器(目前有 geometric、indi、mpc、rl)")
