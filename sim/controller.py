"""串级控制器:位置 → 速度 → 姿态(四元数)→ 角速度 → 混控。

位置环比内环慢(默认每 5 次调用算一次)。这里用的是理想状态反馈,
还没有传感器噪声和状态估计;角速度环是 PI 加陀螺力矩前馈,没有 D 项。
"""
from dataclasses import dataclass, field

import numpy as np

from .dynamics import P, Q, V, W, quat_conj, quat_mul, quat_to_rot, rot_to_quat
from .params import QuadParams


@dataclass
class Reference:
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    jerk: np.ndarray = field(default_factory=lambda: np.zeros(3))   # 加速度的导数
    yaw: float = 0.0
    t: float = 0.0                                                  # 这个参考点对应的时刻(MPC 预瞄用)


@dataclass
class Gains:
    kp_pos: np.ndarray = field(default_factory=lambda: np.array([1.5, 1.5, 2.0]))
    kp_vel: np.ndarray = field(default_factory=lambda: np.array([3.0, 3.0, 4.0]))
    ki_vel: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 1.0]))
    kp_att: np.ndarray = field(default_factory=lambda: np.array([6.0, 6.0, 3.0]))
    kp_rate: np.ndarray = field(default_factory=lambda: np.array([15.0, 15.0, 10.0]))
    ki_rate: np.ndarray = field(default_factory=lambda: np.array([5.0, 5.0, 2.0]))
    v_max: float = 2.0            # m/s,速度指令限幅
    a_xy_max: float = 6.0         # m/s²,水平加速度限幅(约 31° 倾角)
    a_z_max: float = 6.0          # m/s²
    int_vel_max: float = 2.0      # m/s²,速度环积分限幅
    int_rate_max: float = 20.0    # rad/s²,角速度环积分限幅


class CascadedController:
    def __init__(self, params: QuadParams, dt: float, gains: Gains | None = None,
                 pos_divider: int = 5):
        self.p = params
        self.dt = dt
        self.g = gains or Gains()
        self.pos_divider = pos_divider
        self.alloc_inv = np.linalg.inv(params.alloc)
        self.reset()

    def reset(self):
        self._n = 0
        self._int_vel = np.zeros(3)
        self._int_rate = np.zeros(3)
        self._a_des = np.zeros(3)

    def update(self, x: np.ndarray, ref: Reference, acc=None) -> np.ndarray:
        """返回四个电机的转速指令 (rad/s)。acc(加速度计读数)这里用不到。"""
        p, g = self.p, self.g
        pos, vel, q, omega = x[P], x[V], x[Q], x[W]
        R = quat_to_rot(q)

        # 位置环与速度环(慢速)
        if self._n % self.pos_divider == 0:
            dt_pos = self.dt * self.pos_divider
            v_des = np.clip(g.kp_pos * (ref.pos - pos) + ref.vel, -g.v_max, g.v_max)
            e_v = v_des - vel
            self._int_vel = np.clip(self._int_vel + g.ki_vel * e_v * dt_pos,
                                    -g.int_vel_max, g.int_vel_max)
            a = g.kp_vel * e_v + self._int_vel + ref.acc
            n_xy = np.linalg.norm(a[:2])
            if n_xy > g.a_xy_max:
                a[:2] *= g.a_xy_max / n_xy
            a[2] = np.clip(a[2], -g.a_z_max, g.a_z_max)
            self._a_des = a
        self._n += 1

        # 期望推力向量 → 总推力与期望姿态
        f_des = p.mass * (self._a_des + np.array([0.0, 0.0, p.g]))
        thrust = float(np.clip(f_des @ R[:, 2], 0.0, 4.0 * p.f_max))

        zb = f_des / max(np.linalg.norm(f_des), 1e-9)
        xc = np.array([np.cos(ref.yaw), np.sin(ref.yaw), 0.0])
        yb = np.cross(zb, xc)
        yb /= max(np.linalg.norm(yb), 1e-9)
        xb = np.cross(yb, zb)
        q_des = rot_to_quat(np.column_stack([xb, yb, zb]))

        # 姿态环:四元数误差(机体系)→ 期望角速度
        q_err = quat_mul(quat_conj(q), q_des)
        if q_err[0] < 0:
            q_err = -q_err
        omega_des = g.kp_att * 2.0 * q_err[1:]

        # 角速度环:PI → 角加速度 → 力矩
        e_w = omega_des - omega
        self._int_rate = np.clip(self._int_rate + g.ki_rate * e_w * self.dt,
                                 -g.int_rate_max, g.int_rate_max)
        alpha = g.kp_rate * e_w + self._int_rate
        J = p.J
        tau = J @ alpha + np.cross(omega, J @ omega)

        return self.mix(thrust, tau)

    def mix(self, thrust: float, tau: np.ndarray) -> np.ndarray:
        """[总推力, 力矩] → 各电机转速指令。饱和时只做简单限幅。"""
        f = self.alloc_inv @ np.array([thrust, *tau])
        f = np.clip(f, 0.0, self.p.f_max)
        return np.sqrt(f / self.p.kf)


# ---------------------------------------------------------------------------
# 几何控制
# ---------------------------------------------------------------------------

def vee(S: np.ndarray) -> np.ndarray:
    """反对称矩阵 → 向量。"""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


@dataclass
class GeoGains:
    kx: np.ndarray = field(default_factory=lambda: np.array([6.0, 6.0, 10.0]))   # 1/s²
    kv: np.ndarray = field(default_factory=lambda: np.array([4.0, 4.0, 5.0]))    # 1/s
    kR: np.ndarray = field(default_factory=lambda: np.array([110.0, 110.0, 30.0]))  # 1/s²
    kW: np.ndarray = field(default_factory=lambda: np.array([18.0, 18.0, 10.0]))    # 1/s
    a_xy_max: float = 6.0
    a_z_max: float = 6.0


class GeometricController(CascadedController):
    """SE(3) 几何跟踪控制(Lee 等 2010 的结构),带微分平坦前馈。

    与串级 PID 的区别:
    - 姿态误差直接在旋转矩阵上定义,大角度下也成立;
    - 用参考轨迹的加加速度 (jerk) 算出期望角速度作前馈,机体提前转动,
      而不是等位置误差出现后再追;
    - 没有积分项,所以对恒定扰动和模型误差会留下稳态误差。
    """

    def __init__(self, params: QuadParams, dt: float, gains: GeoGains | None = None):
        self.geo = gains or GeoGains()
        super().__init__(params, dt)

    def update(self, x: np.ndarray, ref: Reference, acc=None) -> np.ndarray:
        p, k = self.p, self.geo
        pos, vel, omega = x[P], x[V], x[W]
        R = quat_to_rot(x[Q])

        a = k.kx * (ref.pos - pos) + k.kv * (ref.vel - vel) + ref.acc
        n_xy = np.linalg.norm(a[:2])
        if n_xy > k.a_xy_max:
            a[:2] *= k.a_xy_max / n_xy
        a[2] = np.clip(a[2], -k.a_z_max, k.a_z_max)

        f_des = p.mass * (a + np.array([0.0, 0.0, p.g])) - self._force_disturbance(x, R, acc)
        thrust = float(np.clip(f_des @ R[:, 2], 0.0, 4.0 * p.f_max))

        # 期望姿态:机体 z 轴对准期望推力方向,再由偏航角定出 x、y 轴
        f_norm = max(np.linalg.norm(f_des), 1e-6)
        zb = f_des / f_norm
        xc = np.array([np.cos(ref.yaw), np.sin(ref.yaw), 0.0])
        yb = np.cross(zb, xc)
        yb /= max(np.linalg.norm(yb), 1e-9)
        xb = np.cross(yb, zb)
        R_d = np.column_stack([xb, yb, zb])

        # 微分平坦:推力方向的变化率由参考 jerk 决定 → 期望角速度(偏航率取 0)
        h = p.mass / f_norm * (ref.jerk - (zb @ ref.jerk) * zb)
        omega_d = np.array([-h @ yb, h @ xb, 0.0])

        e_R = 0.5 * vee(R_d.T @ R - R.T @ R_d)
        e_W = omega - R.T @ R_d @ omega_d
        alpha_des = -k.kR * e_R - k.kW * e_W

        w_cmd = self.mix(thrust, self._torque(alpha_des, omega))
        self._after_mix(w_cmd)
        return w_cmd

    # 以下三个钩子由 INDI 重写
    def _force_disturbance(self, x, R, acc) -> np.ndarray:
        return np.zeros(3)

    def _torque(self, alpha_des: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """基于模型的求逆:需要准确的转动惯量,且假设没有未建模的力矩。"""
        J = self.p.J
        return J @ alpha_des + np.cross(omega, J @ omega)

    def _after_mix(self, w_cmd: np.ndarray):
        pass


# ---------------------------------------------------------------------------
# INDI(增量非线性动态逆)
# ---------------------------------------------------------------------------

class IndiController(GeometricController):
    """在几何控制的基础上,把"基于模型算力矩/推力"换成"基于测量做增量"。

    力矩:  τ_指令 = τ_当前 + J · (α_期望 − α_测量)
        α_测量 是滤波后陀螺读数的导数,τ_当前 由电机转速模型算出(同样滤波,保持同步)。
        任何未建模的力矩(电机不一致、重心偏移、阵风)都已经体现在 α_测量 里,
        所以不需要知道它是什么,也不需要积分项慢慢去补。
    推力:  用加速度计量到的实际比力减去模型推力,得到外力估计,再从期望推力里扣掉。

    这里没有电机转速反馈,转速是用一阶模型从指令推出来的(有刷电机的做法)。
    """

    def __init__(self, params: QuadParams, dt: float, gains: GeoGains | None = None,
                 fc: float = 30.0, fc_force: float = 2.0):
        tau_f = 1.0 / (2 * np.pi * fc)
        self._a = dt / (dt + tau_f)                       # 角速度/力矩低通系数
        self._a_force = dt / (dt + 1.0 / (2 * np.pi * fc_force))
        self._a_motor = 1.0 - np.exp(-dt / params.motor_tau)
        super().__init__(params, dt, gains)

    def reset(self):
        super().reset()
        self._w_hat = np.zeros(4)        # 电机转速的模型估计
        self._omega_f = None             # 滤波后的角速度
        self._tau_f = np.zeros(3)        # 滤波后的模型力矩
        self._d_hat = np.zeros(3)        # 外力估计(世界系)

    def _force_disturbance(self, x, R, acc) -> np.ndarray:
        if acc is not None and x[P][2] > 0.05:            # 离地后才更新,避免把地面支持力当成扰动
            p = self.p
            thrust_model = p.kf * np.sum(self._w_hat ** 2)
            d_raw = p.mass * (R @ acc) - thrust_model * R[:, 2]
            self._d_hat += self._a_force * (d_raw - self._d_hat)
        return self._d_hat

    def _torque(self, alpha_des: np.ndarray, omega: np.ndarray) -> np.ndarray:
        if self._omega_f is None:
            self._omega_f = omega.copy()
        prev = self._omega_f
        self._omega_f = prev + self._a * (omega - prev)
        alpha_f = (self._omega_f - prev) / self.dt
        tau_model = (self.p.alloc @ (self.p.kf * self._w_hat ** 2))[1:]
        self._tau_f += self._a * (tau_model - self._tau_f)
        return self._tau_f + self.p.J @ (alpha_des - alpha_f)

    def _after_mix(self, w_cmd: np.ndarray):
        self._w_hat += self._a_motor * (w_cmd - self._w_hat)


# ---------------------------------------------------------------------------
# MPC(模型预测控制)外环 + INDI 内环
# ---------------------------------------------------------------------------

class MpcController(IndiController):
    """位置层用线性 MPC 做规划,姿态和抗扰仍交给 INDI。

    MPC 的模型是每个轴一个三重积分器:状态 [位置, 速度, 加速度],输入是加加速度 (jerk)。
    每 solve_every 个控制周期,从当前测得的位置/速度出发,向前看 horizon·dt_mpc 秒,
    在速度、加速度、jerk 都不超限的前提下,求一条最贴近未来参考位置的轨迹;
    只执行第一段,下个周期重新求解。

    与前面几种控制器的区别:
    - 只需要未来的参考"位置",不需要解析的速度/加速度/jerk 前馈;
    - 约束是规划时就考虑进去的,而不是算完指令再硬限幅;
    - 能提前看到参考的变化(比如航点跳变),提前开始动作。

    加速度这一维是 MPC 内部的状态(上一次规划的结果),不是测量值;
    实际加速度由 INDI 内环去实现,外力由 INDI 的外力观测器扣掉。
    """

    def __init__(self, params: QuadParams, dt: float, gains: GeoGains | None = None,
                 horizon: int = 25, dt_mpc: float = 0.04, solve_every: int = 10,
                 q_pos: float = 400.0, q_acc: float = 0.05, r_jerk: float = 0.005,
                 v_max: float = 2.0, a_xy_max: float = 4.0, a_z_max: float = 5.0,
                 jerk_max: float = 40.0, iters: int = 10):
        from .mpc import AdmmMpc
        h = dt_mpc
        A1 = np.array([[1.0, h, 0.5 * h * h], [0.0, 1.0, h], [0.0, 0.0, 1.0]])
        B1 = np.array([[h ** 3 / 6.0], [0.5 * h * h], [h]])
        eye = np.eye(3)
        x_max = np.array([[np.inf, v_max, a_xy_max]] * 2 + [[np.inf, v_max, a_z_max]]).ravel()
        self.mpc = AdmmMpc(
            A=np.kron(eye, A1), B=np.kron(eye, B1),
            Q=np.kron(eye, np.diag([q_pos, 0.0, q_acc])), R=r_jerk * eye,
            horizon=horizon, u_min=-jerk_max * np.ones(3), u_max=jerk_max * np.ones(3),
            x_min=-x_max, x_max=x_max, rho_u=r_jerk, rho_x=10.0, iters=iters)
        self.dt_mpc, self.solve_every = dt_mpc, solve_every
        self._traj = None
        super().__init__(params, dt, gains)

    def set_trajectory(self, fn):
        """fn(t) → Reference。不设置时,MPC 把当前参考点当作整个预测时域的目标。"""
        self._traj = fn

    def reset(self):
        super().reset()
        self.mpc.reset()
        self._k = 0
        self._acc_plan = np.zeros(3)     # 规划的加速度(MPC 内部状态)
        self._p0 = self._v0 = self._a0 = self._j0 = np.zeros(3)

    def update(self, x: np.ndarray, ref: Reference, acc=None) -> np.ndarray:
        j = self._k % self.solve_every
        if j == 0:
            N = self.mpc.N
            x_ref = np.zeros((N + 1, 9))
            for i in range(N + 1):
                r = self._traj(ref.t + i * self.dt_mpc) if self._traj else ref
                x_ref[i, 0::3] = r.pos
            self._p0, self._v0, self._a0 = x[P].copy(), x[V].copy(), self._acc_plan.copy()
            x0 = np.column_stack([self._p0, self._v0, self._a0]).ravel()
            self._j0, _ = self.mpc.solve(x0, x_ref)
            self._acc_plan = self._a0 + self._j0 * self.solve_every * self.dt
        self._k += 1

        # 两次求解之间,沿规划的第一段(恒定 jerk)插值,作为内环的参考
        s = j * self.dt
        inner = Reference(
            pos=self._p0 + self._v0 * s + 0.5 * self._a0 * s * s + self._j0 * s ** 3 / 6.0,
            vel=self._v0 + self._a0 * s + 0.5 * self._j0 * s * s,
            acc=self._a0 + self._j0 * s,
            jerk=self._j0, yaw=ref.yaw)
        return super().update(x, inner, acc)


CONTROLLERS = {
    "pid": CascadedController,
    "geometric": GeometricController,
    "indi": IndiController,
    "mpc": MpcController,
}


def _rl(params, dt):
    from .rl_policy import RlController          # 延迟导入:没有训练好的策略文件时不影响其他控制器
    return RlController(params, dt)


CONTROLLERS["rl"] = _rl
