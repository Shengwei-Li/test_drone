"""四旋翼 6 自由度刚体动力学,带电机一阶滞后,RK4 积分。

状态向量 x (17,):
    位置 p(3, 世界系) | 速度 v(3, 世界系) | 四元数 q(4, wxyz)
    | 角速度 ω(3, 机体系) | 电机转速 w(4, rad/s)
输入:四个电机的转速指令 (rad/s)。
"""
import numpy as np

from .params import QuadParams

P = slice(0, 3)
V = slice(3, 6)
Q = slice(6, 10)
W = slice(10, 13)
M = slice(13, 17)
NX = 17


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """机体系到世界系的旋转矩阵。"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """旋转矩阵转四元数(Shepperd 方法,数值稳定)。"""
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def initial_state(params: QuadParams, pos=(0.0, 0.0, 0.0), hover: bool = True) -> np.ndarray:
    """静止、水平的初始状态。hover=True 时电机已在悬停转速。"""
    x = np.zeros(NX)
    x[P] = pos
    x[Q] = [1.0, 0.0, 0.0, 0.0]
    if hover:
        x[M] = params.w_hover
    return x


def motor_thrusts(x: np.ndarray, p: QuadParams) -> np.ndarray:
    """各电机的实际推力 (N)。"""
    return p.kf * np.asarray(p.motor_gain) * x[M] ** 2


def deriv(x: np.ndarray, w_cmd: np.ndarray, p: QuadParams,
          ext_force: np.ndarray | None = None) -> np.ndarray:
    v, q, omega, w = x[V], x[Q], x[W], x[M]

    f = motor_thrusts(x, p)                # 各电机推力
    wrench = p.alloc @ f                   # [总推力, τx, τy, τz]
    thrust, tau = wrench[0], wrench[1:]

    R = quat_to_rot(q)
    J = p.J

    dx = np.empty(NX)
    dx[P] = v
    force = R[:, 2] * thrust - p.drag * v
    if ext_force is not None:
        force = force + ext_force          # 外力(如风),世界系
    dx[V] = force / p.mass + np.array([0.0, 0.0, -p.g])
    dx[Q] = 0.5 * quat_mul(q, np.array([0.0, *omega]))
    dx[W] = np.linalg.solve(J, tau - np.cross(omega, J @ omega))
    dx[M] = (w_cmd - w) / p.motor_tau
    return dx


def on_ground(x: np.ndarray) -> bool:
    return bool(x[P][2] <= 0.0 and x[V][2] <= 0.0)


def step(x: np.ndarray, w_cmd: np.ndarray, p: QuadParams, dt: float,
         ground: bool = False, ext_force: np.ndarray | None = None) -> np.ndarray:
    """推进一个步长,步长内转速指令保持不变。

    ground=True 时 z = 0 处有地面。地面模型很粗糙:只要机体碰到地面且没有向上的
    速度,就把它"粘"在地上(速度和角速度清零),直到推力足以把它抬起来。
    没有弹跳、摩擦和翻倒。
    """
    w_cmd = np.clip(w_cmd, 0.0, p.w_max)
    k1 = deriv(x, w_cmd, p, ext_force)
    k2 = deriv(x + 0.5 * dt * k1, w_cmd, p, ext_force)
    k3 = deriv(x + 0.5 * dt * k2, w_cmd, p, ext_force)
    k4 = deriv(x + dt * k3, w_cmd, p, ext_force)
    xn = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    xn[Q] /= np.linalg.norm(xn[Q])
    if ground and on_ground(xn):
        xn[P][2] = 0.0
        xn[V] = 0.0
        xn[W] = 0.0
    return xn
