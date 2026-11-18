"""误差状态卡尔曼滤波器 (ESKF):融合 IMU、ToF 和光流。

名义状态:位置 p、速度 v(世界系)、四元数 q、陀螺零偏 bg、加速度计零偏 ba。
误差状态 (15 维):δp, δv, δθ, δbg, δba。
姿态误差 δθ 定义在机体系:q_真 = q ⊗ exp(δθ)。

流程:每来一帧 IMU 做一次 predict(积分名义状态、传播协方差),
ToF / 光流到达时做 update(修正名义状态、收缩协方差)。

这组传感器观测不到的量:水平位置的绝对值和偏航角(都只能靠积分,会慢慢漂)。
"""
from dataclasses import dataclass

import numpy as np

from .dynamics import NX, P, Q, V, W, quat_mul, quat_to_rot
from .params import QuadParams

# 误差状态的索引
EP, EV, ETH, EBG, EBA = (slice(0, 3), slice(3, 6), slice(6, 9),
                         slice(9, 12), slice(12, 15))
NE = 15


@dataclass
class EstimatorConfig:
    # 滤波器"认为"的噪声水平,默认与 SensorConfig 一致
    gyro_std: float = 0.02
    acc_std: float = 0.5
    gyro_bias_walk: float = 1e-4
    acc_bias_walk: float = 1e-3
    tof_std: float = 0.01
    flow_std: float = 0.3
    flow_min_height: float = 0.1
    min_cos_tilt: float = 0.5
    # 初始不确定度 (1σ)
    init_pos_std: float = 0.01
    init_vel_std: float = 0.05
    init_tilt_std: float = 0.05      # rad
    init_gyro_bias_std: float = 0.02
    init_acc_bias_std: float = 0.1


def skew(a: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def quat_from_rotvec(r: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(r)
    if angle < 1e-9:
        q = np.array([1.0, *(0.5 * r)])
    else:
        q = np.array([np.cos(angle / 2), *(np.sin(angle / 2) * r / angle)])
    return q / np.linalg.norm(q)


class ESKF:
    def __init__(self, params: QuadParams, cfg: EstimatorConfig | None = None,
                 pos0=(0.0, 0.0, 0.0)):
        self.params = params
        self.cfg = c = cfg or EstimatorConfig()
        self.p = np.array(pos0, dtype=float)
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.gyro = np.zeros(3)          # 最近一帧陀螺原始读数
        self.Pm = np.diag(np.repeat(
            [c.init_pos_std, c.init_vel_std, c.init_tilt_std,
             c.init_gyro_bias_std, c.init_acc_bias_std], 3) ** 2)
        self.last_Phi = np.eye(NE)

    # ---------- 上电校准 ----------
    def calibrate_gyro(self, samples: np.ndarray, max_std: float = 0.1) -> bool:
        """用静止时采到的陀螺读数 (N, 3) 标定零偏。

        静止时角速度为 0,读数的平均值就是零偏。如果读数抖动太大(说明机体在动),
        拒绝校准并返回 False。
        """
        samples = np.asarray(samples, dtype=float)
        if len(samples) < 50 or samples.std(axis=0).max() > max_std:
            return False
        self.bg = samples.mean(axis=0)
        std = max(self.cfg.gyro_std / np.sqrt(len(samples)), 1e-3)
        self.Pm[EBG, :] = 0.0
        self.Pm[:, EBG] = 0.0
        self.Pm[EBG, EBG] = np.eye(3) * std ** 2
        return True

    # ---------- 预测 ----------
    def predict(self, gyro: np.ndarray, acc: np.ndarray, dt: float):
        c = self.cfg
        self.gyro = np.asarray(gyro, dtype=float)
        w = self.gyro - self.bg
        a_b = acc - self.ba
        R = quat_to_rot(self.q)
        a_w = R @ a_b + np.array([0.0, 0.0, -self.params.g])

        self.p = self.p + self.v * dt + 0.5 * a_w * dt * dt
        self.v = self.v + a_w * dt
        self.q = quat_mul(self.q, quat_from_rotvec(w * dt))
        self.q /= np.linalg.norm(self.q)

        I3 = np.eye(3)
        Phi = np.eye(NE)
        Phi[EP, EV] = I3 * dt
        Phi[EV, ETH] = -R @ skew(a_b) * dt     # 姿态误差把推力方向带偏
        Phi[EV, EBA] = -R * dt
        Phi[ETH, ETH] = I3 - skew(w) * dt
        Phi[ETH, EBG] = -I3 * dt
        Qd = np.zeros(NE)
        Qd[EV] = (c.acc_std * dt) ** 2
        Qd[ETH] = (c.gyro_std * dt) ** 2
        Qd[EBG] = c.gyro_bias_walk ** 2 * dt
        Qd[EBA] = c.acc_bias_walk ** 2 * dt
        self.Pm = Phi @ self.Pm @ Phi.T + np.diag(Qd)
        self.last_Phi = Phi

    # ---------- 观测模型:返回 (预测值, 雅可比 H) ----------
    def h_tof(self):
        R = quat_to_rot(self.q)
        c22, pz = R[2, 2], self.p[2]
        H = np.zeros((1, NE))
        H[0, 2] = 1.0 / c22
        H[0, ETH] = -pz / c22 ** 2 * np.array([-R[2, 1], R[2, 0], 0.0])
        return np.array([pz / c22]), H

    def h_flow(self):
        R = quat_to_rot(self.q)
        c22, pz = R[2, 2], self.p[2]
        w = self.gyro - self.bg
        v_b = R.T @ self.v
        s = c22 / pz                                   # 1 / 到地面的距离
        pred = v_b[:2] * s + np.array([-w[1], w[0]])
        H = np.zeros((2, NE))
        H[:, EV] = R.T[:2, :] * s
        H[:, 2] = -v_b[:2] * c22 / pz ** 2
        H[:, ETH] = skew(v_b)[:2, :] * s + np.outer(v_b[:2] / pz,
                                                    [-R[2, 1], R[2, 0], 0.0])
        H[0, EBG.start + 1] = 1.0
        H[1, EBG.start + 0] = -1.0
        return pred, H

    # ---------- 更新 ----------
    def update_tof(self, d: float):
        if quat_to_rot(self.q)[2, 2] < self.cfg.min_cos_tilt:
            return
        pred, H = self.h_tof()
        self._update(np.array([d]) - pred, H, np.array([[self.cfg.tof_std ** 2]]))

    def update_flow(self, flow: np.ndarray):
        if (self.p[2] < self.cfg.flow_min_height
                or quat_to_rot(self.q)[2, 2] < self.cfg.min_cos_tilt):
            return
        pred, H = self.h_flow()
        self._update(flow - pred, H, np.eye(2) * self.cfg.flow_std ** 2)

    def _update(self, residual, H, Rm):
        S = H @ self.Pm @ H.T + Rm
        K = self.Pm @ H.T @ np.linalg.inv(S)
        IKH = np.eye(NE) - K @ H
        self.Pm = IKH @ self.Pm @ IKH.T + K @ Rm @ K.T     # Joseph 形式,保持对称正定
        self.inject(K @ residual)

    def inject(self, d: np.ndarray):
        """把误差状态加回名义状态。"""
        self.p = self.p + d[EP]
        self.v = self.v + d[EV]
        self.q = quat_mul(self.q, quat_from_rotvec(d[ETH]))
        self.q /= np.linalg.norm(self.q)
        self.bg = self.bg + d[EBG]
        self.ba = self.ba + d[EBA]

    # ---------- 输出 ----------
    def state_vector(self) -> np.ndarray:
        """按仿真状态向量的布局输出估计值,供控制器直接使用(电机转速项为 0)。"""
        x = np.zeros(NX)
        x[P], x[V], x[Q] = self.p, self.v, self.q
        x[W] = self.gyro - self.bg
        return x
