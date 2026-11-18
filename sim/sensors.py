"""传感器模型:IMU、下视 ToF 测距、下视光流。

噪声默认值按"飞行中带振动"的量级取,偏保守,不是芯片手册上的静态噪声。
地面假设为 z = 0 的平面。
"""
from dataclasses import dataclass

import numpy as np

from .dynamics import P, Q, V, W, motor_thrusts, on_ground, quat_to_rot
from .params import QuadParams


@dataclass
class SensorConfig:
    gyro_std: float = 0.02          # rad/s,每个采样的白噪声
    acc_std: float = 0.5            # m/s²,每个采样的白噪声
    gyro_bias_std: float = 0.02     # rad/s,上电零偏的分布
    acc_bias_std: float = 0.1       # m/s²
    gyro_bias_walk: float = 1e-4    # rad/s/√s,零偏随机游走
    acc_bias_walk: float = 1e-3     # m/s²/√s
    tof_std: float = 0.01           # m
    tof_range: tuple = (0.02, 4.0)  # m,有效量程
    flow_std: float = 0.3           # rad/s
    flow_min_dist: float = 0.08     # m,光流最小工作距离
    min_cos_tilt: float = 0.5       # 倾角超过 60° 时下视传感器无效

    @classmethod
    def ideal(cls) -> "SensorConfig":
        """无噪声、无零偏,用于测试。"""
        return cls(gyro_std=0, acc_std=0, gyro_bias_std=0, acc_bias_std=0,
                   gyro_bias_walk=0, acc_bias_walk=0, tof_std=0, flow_std=0)


def specific_force(x: np.ndarray, p: QuadParams,
                   ext_force: np.ndarray | None = None) -> np.ndarray:
    """理想加速度计读数(机体系比力):除重力以外的合力 / 质量。

    悬停时约为 [0, 0, +g];静止在地面上时也是(地面支持力代替了推力)。
    """
    R = quat_to_rot(x[Q])
    if on_ground(x) and not np.any(x[V]):
        return R.T @ np.array([0.0, 0.0, p.g])
    force_body = np.array([0.0, 0.0, motor_thrusts(x, p).sum()]) - p.drag * (R.T @ x[V])
    if ext_force is not None:
        force_body = force_body + R.T @ ext_force
    return force_body / p.mass


class SensorSuite:
    def __init__(self, params: QuadParams, cfg: SensorConfig | None = None,
                 rng: np.random.Generator | None = None):
        self.p = params
        self.cfg = cfg or SensorConfig()
        self.rng = rng or np.random.default_rng(0)
        self.gyro_bias = self.rng.normal(0, 1, 3) * self.cfg.gyro_bias_std
        self.acc_bias = self.rng.normal(0, 1, 3) * self.cfg.acc_bias_std

    def imu(self, x: np.ndarray, dt: float, ext_force: np.ndarray | None = None):
        """返回 (陀螺 rad/s, 加速度计 m/s²),均在机体系。dt 为采样间隔。"""
        c = self.cfg
        f_body = specific_force(x, self.p, ext_force)
        self.gyro_bias += self.rng.normal(0, 1, 3) * c.gyro_bias_walk * np.sqrt(dt)
        self.acc_bias += self.rng.normal(0, 1, 3) * c.acc_bias_walk * np.sqrt(dt)
        gyro = x[W] + self.gyro_bias + self.rng.normal(0, 1, 3) * c.gyro_std
        acc = f_body + self.acc_bias + self.rng.normal(0, 1, 3) * c.acc_std
        return gyro, acc

    def _ground_distance(self, x: np.ndarray):
        """沿机体 -z 轴到地面的距离;姿态太斜时返回 None。"""
        cos_tilt = quat_to_rot(x[Q])[2, 2]
        if cos_tilt < self.cfg.min_cos_tilt:
            return None
        return x[P][2] / cos_tilt

    def tof(self, x: np.ndarray):
        """ToF 测距 (m),超出量程返回 None。"""
        d = self._ground_distance(x)
        lo, hi = self.cfg.tof_range
        if d is None or not (lo <= d <= hi):
            return None
        return d + self.rng.normal() * self.cfg.tof_std

    def flow(self, x: np.ndarray):
        """光流 (rad/s, 2 维):[v_bx/d - ω_y, v_by/d + ω_x]。

        v_b 是机体系速度,d 是到地面的距离。机体自身转动也会产生光流,
        所以估计器里要用陀螺把后一项补偿掉。
        """
        d = self._ground_distance(x)
        if d is None or d < self.cfg.flow_min_dist:
            return None
        v_b = quat_to_rot(x[Q]).T @ x[V]
        w = x[W]
        ideal = np.array([v_b[0] / d - w[1], v_b[1] / d + w[0]])
        return ideal + self.rng.normal(0, 1, 2) * self.cfg.flow_std
