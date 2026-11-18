"""机体参数。

默认值取自开源仿真里常用的 Crazyflie 2.x(X 型)数值,只是占位:
等自己的机子做完系统辨识后,换成实测值。
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuadParams:
    mass: float = 0.027                 # kg
    arm: float = 0.0397                 # m,机体中心到电机的距离
    inertia: tuple = (1.4e-5, 1.4e-5, 2.17e-5)  # kg·m²,主轴转动惯量
    kf: float = 2.88e-8                 # N/(rad/s)²,推力系数
    km: float = 7.24e-10                # N·m/(rad/s)²,反扭矩系数
    motor_tau: float = 0.05             # s,电机转速一阶响应时间常数
    thrust_to_weight: float = 2.25      # 最大总推力 / 重力
    drag: float = 0.01                  # N/(m/s),线性气动阻力
    g: float = 9.81                     # m/s²
    # 各电机的推力倍率。只用来在"被控对象"上制造与模型不一致的情况
    # (比如某个电机偏弱);控制器手里的模型始终认为是 1。
    motor_gain: tuple = (1.0, 1.0, 1.0, 1.0)

    @property
    def J(self) -> np.ndarray:
        return np.diag(self.inertia)

    @property
    def f_max(self) -> float:
        """单个电机的最大推力 (N)。"""
        return self.thrust_to_weight * self.mass * self.g / 4.0

    @property
    def w_max(self) -> float:
        """电机最大转速 (rad/s)。"""
        return float(np.sqrt(self.f_max / self.kf))

    @property
    def w_hover(self) -> float:
        """悬停转速 (rad/s)。"""
        return float(np.sqrt(self.mass * self.g / 4.0 / self.kf))

    @property
    def motor_xy(self) -> np.ndarray:
        """四个电机在机体系中的 (x, y) 位置,形状 (4, 2)。

        M1 右前, M2 右后, M3 左后, M4 左前。
        """
        d = self.arm / np.sqrt(2.0)
        return np.array([[d, -d], [-d, -d], [-d, d], [d, d]])

    @property
    def motor_dir(self) -> np.ndarray:
        """各电机反扭矩在机体 z 轴上的符号(对角电机同向)。"""
        return np.array([-1.0, 1.0, -1.0, 1.0])

    @property
    def alloc(self) -> np.ndarray:
        """控制分配矩阵 A:[总推力, τx, τy, τz] = A @ [f1..f4]。"""
        x, y = self.motor_xy[:, 0], self.motor_xy[:, 1]
        return np.vstack([
            np.ones(4),
            y,                               # τx = Σ y_i f_i
            -x,                              # τy = -Σ x_i f_i
            self.motor_dir * self.km / self.kf,
        ])
