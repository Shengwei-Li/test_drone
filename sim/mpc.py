"""线性 MPC 求解器:ADMM + 预先算好的 Riccati 增益(TinyMPC 的做法)。

问题形式(k = 0..N-1,终端代价用无限时域的 P):

    min  Σ ½(x_k − r_k)ᵀQ(x_k − r_k) + ½u_kᵀR u_k
    s.t. x_{k+1} = A x_k + B u_k
         u_min ≤ u_k ≤ u_max,   x_min ≤ x_k ≤ x_max   (k = 1..N-1)

做法:把约束挪到"副本变量"上(u = z,x = v),交替做三件事:
  1. 不管约束,解一个带线性项的 LQR(一次反向递推 + 一次正向递推);
  2. 把结果裁剪到约束范围内,得到 z、v;
  3. 用对偶变量 y、g 累计"LQR 解"和"裁剪后的解"的差,下一轮把 LQR 往可行方向推。

因为 A、B、Q、R 不变,各时刻的增益 K 只在初始化时算一次,在线部分只有矩阵乘向量,
没有矩阵求逆 —— 这是它能在单片机上跑的原因。这个文件故意只用最基本的运算,
以后可以逐行翻译成 C。
"""
import numpy as np


class AdmmMpc:
    def __init__(self, A, B, Q, R, horizon: int, u_min, u_max, x_min, x_max,
                 rho_u: float = 1.0, rho_x: float = 1.0, iters: int = 10):
        self.A, self.B, self.Q = A, B, Q
        self.N, self.iters = horizon, iters
        n, m = B.shape
        self.n, self.m = n, m
        self.u_min, self.u_max = np.asarray(u_min, float), np.asarray(u_max, float)
        self.x_min, self.x_max = np.asarray(x_min, float), np.asarray(x_max, float)
        # 没有约束的分量不加惩罚(ρ = 0),否则它们会被无谓地拖慢收敛
        self.rho_u = rho_u * (np.isfinite(self.u_min) | np.isfinite(self.u_max))
        self.rho_x = rho_x * (np.isfinite(self.x_min) | np.isfinite(self.x_max))

        # 离线第一步:原始 Q、R 的无限时域 LQR,它的 P 作为终端代价
        P = Q.copy()
        for _ in range(5000):
            K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
            P_new = Q + A.T @ P @ (A - B @ K)
            if np.max(np.abs(P_new - P)) < 1e-9 * max(1.0, np.max(np.abs(P))):
                P = P_new
                break
            P = P_new
        self.P = 0.5 * (P + P.T)
        self.K_lqr = np.linalg.solve(R + B.T @ self.P @ B, B.T @ self.P @ A)

        # 离线第二步:加上 ADMM 惩罚项后的有限时域增益,每个时刻一组(共 N 组,在线只查表)
        Qt = Q + np.diag(self.rho_x)
        Rt = R + np.diag(self.rho_u)
        self.K = np.zeros((horizon, m, n))
        self.C1 = np.zeros((horizon, m, m))
        self.AmBKt = np.zeros((horizon, n, n))
        Pk = self.P
        for k in range(horizon - 1, -1, -1):
            self.C1[k] = np.linalg.inv(Rt + B.T @ Pk @ B)
            self.K[k] = self.C1[k] @ B.T @ Pk @ A
            self.AmBKt[k] = (A - B @ self.K[k]).T
            Pk = Qt + A.T @ Pk @ (A - B @ self.K[k])
            Pk = 0.5 * (Pk + Pk.T)
        self.reset()

    def reset(self):
        N, n, m = self.N, self.n, self.m
        self.x = np.zeros((N + 1, n))
        self.u = np.zeros((N, m))
        self.z = np.zeros((N, m))      # 输入的副本(满足约束)
        self.y = np.zeros((N, m))      # 输入的对偶变量
        self.v = np.zeros((N + 1, n))  # 状态的副本(满足约束)
        self.g = np.zeros((N + 1, n))  # 状态的对偶变量
        self.d = np.zeros((N, m))

    def solve(self, x0, x_ref, iters: int | None = None):
        """x_ref 形状 (N+1, n)。返回 (第一步输入, 规划的状态序列)。

        z、v、y、g 保留到下次调用作为热启动,所以每次只需要迭代几轮。
        """
        A, B, K, C1, AmBKt = self.A, self.B, self.K, self.C1, self.AmBKt
        N = self.N
        qref = -(x_ref @ self.Q)                  # Q 对称
        p_term = -(self.P @ x_ref[N])
        for _ in range(iters or self.iters):
            # 反向:只更新代价的线性项
            p = p_term
            for k in range(N - 1, -1, -1):
                r = -self.rho_u * (self.z[k] - self.y[k])
                self.d[k] = C1[k] @ (B.T @ p + r)
                q = qref[k] - self.rho_x * (self.v[k] - self.g[k])
                p = q + AmBKt[k] @ p - K[k].T @ r
            # 正向:推出状态和输入
            self.x[0] = x0
            for k in range(N):
                self.u[k] = -K[k] @ self.x[k] - self.d[k]
                self.x[k + 1] = A @ self.x[k] + B @ self.u[k]
            # 裁剪到约束内,更新对偶变量
            self.z = np.clip(self.u + self.y, self.u_min, self.u_max)
            self.y += self.u - self.z
            self.v[1:N] = np.clip(self.x[1:N] + self.g[1:N], self.x_min, self.x_max)
            self.g[1:N] += self.x[1:N] - self.v[1:N]
        return self.z[0].copy(), self.x
