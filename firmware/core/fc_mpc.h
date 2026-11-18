/* 位置层的线性 MPC(ADMM + 预先算好的增益表),内环用 fc_ctrl 的 INDI。
 * 与 sim/mpc.py + sim/controller.py 的 MpcController 对应。
 *
 * 三个轴的模型相同(三重积分器:位置、速度、加速度,输入是 jerk),互不耦合,
 * 所以这里每个轴单独解一个 3 状态的小问题,共用同一套增益表;
 * Python 版把三个轴拼成 9 状态一起解,数学上等价。
 * 增益表 K、C1、AmBKt、P 由 Python 离线算好(firmware/sil.py 的 mpc_tables),上板时生成成常量数组。
 */
#ifndef FC_MPC_H
#define FC_MPC_H

#include "fc_ctrl.h"

#define FC_MPC_N 25

typedef struct {
    fc_float A[3][3], B[3], Qd[3], P[3][3];
    fc_float K[FC_MPC_N][3], C1[FC_MPC_N], AmBKt[FC_MPC_N][3][3];
    fc_float rho_u, rho_x[3];
    fc_float u_max;                   /* jerk 上限 */
    fc_float x_max[3][3];             /* [轴][状态] 的上限;位置不设限(填一个很大的数) */
    fc_float dt_mpc;
    int iters, solve_every;
} fc_mpc_tables;

typedef struct {
    fc_mpc_tables t;
    fc_ctrl inner;
    /* ADMM 的变量,保留到下次求解作热启动。[轴][时刻] */
    fc_float x[3][FC_MPC_N + 1][3], u[3][FC_MPC_N], d[3][FC_MPC_N];
    fc_float z[3][FC_MPC_N], y[3][FC_MPC_N];
    fc_float v[3][FC_MPC_N + 1][3], g[3][FC_MPC_N + 1][3];
    int k;
    fc_float acc_plan[3], p0[3], v0[3], a0[3], j0[3];
} fc_mpc;

void fc_mpc_init(fc_mpc *m, const fc_mpc_tables *t, const fc_vehicle *veh, const fc_ctrl_gains *gains, fc_float dt);
/* preview:未来 FC_MPC_N+1 个时刻的参考位置(间隔 dt_mpc),按 [时刻][xyz] 排列。
   只在重新规划的那一拍(每 solve_every 拍一次)会被读取。 */
void fc_mpc_update(fc_mpc *m, const fc_float pos[3], const fc_float vel[3], const fc_float quat[4],
                   const fc_float omega[3], const fc_float *preview, fc_float yaw, const fc_float *acc,
                   fc_float w_cmd[4]);
static inline int fc_mpc_needs_preview(const fc_mpc *m) { return m->k % m->t.solve_every == 0; }

#endif
