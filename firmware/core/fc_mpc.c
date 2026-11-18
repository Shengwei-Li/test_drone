#include "fc_mpc.h"

#include <string.h>

void fc_mpc_init(fc_mpc *m, const fc_mpc_tables *t, const fc_vehicle *veh, const fc_ctrl_gains *gains, fc_float dt) {
    memset(m, 0, sizeof(*m));
    m->t = *t;
    fc_ctrl_init(&m->inner, veh, gains, dt, 1);
}

/* 解一个轴:x0 是 [位置, 速度, 加速度],r 是未来 N+1 个时刻的参考位置。返回第一步的 jerk。 */
static fc_float solve_axis(fc_mpc *m, int ax, const fc_float x0[3], const fc_float r[FC_MPC_N + 1]) {
    const fc_mpc_tables *t = &m->t;
    fc_float (*x)[3] = m->x[ax], (*v)[3] = m->v[ax], (*g)[3] = m->g[ax];
    fc_float *u = m->u[ax], *d = m->d[ax], *z = m->z[ax], *y = m->y[ax];
    int it, k, i, j;

    for (it = 0; it < t->iters; it++) {
        /* 反向:只更新代价的线性项 p */
        fc_float p[3], pn[3];
        for (i = 0; i < 3; i++) p[i] = -t->P[i][0] * r[FC_MPC_N];
        for (k = FC_MPC_N - 1; k >= 0; k--) {
            fc_float rr = -t->rho_u * (z[k] - y[k]);
            d[k] = t->C1[k] * (t->B[0] * p[0] + t->B[1] * p[1] + t->B[2] * p[2] + rr);
            for (i = 0; i < 3; i++) {
                fc_float q = -t->rho_x[i] * (v[k][i] - g[k][i]);
                if (i == 0) q -= t->Qd[0] * r[k];
                pn[i] = q + t->AmBKt[k][i][0] * p[0] + t->AmBKt[k][i][1] * p[1] + t->AmBKt[k][i][2] * p[2]
                        - t->K[k][i] * rr;
            }
            for (i = 0; i < 3; i++) p[i] = pn[i];
        }
        /* 正向:推出状态和输入 */
        for (i = 0; i < 3; i++) x[0][i] = x0[i];
        for (k = 0; k < FC_MPC_N; k++) {
            u[k] = -(t->K[k][0] * x[k][0] + t->K[k][1] * x[k][1] + t->K[k][2] * x[k][2]) - d[k];
            for (i = 0; i < 3; i++) {
                fc_float s = t->B[i] * u[k];
                for (j = 0; j < 3; j++) s += t->A[i][j] * x[k][j];
                x[k + 1][i] = s;
            }
        }
        /* 裁剪到约束内,更新对偶变量 */
        for (k = 0; k < FC_MPC_N; k++) {
            z[k] = fc_clamp(u[k] + y[k], -t->u_max, t->u_max);
            y[k] += u[k] - z[k];
        }
        for (k = 1; k < FC_MPC_N; k++)
            for (i = 1; i < 3; i++) {                        /* 位置不设限,跳过 */
                v[k][i] = fc_clamp(x[k][i] + g[k][i], -t->x_max[ax][i], t->x_max[ax][i]);
                g[k][i] += x[k][i] - v[k][i];
            }
    }
    return z[0];
}

void fc_mpc_update(fc_mpc *m, const fc_float pos[3], const fc_float vel[3], const fc_float quat[4],
                   const fc_float omega[3], const fc_float *preview, fc_float yaw, const fc_float *acc,
                   fc_float w_cmd[4]) {
    const fc_mpc_tables *t = &m->t;
    fc_reference ref;
    fc_float s, dt = m->inner.dt;
    int j = m->k % t->solve_every, ax, k;

    if (j == 0) {
        for (ax = 0; ax < 3; ax++) {
            fc_float r[FC_MPC_N + 1], x0[3];
            for (k = 0; k <= FC_MPC_N; k++) r[k] = preview[3 * k + ax];
            m->p0[ax] = pos[ax]; m->v0[ax] = vel[ax]; m->a0[ax] = m->acc_plan[ax];
            x0[0] = m->p0[ax]; x0[1] = m->v0[ax]; x0[2] = m->a0[ax];
            m->j0[ax] = solve_axis(m, ax, x0, r);
            m->acc_plan[ax] = m->a0[ax] + m->j0[ax] * (fc_float)t->solve_every * dt;
        }
    }
    m->k++;

    /* 两次求解之间,沿规划的第一段(恒定 jerk)插值,作为内环的参考 */
    s = (fc_float)j * dt;
    for (ax = 0; ax < 3; ax++) {
        ref.pos[ax] = m->p0[ax] + m->v0[ax] * s + (fc_float)0.5 * m->a0[ax] * s * s + m->j0[ax] * s * s * s / 6;
        ref.vel[ax] = m->v0[ax] + m->a0[ax] * s + (fc_float)0.5 * m->j0[ax] * s * s;
        ref.acc[ax] = m->a0[ax] + m->j0[ax] * s;
        ref.jerk[ax] = m->j0[ax];
    }
    ref.yaw = yaw;
    fc_ctrl_update(&m->inner, pos, vel, quat, omega, &ref, acc, w_cmd);
}
