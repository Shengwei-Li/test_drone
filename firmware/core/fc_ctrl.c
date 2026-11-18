#include "fc_ctrl.h"

void fc_ctrl_default_gains(fc_ctrl_gains *k) {
    k->kx[0] = 6;   k->kx[1] = 6;   k->kx[2] = 10;
    k->kv[0] = 4;   k->kv[1] = 4;   k->kv[2] = 5;
    k->kR[0] = 110; k->kR[1] = 110; k->kR[2] = 30;
    k->kW[0] = 18;  k->kW[1] = 18;  k->kW[2] = 10;
    k->a_xy_max = 6; k->a_z_max = 6;
    k->fc_rate = 30; k->fc_force = 2;
}

void fc_ctrl_init(fc_ctrl *c, const fc_vehicle *veh, const fc_ctrl_gains *k, fc_float dt, int use_indi) {
    int i;
    c->veh = *veh;
    c->k = *k;
    c->dt = dt;
    c->use_indi = use_indi;
    c->a_rate = dt / (dt + 1 / (2 * FC_PI * k->fc_rate));
    c->a_force = dt / (dt + 1 / (2 * FC_PI * k->fc_force));
    c->a_motor = 1 - FC_EXP(-dt / veh->motor_tau);
    for (i = 0; i < 4; i++) c->w_hat[i] = 0;
    for (i = 0; i < 3; i++) { c->omega_f[i] = 0; c->tau_f[i] = 0; c->d_hat[i] = 0; }
    c->omega_f_valid = 0;
}

void fc_mix(const fc_vehicle *veh, fc_float thrust, const fc_float tau[3], fc_float w_cmd[4]) {
    fc_float u[4];
    int i, j;
    u[0] = thrust; u[1] = tau[0]; u[2] = tau[1]; u[3] = tau[2];
    for (i = 0; i < 4; i++) {
        fc_float f = 0;
        for (j = 0; j < 4; j++) f += veh->alloc_inv[i][j] * u[j];
        f = fc_clamp(f, 0, veh->f_max);
        w_cmd[i] = FC_SQRT(f / veh->kf);
    }
}

void fc_ctrl_update(fc_ctrl *c, const fc_float pos[3], const fc_float vel[3], const fc_float quat[4],
                    const fc_float omega[3], const fc_reference *ref, const fc_float *acc, fc_float w_cmd[4]) {
    const fc_vehicle *v = &c->veh;
    const fc_ctrl_gains *k = &c->k;
    fc_float R[3][3], a[3], f_des[3], zb[3], xc[3], yb[3], xb[3], Rd[3][3];
    fc_float h[3], omega_d[3], e_R[3], e_W[3], alpha[3], tau[3], M[3][3], rd_w[3];
    fc_float n_xy, f_norm, thrust, n, zj;
    int i, j;

    fc_quat_to_rot(quat, R);

    /* 位置环:期望加速度 = 反馈 + 前馈,然后限幅 */
    for (i = 0; i < 3; i++)
        a[i] = k->kx[i] * (ref->pos[i] - pos[i]) + k->kv[i] * (ref->vel[i] - vel[i]) + ref->acc[i];
    n_xy = FC_SQRT(a[0] * a[0] + a[1] * a[1]);
    if (n_xy > k->a_xy_max) { a[0] *= k->a_xy_max / n_xy; a[1] *= k->a_xy_max / n_xy; }
    a[2] = fc_clamp(a[2], -k->a_z_max, k->a_z_max);

    /* INDI 外力观测:加速度计量到的实际比力 − 模型推力 = 外力。离地后才更新,避免把地面支持力当成扰动 */
    if (c->use_indi && acc && pos[2] > (fc_float)0.05) {
        fc_float t_model = 0;
        for (i = 0; i < 4; i++) t_model += v->kf * c->w_hat[i] * c->w_hat[i];
        for (i = 0; i < 3; i++) {
            fc_float d_raw = v->mass * (R[i][0] * acc[0] + R[i][1] * acc[1] + R[i][2] * acc[2]) - t_model * R[i][2];
            c->d_hat[i] += c->a_force * (d_raw - c->d_hat[i]);
        }
    }

    /* 期望推力向量 → 总推力(投影到当前机体 z 轴) */
    for (i = 0; i < 3; i++) f_des[i] = v->mass * a[i] - (c->use_indi ? c->d_hat[i] : 0);
    f_des[2] += v->mass * v->g;
    thrust = fc_clamp(f_des[0] * R[0][2] + f_des[1] * R[1][2] + f_des[2] * R[2][2], 0, 4 * v->f_max);

    /* 期望姿态:机体 z 轴对准期望推力方向,再由偏航角定出 x、y 轴 */
    f_norm = fc_norm3(f_des);
    if (f_norm < (fc_float)1e-6) f_norm = (fc_float)1e-6;
    for (i = 0; i < 3; i++) zb[i] = f_des[i] / f_norm;
    xc[0] = FC_COS(ref->yaw); xc[1] = FC_SIN(ref->yaw); xc[2] = 0;
    fc_cross3(zb, xc, yb);
    n = fc_norm3(yb);
    if (n < (fc_float)1e-9) n = (fc_float)1e-9;
    for (i = 0; i < 3; i++) yb[i] /= n;
    fc_cross3(yb, zb, xb);
    for (i = 0; i < 3; i++) { Rd[i][0] = xb[i]; Rd[i][1] = yb[i]; Rd[i][2] = zb[i]; }

    /* 微分平坦:推力方向的变化率由参考 jerk 决定 → 期望角速度(偏航率取 0) */
    zj = fc_dot3(zb, ref->jerk);
    for (i = 0; i < 3; i++) h[i] = v->mass / f_norm * (ref->jerk[i] - zj * zb[i]);
    omega_d[0] = -fc_dot3(h, yb); omega_d[1] = fc_dot3(h, xb); omega_d[2] = 0;

    /* 姿态误差 e_R = ½ vee(Rdᵀ R − Rᵀ Rd),角速度误差 e_W = ω − Rᵀ Rd ω_d */
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            M[i][j] = Rd[0][i] * R[0][j] + Rd[1][i] * R[1][j] + Rd[2][i] * R[2][j];      /* Rdᵀ R */
    e_R[0] = (fc_float)0.5 * (M[2][1] - M[1][2]);
    e_R[1] = (fc_float)0.5 * (M[0][2] - M[2][0]);
    e_R[2] = (fc_float)0.5 * (M[1][0] - M[0][1]);
    for (i = 0; i < 3; i++) rd_w[i] = Rd[i][0] * omega_d[0] + Rd[i][1] * omega_d[1] + Rd[i][2] * omega_d[2];
    for (i = 0; i < 3; i++) {
        e_W[i] = omega[i] - (R[0][i] * rd_w[0] + R[1][i] * rd_w[1] + R[2][i] * rd_w[2]);
        alpha[i] = -k->kR[i] * e_R[i] - k->kW[i] * e_W[i];
    }

    if (c->use_indi) {
        /* 力矩增量:τ = τ_当前 + J (α_期望 − α_测量)。α_测量 是滤波后角速度的差分,
           τ_当前 由电机转速模型算出并用同样的滤波器,保持两者同步。 */
        fc_float prev[3], f[4];
        if (!c->omega_f_valid) { for (i = 0; i < 3; i++) c->omega_f[i] = omega[i]; c->omega_f_valid = 1; }
        for (i = 0; i < 4; i++) f[i] = v->kf * c->w_hat[i] * c->w_hat[i];
        for (i = 0; i < 3; i++) {
            fc_float alpha_f, tau_model = 0;
            prev[i] = c->omega_f[i];
            c->omega_f[i] = prev[i] + c->a_rate * (omega[i] - prev[i]);
            alpha_f = (c->omega_f[i] - prev[i]) / c->dt;
            for (j = 0; j < 4; j++) tau_model += v->alloc[1 + i][j] * f[j];
            c->tau_f[i] += c->a_rate * (tau_model - c->tau_f[i]);
            tau[i] = c->tau_f[i] + v->J[i] * (alpha[i] - alpha_f);
        }
    } else {
        /* 基于模型的求逆:τ = J α + ω × Jω */
        fc_float Jw[3], gyro_t[3];
        for (i = 0; i < 3; i++) Jw[i] = v->J[i] * omega[i];
        fc_cross3(omega, Jw, gyro_t);
        for (i = 0; i < 3; i++) tau[i] = v->J[i] * alpha[i] + gyro_t[i];
    }

    fc_mix(v, thrust, tau, w_cmd);
    if (c->use_indi)
        for (i = 0; i < 4; i++) c->w_hat[i] += c->a_motor * (w_cmd[i] - c->w_hat[i]);
}
