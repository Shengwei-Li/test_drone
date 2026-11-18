/* 位置/姿态控制器:SE(3) 几何控制 + INDI(增量非线性动态逆)。
 * 与 sim/controller.py 的 GeometricController / IndiController 一一对应。
 * 输入状态和参考轨迹,输出四个电机的转速指令 (rad/s)。
 */
#ifndef FC_CTRL_H
#define FC_CTRL_H

#include "fc_math.h"

typedef struct {
    fc_float mass, g, kf, f_max, motor_tau;
    fc_float J[3];                    /* 主轴转动惯量 */
    fc_float alloc[4][4];             /* [总推力, τx, τy, τz] = alloc · [f1..f4] */
    fc_float alloc_inv[4][4];
} fc_vehicle;

typedef struct {
    fc_float kx[3], kv[3], kR[3], kW[3];
    fc_float a_xy_max, a_z_max;
    fc_float fc_rate;                 /* 角速度/力矩低通截止频率 (Hz) */
    fc_float fc_force;                /* 外力观测器低通截止频率 (Hz) */
} fc_ctrl_gains;

typedef struct {
    fc_float pos[3], vel[3], acc[3], jerk[3], yaw;
} fc_reference;

typedef struct {
    fc_vehicle veh;
    fc_ctrl_gains k;
    fc_float dt;
    int use_indi;                     /* 0 = 纯几何控制(基于模型),1 = INDI */
    fc_float a_rate, a_force, a_motor;   /* 各低通的系数 */
    /* INDI 的内部状态 */
    fc_float w_hat[4];                /* 电机转速的模型估计 */
    fc_float omega_f[3];              /* 滤波后的角速度 */
    int omega_f_valid;
    fc_float tau_f[3];                /* 滤波后的模型力矩 */
    fc_float d_hat[3];                /* 外力估计(世界系) */
} fc_ctrl;

void fc_ctrl_default_gains(fc_ctrl_gains *k);
void fc_ctrl_init(fc_ctrl *c, const fc_vehicle *veh, const fc_ctrl_gains *k, fc_float dt, int use_indi);
/* acc:加速度计读数(去零偏后,机体系);没有时传 NULL,此时不更新外力估计 */
void fc_ctrl_update(fc_ctrl *c, const fc_float pos[3], const fc_float vel[3], const fc_float quat[4],
                    const fc_float omega[3], const fc_reference *ref, const fc_float *acc, fc_float w_cmd[4]);
/* 混控:总推力和三轴力矩 → 各电机转速,饱和时只做限幅 */
void fc_mix(const fc_vehicle *veh, fc_float thrust, const fc_float tau[3], fc_float w_cmd[4]);

#endif
