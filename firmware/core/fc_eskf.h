/* 15 维误差状态卡尔曼滤波器:IMU 预测,测距 (ToF) 和光流更新。
 * 与 sim/estimator.py 一一对应,公式和变量名保持一致。
 * 误差状态顺序:δp(0-2) δv(3-5) δθ(6-8,机体系) δbg(9-11) δba(12-14)。
 */
#ifndef FC_ESKF_H
#define FC_ESKF_H

#include "fc_math.h"

#define FC_NE 15

typedef struct {
    fc_float gyro_std, acc_std, gyro_bias_walk, acc_bias_walk;
    fc_float tof_std, flow_std, flow_min_height, min_cos_tilt;
    fc_float init_pos_std, init_vel_std, init_tilt_std, init_gyro_bias_std, init_acc_bias_std;
} fc_eskf_cfg;

typedef struct {
    fc_float p[3], v[3], q[4], bg[3], ba[3];
    fc_float gyro[3];                 /* 最近一帧陀螺原始读数 */
    fc_float P[FC_NE][FC_NE];         /* 误差协方差 */
    fc_float g;
    fc_eskf_cfg cfg;
} fc_eskf;

void fc_eskf_init(fc_eskf *e, const fc_eskf_cfg *cfg, fc_float g, const fc_float pos0[3]);
/* 用静止时的 n 帧陀螺读数标定零偏;抖动过大或样本太少时返回 0 且不改状态 */
int fc_eskf_calibrate_gyro(fc_eskf *e, const fc_float *samples, int n, fc_float max_std);
/* 直接写入零偏(由 n 帧读数平均得到),并相应收紧零偏的不确定度 */
void fc_eskf_set_gyro_bias(fc_eskf *e, const fc_float mean[3], int n);
void fc_eskf_predict(fc_eskf *e, const fc_float gyro[3], const fc_float acc[3], fc_float dt);
void fc_eskf_update_tof(fc_eskf *e, fc_float dist);
void fc_eskf_update_flow(fc_eskf *e, const fc_float flow[2]);
/* 去掉零偏后的角速度估计 */
void fc_eskf_omega(const fc_eskf *e, fc_float omega[3]);

#endif
