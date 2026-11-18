/* 强化学习策略的推理:28 → 64 → 64 → 4 的全连接网络,tanh 激活。
 * 与 sim/rl_policy.py 一一对应(观测的组成、缩放系数、积分项、动作到转速的换算)。
 * 每 FC_POLICY_DT 秒调用一次(100 Hz)。
 */
#ifndef FC_POLICY_H
#define FC_POLICY_H

#include "fc_math.h"

#define FC_OBS 28
#define FC_HID 64
#define FC_ACT 4
#define FC_POLICY_DT ((fc_float)0.01)

typedef struct {
    fc_float W0[FC_HID][FC_OBS], b0[FC_HID];
    fc_float W1[FC_HID][FC_HID], b1[FC_HID];
    fc_float W2[FC_ACT][FC_HID], b2[FC_ACT];
    fc_float w_hover, w_max;
    fc_float act[FC_ACT];             /* 上一次的动作 */
    fc_float integ[3];                /* 位置误差积分 */
} fc_policy;

void fc_policy_reset(fc_policy *p, fc_float w_hover, fc_float w_max);
void fc_policy_update(fc_policy *p, const fc_float pos[3], const fc_float vel[3], const fc_float quat[4],
                      const fc_float omega[3], const fc_float ref_pos[3], const fc_float ref_vel[3],
                      const fc_float ref_acc[3], fc_float w_cmd[4]);

#endif
