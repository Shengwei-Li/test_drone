#include "fc_policy.h"

#define S_POS ((fc_float)1.0)
#define S_VEL ((fc_float)0.5)
#define S_ACC ((fc_float)0.2)
#define S_OMEGA ((fc_float)0.2)
#define S_INT ((fc_float)4.0)
#define ACT_SCALE ((fc_float)0.5)
#define INT_RADIUS ((fc_float)0.3)
#define INT_CLIP ((fc_float)0.5)

void fc_policy_reset(fc_policy *p, fc_float w_hover, fc_float w_max) {
    int i;
    p->w_hover = w_hover;
    p->w_max = w_max;
    for (i = 0; i < FC_ACT; i++) p->act[i] = 0;
    for (i = 0; i < 3; i++) p->integ[i] = 0;
}

void fc_policy_update(fc_policy *p, const fc_float pos[3], const fc_float vel[3], const fc_float quat[4],
                      const fc_float omega[3], const fc_float ref_pos[3], const fc_float ref_vel[3],
                      const fc_float ref_acc[3], fc_float w_cmd[4]) {
    fc_float obs[FC_OBS], h0[FC_HID], h1[FC_HID], R[3][3], e_p[3];
    int i, j, n = 0;

    for (i = 0; i < 3; i++) e_p[i] = ref_pos[i] - pos[i];
    if (fc_norm3(e_p) < INT_RADIUS)                    /* 只在目标附近积分,避免远距离时积分饱和 */
        for (i = 0; i < 3; i++) p->integ[i] = fc_clamp(p->integ[i] + e_p[i] * FC_POLICY_DT, -INT_CLIP, INT_CLIP);

    fc_quat_to_rot(quat, R);
    for (i = 0; i < 3; i++) obs[n++] = fc_clamp(e_p[i] * S_POS, -2, 2);
    for (i = 0; i < 3; i++) obs[n++] = fc_clamp((ref_vel[i] - vel[i]) * S_VEL, -2, 2);
    for (i = 0; i < 3; i++) obs[n++] = fc_clamp(ref_acc[i] * S_ACC, -2, 2);
    for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) obs[n++] = R[i][j];
    for (i = 0; i < 3; i++) obs[n++] = fc_clamp(omega[i] * S_OMEGA, -3, 3);
    for (i = 0; i < FC_ACT; i++) obs[n++] = p->act[i];
    for (i = 0; i < 3; i++) obs[n++] = p->integ[i] * S_INT;

    for (i = 0; i < FC_HID; i++) {
        fc_float s = p->b0[i];
        for (j = 0; j < FC_OBS; j++) s += p->W0[i][j] * obs[j];
        h0[i] = FC_TANH(s);
    }
    for (i = 0; i < FC_HID; i++) {
        fc_float s = p->b1[i];
        for (j = 0; j < FC_HID; j++) s += p->W1[i][j] * h0[j];
        h1[i] = FC_TANH(s);
    }
    for (i = 0; i < FC_ACT; i++) {
        fc_float s = p->b2[i];
        for (j = 0; j < FC_HID; j++) s += p->W2[i][j] * h1[j];
        p->act[i] = fc_clamp(s, -1, 1);
        w_cmd[i] = fc_clamp(p->w_hover * (1 + ACT_SCALE * p->act[i]), 0, p->w_max);
    }
}
