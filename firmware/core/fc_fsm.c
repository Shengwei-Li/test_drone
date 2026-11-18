#include "fc_fsm.h"

void fc_fsm_default_cfg(fc_fsm_cfg *c) {
    c->link_timeout = (fc_float)0.5;
    c->v_low = (fc_float)3.3; c->v_low_time = (fc_float)1.0;
    c->flip_cos = (fc_float)0.17; c->flip_time = (fc_float)0.5;      /* 约 80°,持续 0.5 s */
    c->fence_xy = (fc_float)3.0; c->fence_z = (fc_float)3.0;
    c->land_speed = (fc_float)0.3;
    c->land_height = (fc_float)0.04; c->land_time = (fc_float)0.3;
    c->arm_min_cos = (fc_float)0.9;                                   /* 约 25° 以内 */
}

void fc_fsm_init(fc_fsm *f, const fc_fsm_cfg *c) {
    int i;
    f->cfg = *c;
    f->mode = FC_MODE_DISARMED;
    f->reason = FC_WHY_NONE;
    f->t_link = f->t_batt = f->t_flip = f->t_ground = 0;
    for (i = 0; i < 3; i++) f->land_pos[i] = 0;
}

static void go(fc_fsm *f, fc_mode m, fc_reason why) { f->mode = m; f->reason = why; }

static void start_landing(fc_fsm *f, const fc_fsm_in *in, fc_reason why) {
    int i;
    for (i = 0; i < 3; i++) f->land_pos[i] = in->pos[i];
    f->t_ground = 0;
    go(f, FC_MODE_LANDING, why);
}

void fc_fsm_step(fc_fsm *f, const fc_fsm_in *in, fc_float dt) {
    const fc_fsm_cfg *c = &f->cfg;
    int armed = fc_fsm_motors_on(f);

    /* 计时器:条件成立时累加,不成立时清零 */
    f->t_link = in->link_rx ? 0 : f->t_link + dt;
    f->t_batt = (in->vbat < c->v_low) ? f->t_batt + dt : 0;
    f->t_flip = (armed && in->cos_tilt < c->flip_cos) ? f->t_flip + dt : 0;

    if (in->cmd_disarm) {                                   /* 上锁指令最优先 */
        go(f, FC_MODE_DISARMED, FC_WHY_COMMAND);
        return;
    }
    if (armed) {                                            /* 必须立刻停转的故障 */
        if (!in->est_ok) { go(f, FC_MODE_FAULT, FC_WHY_ESTIMATOR); return; }
        if (f->t_flip > c->flip_time) { go(f, FC_MODE_FAULT, FC_WHY_FLIPPED); return; }
    }

    switch (f->mode) {
    case FC_MODE_DISARMED:
        if (in->cmd_arm) {
            if (in->calibrated && in->est_ok && in->cos_tilt > c->arm_min_cos && in->vbat > c->v_low
                && f->t_link < c->link_timeout)
                go(f, FC_MODE_FLYING, FC_WHY_COMMAND);
            else
                f->reason = FC_WHY_NOT_READY;
        }
        break;
    case FC_MODE_FLYING:
        if (in->cmd_land) start_landing(f, in, FC_WHY_COMMAND);
        else if (f->t_link > c->link_timeout) start_landing(f, in, FC_WHY_LINK_LOST);
        else if (f->t_batt > c->v_low_time) start_landing(f, in, FC_WHY_LOW_BATTERY);
        else if (in->pos[0] > c->fence_xy || in->pos[0] < -c->fence_xy || in->pos[1] > c->fence_xy
                 || in->pos[1] < -c->fence_xy || in->pos[2] > c->fence_z)
            start_landing(f, in, FC_WHY_GEOFENCE);
        break;
    case FC_MODE_LANDING:
        f->land_pos[2] -= c->land_speed * dt;               /* 目标点匀速下降,可以降到地面以下以保证压到地 */
        f->t_ground = (in->pos[2] < c->land_height) ? f->t_ground + dt : 0;
        if (f->t_ground > c->land_time) go(f, FC_MODE_DISARMED, FC_WHY_TOUCHDOWN);
        break;
    default:                                                /* 故障:等上锁指令 */
        break;
    }
}
