#include "fc_app.h"

void fc_app_init(fc_app *a, const fc_vehicle *veh, const fc_eskf_cfg *est_cfg, const fc_fsm_cfg *fsm_cfg,
                 fc_float dt, fc_float w_hover, fc_float w_max) {
    static const fc_float zero[3] = {0, 0, 0};
    fc_ctrl_gains k;
    int i;
    fc_eskf_init(&a->est, est_cfg, veh->g, zero);
    fc_ctrl_default_gains(&k);
    fc_ctrl_init(&a->ctrl, veh, &k, dt, 1);
    fc_fsm_init(&a->fsm, fsm_cfg);
    fc_proto_init(&a->rx);
    a->policy = 0;
    a->use_policy = 0;
    a->dt = dt; a->w_hover = w_hover; a->w_max = w_max;
    a->calibrated = 0; a->calib_n = 0;
    for (i = 0; i < 3; i++) {
        a->calib_sum[i] = 0; a->calib_sq[i] = 0;
        a->sp.pos[i] = 0; a->sp.vel[i] = 0; a->sp.acc[i] = 0; a->sp.jerk[i] = 0;
    }
    a->sp.yaw = 0;
    a->tick = 0;
    a->policy_div = 0;
    for (i = 0; i < 4; i++) a->last_motor[i] = 0;
}

static int finite3(const fc_float v[3]) {
    int i;
    for (i = 0; i < 3; i++) if (!(v[i] == v[i]) || v[i] > (fc_float)1e6 || v[i] < (fc_float)-1e6) return 0;
    return 1;
}

/* 上电后机体静止时累加陀螺读数;攒够后算均值和抖动,抖动小才接受,否则重来 */
static void calibrate(fc_app *a, const fc_float gyro[3]) {
    int i;
    for (i = 0; i < 3; i++) { a->calib_sum[i] += gyro[i]; a->calib_sq[i] += gyro[i] * gyro[i]; }
    if (++a->calib_n < FC_CALIB_SAMPLES) return;
    {
        fc_float n = (fc_float)a->calib_n, mean[3], std_max = 0;
        for (i = 0; i < 3; i++) {
            fc_float var;
            mean[i] = a->calib_sum[i] / n;
            var = a->calib_sq[i] / n - mean[i] * mean[i];
            if (var > std_max) std_max = var;
        }
        if (std_max < (fc_float)(0.1 * 0.1)) {
            fc_eskf_set_gyro_bias(&a->est, mean, a->calib_n);
            a->calibrated = 1;
        }
        a->calib_n = 0;
        for (i = 0; i < 3; i++) { a->calib_sum[i] = 0; a->calib_sq[i] = 0; }
    }
}

static void handle(fc_app *a, const fc_msg *m, fc_fsm_in *fi) {
    int i;
    fi->link_rx = 1;
    switch (m->type) {
    case FC_MSG_ARM: fi->cmd_arm = 1; break;
    case FC_MSG_DISARM: fi->cmd_disarm = 1; break;
    case FC_MSG_LAND: fi->cmd_land = 1; break;
    case FC_MSG_SETPOINT:
        if (m->len == 40) {
            for (i = 0; i < 3; i++) {
                a->sp.pos[i] = (fc_float)fc_proto_get_f32(&m->payload[4 * i]);
                a->sp.vel[i] = (fc_float)fc_proto_get_f32(&m->payload[12 + 4 * i]);
                a->sp.acc[i] = (fc_float)fc_proto_get_f32(&m->payload[24 + 4 * i]);
            }
            a->sp.yaw = (fc_float)fc_proto_get_f32(&m->payload[36]);
        }
        break;
    default: break;
    }
}

static void telemetry(const fc_app *a, fc_float vbat, fc_outputs *out) {
    uint8_t pl[46];
    int i;
    pl[0] = (uint8_t)a->fsm.mode;
    pl[1] = (uint8_t)a->fsm.reason;
    fc_proto_put_f32(&pl[2], (float)vbat);
    for (i = 0; i < 3; i++) fc_proto_put_f32(&pl[6 + 4 * i], (float)a->est.p[i]);
    for (i = 0; i < 3; i++) fc_proto_put_f32(&pl[18 + 4 * i], (float)a->est.v[i]);
    for (i = 0; i < 4; i++) fc_proto_put_f32(&pl[30 + 4 * i], (float)a->est.q[i]);
    out->tx_len = fc_proto_encode(FC_MSG_TELEMETRY, pl, 46, out->tx);
}

void fc_app_step(fc_app *a, const fc_inputs *in, fc_outputs *out) {
    fc_fsm_in fi;
    fc_msg msg;
    fc_float R[3][3], omega[3], acc_b[3];
    fc_reference ref;
    int i;

    /* 1. 通信:解析这一拍收到的字节 */
    fi.cmd_arm = fi.cmd_disarm = fi.cmd_land = 0;
    fi.link_rx = 0;
    for (i = 0; i < in->rx_len; i++)
        if (fc_proto_feed(&a->rx, in->rx[i], &msg)) handle(a, &msg, &fi);

    /* 2. 估计:上锁时先做陀螺校准,校准完保持静止状态等待解锁;解锁后才开始积分 */
    if (!a->calibrated) {
        calibrate(a, in->gyro);
        for (i = 0; i < 3; i++) a->est.gyro[i] = in->gyro[i];
    } else if (a->fsm.mode == FC_MODE_DISARMED) {
        for (i = 0; i < 3; i++) a->est.gyro[i] = in->gyro[i];
    } else {
        fc_eskf_predict(&a->est, in->gyro, in->acc, a->dt);
        if (in->has_flow) fc_eskf_update_flow(&a->est, in->flow);
        if (in->has_tof) fc_eskf_update_tof(&a->est, in->tof);
    }
    fc_quat_to_rot(a->est.q, R);
    fc_eskf_omega(&a->est, omega);

    /* 3. 状态机 */
    fi.calibrated = a->calibrated;
    fi.est_ok = finite3(a->est.p) && finite3(a->est.v) && finite3(omega);
    fi.vbat = in->vbat;
    fi.cos_tilt = R[2][2];
    for (i = 0; i < 3; i++) fi.pos[i] = a->est.p[i];
    {
        fc_mode before = a->fsm.mode;
        fc_fsm_step(&a->fsm, &fi, a->dt);
        if (before == FC_MODE_DISARMED && a->fsm.mode == FC_MODE_FLYING) {
            /* 刚解锁:控制器内部状态清零,从干净的状态起飞 */
            fc_ctrl_gains k = a->ctrl.k;
            fc_vehicle veh = a->ctrl.veh;
            fc_ctrl_init(&a->ctrl, &veh, &k, a->dt, 1);
            if (a->policy) fc_policy_reset(a->policy, a->w_hover, a->w_max);
            a->policy_div = 0;
        }
    }

    /* 4. 控制 */
    if (!fc_fsm_motors_on(&a->fsm)) {
        for (i = 0; i < 4; i++) out->motor[i] = 0;
    } else {
        if (a->fsm.mode == FC_MODE_LANDING) {               /* 降落:忽略地面的目标,用状态机给的下降点 */
            for (i = 0; i < 3; i++) { ref.pos[i] = a->fsm.land_pos[i]; ref.vel[i] = 0; ref.acc[i] = 0; ref.jerk[i] = 0; }
            ref.vel[2] = -a->fsm.cfg.land_speed;
            ref.yaw = a->sp.yaw;
        } else {
            ref = a->sp;
        }
        for (i = 0; i < 3; i++) acc_b[i] = in->acc[i] - a->est.ba[i];
        if (a->use_policy && a->policy && a->fsm.mode == FC_MODE_FLYING) {
            int every = (int)(FC_POLICY_DT / a->dt + (fc_float)0.5);
            if (a->policy_div++ % (every < 1 ? 1 : every) == 0)
                fc_policy_update(a->policy, a->est.p, a->est.v, a->est.q, omega, ref.pos, ref.vel, ref.acc,
                                 a->last_motor);
            for (i = 0; i < 4; i++) out->motor[i] = a->last_motor[i];
        } else {
            fc_ctrl_update(&a->ctrl, a->est.p, a->est.v, a->est.q, omega, &ref, acc_b, out->motor);
        }
    }

    /* 5. 遥测 */
    out->tx_len = 0;
    if (a->tick++ % FC_TELEMETRY_EVERY == 0) telemetry(a, in->vbat, out);
}
