/* 软件在环 (SIL) 用的动态库接口:把核心库包成一组只收发数组的函数,供 Python (ctypes) 调用。
 * 这一层只在电脑上用;上板时由真正的固件主循环直接调用 core/ 里的函数。
 */
#include "../core/fc_ctrl.h"
#include "../core/fc_eskf.h"
#include "../core/fc_policy.h"

#include <string.h>

#ifdef _WIN32
#define FC_API __declspec(dllexport)
#else
#define FC_API __attribute__((visibility("default")))
#endif

static fc_eskf g_eskf;
static fc_ctrl g_ctrl;
static fc_policy g_policy;

FC_API int fc_sil_float_size(void) { return (int)sizeof(fc_float); }
FC_API int fc_sil_state_bytes(void) { return (int)(sizeof(fc_eskf) + sizeof(fc_ctrl) + sizeof(fc_policy)); }

/* ---- 估计器 ---- */
FC_API void fc_sil_eskf_init(const fc_float cfg[13], fc_float g, const fc_float pos0[3]) {
    fc_eskf_cfg c;
    c.gyro_std = cfg[0]; c.acc_std = cfg[1]; c.gyro_bias_walk = cfg[2]; c.acc_bias_walk = cfg[3];
    c.tof_std = cfg[4]; c.flow_std = cfg[5]; c.flow_min_height = cfg[6]; c.min_cos_tilt = cfg[7];
    c.init_pos_std = cfg[8]; c.init_vel_std = cfg[9]; c.init_tilt_std = cfg[10];
    c.init_gyro_bias_std = cfg[11]; c.init_acc_bias_std = cfg[12];
    fc_eskf_init(&g_eskf, &c, g, pos0);
}
FC_API int fc_sil_eskf_calibrate(const fc_float *samples, int n, fc_float max_std) {
    return fc_eskf_calibrate_gyro(&g_eskf, samples, n, max_std);
}
FC_API void fc_sil_eskf_predict(const fc_float gyro[3], const fc_float acc[3], fc_float dt) {
    fc_eskf_predict(&g_eskf, gyro, acc, dt);
}
FC_API void fc_sil_eskf_update_tof(fc_float d) { fc_eskf_update_tof(&g_eskf, d); }
FC_API void fc_sil_eskf_update_flow(const fc_float flow[2]) { fc_eskf_update_flow(&g_eskf, flow); }
/* out:p(3) v(3) q(4) bg(3) ba(3) gyro(3) = 19 个数;P:15×15 按行 */
FC_API void fc_sil_eskf_get(fc_float out[19], fc_float *P) {
    int i, j, n = 0;
    for (i = 0; i < 3; i++) out[n++] = g_eskf.p[i];
    for (i = 0; i < 3; i++) out[n++] = g_eskf.v[i];
    for (i = 0; i < 4; i++) out[n++] = g_eskf.q[i];
    for (i = 0; i < 3; i++) out[n++] = g_eskf.bg[i];
    for (i = 0; i < 3; i++) out[n++] = g_eskf.ba[i];
    for (i = 0; i < 3; i++) out[n++] = g_eskf.gyro[i];
    if (P) for (i = 0; i < FC_NE; i++) for (j = 0; j < FC_NE; j++) P[i * FC_NE + j] = g_eskf.P[i][j];
}

/* ---- 控制器 ----
 * veh:mass g kf f_max motor_tau J(3) alloc(16) alloc_inv(16) = 40 个数 */
FC_API void fc_sil_ctrl_init(const fc_float veh[40], fc_float dt, int use_indi) {
    fc_vehicle v;
    fc_ctrl_gains k;
    int i, j;
    v.mass = veh[0]; v.g = veh[1]; v.kf = veh[2]; v.f_max = veh[3]; v.motor_tau = veh[4];
    for (i = 0; i < 3; i++) v.J[i] = veh[5 + i];
    for (i = 0; i < 4; i++) for (j = 0; j < 4; j++) { v.alloc[i][j] = veh[8 + 4 * i + j]; v.alloc_inv[i][j] = veh[24 + 4 * i + j]; }
    fc_ctrl_default_gains(&k);
    fc_ctrl_init(&g_ctrl, &v, &k, dt, use_indi);
}
/* state:pos(3) vel(3) quat(4) omega(3) = 13;ref:pos vel acc jerk yaw = 13;acc 可为 NULL */
FC_API void fc_sil_ctrl_update(const fc_float state[13], const fc_float ref[13], const fc_float *acc, fc_float w_cmd[4]) {
    fc_reference r;
    int i;
    for (i = 0; i < 3; i++) { r.pos[i] = ref[i]; r.vel[i] = ref[3 + i]; r.acc[i] = ref[6 + i]; r.jerk[i] = ref[9 + i]; }
    r.yaw = ref[12];
    fc_ctrl_update(&g_ctrl, &state[0], &state[3], &state[6], &state[10], &r, acc, w_cmd);
}

/* ---- 强化学习策略 ---- */
FC_API void fc_sil_policy_load(const fc_float *W0, const fc_float *b0, const fc_float *W1, const fc_float *b1,
                               const fc_float *W2, const fc_float *b2, fc_float w_hover, fc_float w_max) {
    int i, j;
    for (i = 0; i < FC_HID; i++) { g_policy.b0[i] = b0[i]; g_policy.b1[i] = b1[i];
        for (j = 0; j < FC_OBS; j++) g_policy.W0[i][j] = W0[i * FC_OBS + j];
        for (j = 0; j < FC_HID; j++) g_policy.W1[i][j] = W1[i * FC_HID + j]; }
    for (i = 0; i < FC_ACT; i++) { g_policy.b2[i] = b2[i];
        for (j = 0; j < FC_HID; j++) g_policy.W2[i][j] = W2[i * FC_HID + j]; }
    fc_policy_reset(&g_policy, w_hover, w_max);
}
/* ref:pos(3) vel(3) acc(3) */
FC_API void fc_sil_policy_update(const fc_float state[13], const fc_float ref[9], fc_float w_cmd[4]) {
    fc_policy_update(&g_policy, &state[0], &state[3], &state[6], &state[10], &ref[0], &ref[3], &ref[6], w_cmd);
}

/* ---- 基准:在库内部循环 n 次,由调用方计时(避免把 Python 调用开销算进去) ---- */
FC_API void fc_sil_bench(int which, int n) {
    static const fc_float gyro[3] = {(fc_float)0.01, (fc_float)-0.02, (fc_float)0.005};
    static const fc_float acc[3] = {(fc_float)0.1, (fc_float)-0.1, (fc_float)9.8};
    static const fc_float flow[2] = {(fc_float)0.01, (fc_float)-0.01};
    static const fc_float state[13] = {0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
    static const fc_float ref[13] = {(fc_float)0.1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    fc_float w[4];
    fc_eskf saved = g_eskf;
    int i;
    for (i = 0; i < n; i++) {
        if (which == 0) fc_eskf_predict(&g_eskf, gyro, acc, (fc_float)1e-6);
        else if (which == 1) { g_eskf.p[2] = 1; fc_eskf_update_tof(&g_eskf, 1); }
        else if (which == 2) { g_eskf.p[2] = 1; fc_eskf_update_flow(&g_eskf, flow); }
        else if (which == 3) fc_sil_ctrl_update(state, ref, acc, w);
        else fc_sil_policy_update(state, ref, w);
    }
    g_eskf = saved;
}

/* ---- MPC ----
 * tab:A(9) B(3) Qd(3) P(9) K(N*3) C1(N) AmBKt(N*9) rho_u rho_x(3) u_max x_max(9) dt_mpc iters solve_every */
#include "../core/fc_mpc.h"
static fc_mpc g_mpc;
FC_API int fc_sil_mpc_table_len(void) { return 9 + 3 + 3 + 9 + FC_MPC_N * 13 + 1 + 3 + 1 + 9 + 3; }
FC_API void fc_sil_mpc_init(const fc_float *tab, const fc_float veh[40], fc_float dt) {
    fc_mpc_tables t;
    fc_ctrl_gains k;
    int i, j, n = 0, s;
    for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) t.A[i][j] = tab[n++];
    for (i = 0; i < 3; i++) t.B[i] = tab[n++];
    for (i = 0; i < 3; i++) t.Qd[i] = tab[n++];
    for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) t.P[i][j] = tab[n++];
    for (s = 0; s < FC_MPC_N; s++) for (i = 0; i < 3; i++) t.K[s][i] = tab[n++];
    for (s = 0; s < FC_MPC_N; s++) t.C1[s] = tab[n++];
    for (s = 0; s < FC_MPC_N; s++) for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) t.AmBKt[s][i][j] = tab[n++];
    t.rho_u = tab[n++];
    for (i = 0; i < 3; i++) t.rho_x[i] = tab[n++];
    t.u_max = tab[n++];
    for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) t.x_max[i][j] = tab[n++];
    t.dt_mpc = tab[n++]; t.iters = (int)tab[n++]; t.solve_every = (int)tab[n++];
    fc_sil_ctrl_init(veh, dt, 1);
    fc_ctrl_default_gains(&k);
    fc_mpc_init(&g_mpc, &t, &g_ctrl.veh, &k, dt);
}
FC_API int fc_sil_mpc_needs_preview(void) { return fc_mpc_needs_preview(&g_mpc); }
FC_API void fc_sil_mpc_update(const fc_float state[13], const fc_float *preview, fc_float yaw, const fc_float *acc,
                              fc_float w_cmd[4]) {
    fc_mpc_update(&g_mpc, &state[0], &state[3], &state[6], &state[10], preview, yaw, acc, w_cmd);
}

/* ---- 通信协议 ---- */
#include "../core/fc_proto.h"
static fc_proto_parser g_parser;
FC_API void fc_sil_proto_reset(void) { fc_proto_init(&g_parser); }
/* 喂一串字节,把解出的帧依次写到 out:每帧 [类型, 长度, 载荷...];返回帧数 */
FC_API int fc_sil_proto_parse(const uint8_t *data, int n, uint8_t *out, int *n_bad) {
    fc_msg m;
    int i, count = 0, w = 0;
    for (i = 0; i < n; i++)
        if (fc_proto_feed(&g_parser, data[i], &m)) {
            out[w++] = m.type; out[w++] = m.len;
            memcpy(&out[w], m.payload, m.len); w += m.len;
            count++;
        }
    *n_bad = (int)g_parser.n_bad;
    return count;
}
FC_API int fc_sil_proto_encode(int type, const uint8_t *payload, int len, uint8_t *out) {
    return fc_proto_encode((uint8_t)type, payload, (uint8_t)len, out);
}

/* ---- 整机主循环 ---- */
#include "../core/fc_app.h"
static fc_app g_app;
FC_API void fc_sil_app_init(const fc_float est_cfg[13], const fc_float veh[40], fc_float dt, fc_float w_hover,
                            fc_float w_max, int use_policy) {
    fc_fsm_cfg fc;
    static const fc_float zero[3] = {0, 0, 0};
    fc_sil_eskf_init(est_cfg, veh[1], zero);
    fc_sil_ctrl_init(veh, dt, 1);
    fc_fsm_default_cfg(&fc);
    fc_app_init(&g_app, &g_ctrl.veh, &g_eskf.cfg, &fc, dt, w_hover, w_max);
    g_app.policy = &g_policy;                      /* 权重需先用 fc_sil_policy_load 装入 */
    g_app.use_policy = use_policy;
}
/* sens:gyro(3) acc(3) has_flow flow(2) has_tof tof vbat = 12 个数
 * out:motor(4) mode reason calibrated p(3) v(3) q(4) = 17 个数;返回要发出的字节数(写在 tx) */
FC_API int fc_sil_app_step(const fc_float sens[12], const uint8_t *rx, int rx_len, fc_float out[17], uint8_t *tx) {
    fc_inputs in;
    fc_outputs o;
    int i, n = 0;
    for (i = 0; i < 3; i++) { in.gyro[i] = sens[i]; in.acc[i] = sens[3 + i]; }
    in.has_flow = sens[6] > (fc_float)0.5; in.flow[0] = sens[7]; in.flow[1] = sens[8];
    in.has_tof = sens[9] > (fc_float)0.5; in.tof = sens[10];
    in.vbat = sens[11];
    in.rx = rx; in.rx_len = rx_len;
    fc_app_step(&g_app, &in, &o);
    for (i = 0; i < 4; i++) out[n++] = o.motor[i];
    out[n++] = (fc_float)g_app.fsm.mode; out[n++] = (fc_float)g_app.fsm.reason; out[n++] = (fc_float)g_app.calibrated;
    for (i = 0; i < 3; i++) out[n++] = g_app.est.p[i];
    for (i = 0; i < 3; i++) out[n++] = g_app.est.v[i];
    for (i = 0; i < 4; i++) out[n++] = g_app.est.q[i];
    memcpy(tx, o.tx, (size_t)o.tx_len);
    return o.tx_len;
}
