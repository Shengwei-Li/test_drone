#include "fc_eskf.h"

void fc_eskf_init(fc_eskf *e, const fc_eskf_cfg *cfg, fc_float g, const fc_float pos0[3]) {
    int i, j;
    fc_float std[5];
    e->cfg = *cfg;
    e->g = g;
    for (i = 0; i < 3; i++) { e->p[i] = pos0[i]; e->v[i] = 0; e->bg[i] = 0; e->ba[i] = 0; e->gyro[i] = 0; }
    e->q[0] = 1; e->q[1] = e->q[2] = e->q[3] = 0;
    std[0] = cfg->init_pos_std; std[1] = cfg->init_vel_std; std[2] = cfg->init_tilt_std;
    std[3] = cfg->init_gyro_bias_std; std[4] = cfg->init_acc_bias_std;
    for (i = 0; i < FC_NE; i++)
        for (j = 0; j < FC_NE; j++)
            e->P[i][j] = (i == j) ? std[i / 3] * std[i / 3] : 0;
}

int fc_eskf_calibrate_gyro(fc_eskf *e, const fc_float *samples, int n, fc_float max_std) {
    fc_float mean[3] = {0, 0, 0}, var[3] = {0, 0, 0}, std;
    int i, k;
    if (n < 50) return 0;
    for (i = 0; i < n; i++) for (k = 0; k < 3; k++) mean[k] += samples[3 * i + k];
    for (k = 0; k < 3; k++) mean[k] /= (fc_float)n;
    for (i = 0; i < n; i++)
        for (k = 0; k < 3; k++) { fc_float d = samples[3 * i + k] - mean[k]; var[k] += d * d; }
    for (k = 0; k < 3; k++)
        if (FC_SQRT(var[k] / (fc_float)n) > max_std) return 0;      /* 机体在动,拒绝校准 */
    fc_eskf_set_gyro_bias(e, mean, n);
    (void)std;
    return 1;
}

void fc_eskf_set_gyro_bias(fc_eskf *e, const fc_float mean[3], int n) {
    fc_float std = e->cfg.gyro_std / FC_SQRT((fc_float)n);
    int i, k;
    if (std < (fc_float)1e-3) std = (fc_float)1e-3;
    for (k = 0; k < 3; k++) e->bg[k] = mean[k];
    for (i = 0; i < FC_NE; i++)
        for (k = 9; k < 12; k++) { e->P[i][k] = 0; e->P[k][i] = 0; }
    for (k = 9; k < 12; k++) e->P[k][k] = std * std;
}

void fc_eskf_predict(fc_eskf *e, const fc_float gyro[3], const fc_float acc[3], fc_float dt) {
    static fc_float Phi[FC_NE][FC_NE], T[FC_NE][FC_NE];   /* 静态:不占用栈 */
    fc_float R[3][3], w[3], ab[3], aw[3], rv[3], Qd[FC_NE], S[3][3];
    const fc_eskf_cfg *c = &e->cfg;
    int i, j, k;

    for (i = 0; i < 3; i++) { e->gyro[i] = gyro[i]; w[i] = gyro[i] - e->bg[i]; ab[i] = acc[i] - e->ba[i]; }
    fc_quat_to_rot(e->q, R);
    for (i = 0; i < 3; i++) aw[i] = R[i][0] * ab[0] + R[i][1] * ab[1] + R[i][2] * ab[2];
    aw[2] -= e->g;

    /* 名义状态积分 */
    for (i = 0; i < 3; i++) {
        e->p[i] += e->v[i] * dt + (fc_float)0.5 * aw[i] * dt * dt;
        e->v[i] += aw[i] * dt;
        rv[i] = w[i] * dt;
    }
    fc_quat_apply_rotvec(e->q, rv);

    /* 误差状态转移矩阵 Φ(用积分前的 R,与 Python 版本一致) */
    for (i = 0; i < FC_NE; i++) for (j = 0; j < FC_NE; j++) Phi[i][j] = (fc_float)(i == j);
    S[0][0] = 0;      S[0][1] = -ab[2]; S[0][2] = ab[1];
    S[1][0] = ab[2];  S[1][1] = 0;      S[1][2] = -ab[0];
    S[2][0] = -ab[1]; S[2][1] = ab[0];  S[2][2] = 0;
    for (i = 0; i < 3; i++) {
        Phi[i][3 + i] = dt;                                           /* δp ← δv */
        for (j = 0; j < 3; j++) {
            Phi[3 + i][6 + j] = -(R[i][0] * S[0][j] + R[i][1] * S[1][j] + R[i][2] * S[2][j]) * dt;  /* δv ← δθ */
            Phi[3 + i][12 + j] = -R[i][j] * dt;                       /* δv ← δba */
        }
        Phi[6 + i][9 + i] = -dt;                                      /* δθ ← δbg */
    }
    Phi[6][7] += w[2] * dt;  Phi[6][8] -= w[1] * dt;                  /* δθ ← δθ:I − [w]× dt */
    Phi[7][6] -= w[2] * dt;  Phi[7][8] += w[0] * dt;
    Phi[8][6] += w[1] * dt;  Phi[8][7] -= w[0] * dt;

    for (i = 0; i < FC_NE; i++) Qd[i] = 0;
    for (i = 0; i < 3; i++) {
        Qd[3 + i] = (c->acc_std * dt) * (c->acc_std * dt);
        Qd[6 + i] = (c->gyro_std * dt) * (c->gyro_std * dt);
        Qd[9 + i] = c->gyro_bias_walk * c->gyro_bias_walk * dt;
        Qd[12 + i] = c->acc_bias_walk * c->acc_bias_walk * dt;
    }

    /* P ← Φ P Φᵀ + Q */
    for (i = 0; i < FC_NE; i++)
        for (j = 0; j < FC_NE; j++) {
            fc_float s = 0;
            for (k = 0; k < FC_NE; k++) s += Phi[i][k] * e->P[k][j];
            T[i][j] = s;
        }
    for (i = 0; i < FC_NE; i++)
        for (j = 0; j < FC_NE; j++) {
            fc_float s = 0;
            for (k = 0; k < FC_NE; k++) s += T[i][k] * Phi[j][k];
            e->P[i][j] = s + ((i == j) ? Qd[i] : 0);
        }
}

static void inject(fc_eskf *e, const fc_float d[FC_NE]) {
    int i;
    for (i = 0; i < 3; i++) { e->p[i] += d[i]; e->v[i] += d[3 + i]; e->bg[i] += d[9 + i]; e->ba[i] += d[12 + i]; }
    fc_quat_apply_rotvec(e->q, &d[6]);
}

/* 通用更新:m 个观测(1 或 2),观测噪声方差 r_var(各分量相同、互不相关)。Joseph 形式。 */
static void update(fc_eskf *e, int m, const fc_float res[2], fc_float H[2][FC_NE], fc_float r_var) {
    static fc_float A[FC_NE][FC_NE], T[FC_NE][FC_NE];
    fc_float PHt[FC_NE][2], S[2][2], Si[2][2], K[FC_NE][2], d[FC_NE];
    int i, j, k, a;

    for (i = 0; i < FC_NE; i++)
        for (a = 0; a < m; a++) {
            fc_float s = 0;
            for (k = 0; k < FC_NE; k++) s += e->P[i][k] * H[a][k];
            PHt[i][a] = s;
        }
    for (a = 0; a < m; a++)
        for (j = 0; j < m; j++) {
            fc_float s = (a == j) ? r_var : 0;
            for (k = 0; k < FC_NE; k++) s += H[a][k] * PHt[k][j];
            S[a][j] = s;
        }
    if (m == 1) {
        Si[0][0] = 1 / S[0][0];
    } else {
        fc_float det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
        Si[0][0] = S[1][1] / det;  Si[0][1] = -S[0][1] / det;
        Si[1][0] = -S[1][0] / det; Si[1][1] = S[0][0] / det;
    }
    for (i = 0; i < FC_NE; i++)
        for (a = 0; a < m; a++) {
            fc_float s = 0;
            for (j = 0; j < m; j++) s += PHt[i][j] * Si[j][a];
            K[i][a] = s;
        }
    /* A = I − K H;P ← A P Aᵀ + K R Kᵀ */
    for (i = 0; i < FC_NE; i++)
        for (j = 0; j < FC_NE; j++) {
            fc_float s = (fc_float)(i == j);
            for (a = 0; a < m; a++) s -= K[i][a] * H[a][j];
            A[i][j] = s;
        }
    for (i = 0; i < FC_NE; i++)
        for (j = 0; j < FC_NE; j++) {
            fc_float s = 0;
            for (k = 0; k < FC_NE; k++) s += A[i][k] * e->P[k][j];
            T[i][j] = s;
        }
    for (i = 0; i < FC_NE; i++)
        for (j = 0; j < FC_NE; j++) {
            fc_float s = 0;
            for (k = 0; k < FC_NE; k++) s += T[i][k] * A[j][k];
            for (a = 0; a < m; a++) s += K[i][a] * r_var * K[j][a];
            e->P[i][j] = s;
        }
    for (i = 0; i < FC_NE; i++) {
        d[i] = 0;
        for (a = 0; a < m; a++) d[i] += K[i][a] * res[a];
    }
    inject(e, d);
}

void fc_eskf_update_tof(fc_eskf *e, fc_float dist) {
    fc_float R[3][3], H[2][FC_NE], res[2], c22, pz, k;
    int j;
    fc_quat_to_rot(e->q, R);
    c22 = R[2][2]; pz = e->p[2];
    if (c22 < e->cfg.min_cos_tilt) return;
    for (j = 0; j < FC_NE; j++) H[0][j] = 0;
    H[0][2] = 1 / c22;                              /* 测到的斜距 = 高度 / cos(倾角) */
    k = -pz / (c22 * c22);
    H[0][6] = k * -R[2][1]; H[0][7] = k * R[2][0]; H[0][8] = 0;
    res[0] = dist - pz / c22;
    update(e, 1, res, H, e->cfg.tof_std * e->cfg.tof_std);
}

void fc_eskf_update_flow(fc_eskf *e, const fc_float flow[2]) {
    fc_float R[3][3], H[2][FC_NE], res[2], w[3], vb[3], dc[3], c22, pz, s;
    int i, j;
    fc_quat_to_rot(e->q, R);
    c22 = R[2][2]; pz = e->p[2];
    if (pz < e->cfg.flow_min_height || c22 < e->cfg.min_cos_tilt) return;
    for (i = 0; i < 3; i++) {
        w[i] = e->gyro[i] - e->bg[i];
        vb[i] = R[0][i] * e->v[0] + R[1][i] * e->v[1] + R[2][i] * e->v[2];     /* Rᵀ v */
    }
    s = c22 / pz;                                   /* 1 / 到地面的距离 */
    dc[0] = -R[2][1]; dc[1] = R[2][0]; dc[2] = 0;   /* c22 对 δθ 的导数 */
    for (i = 0; i < 2; i++) for (j = 0; j < FC_NE; j++) H[i][j] = 0;
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 3; j++) H[i][3 + j] = R[j][i] * s;
        H[i][2] = -vb[i] * c22 / (pz * pz);
    }
    /* skew(vb) 的前两行 × s,加上 (vb/pz) ⊗ dc */
    H[0][6] = vb[0] / pz * dc[0];             H[0][7] = -vb[2] * s + vb[0] / pz * dc[1]; H[0][8] = vb[1] * s;
    H[1][6] = vb[2] * s + vb[1] / pz * dc[0]; H[1][7] = vb[1] / pz * dc[1];              H[1][8] = -vb[0] * s;
    H[0][10] = 1;                                   /* 光流里的转动分量含陀螺零偏 */
    H[1][9] = -1;
    res[0] = flow[0] - (vb[0] * s - w[1]);
    res[1] = flow[1] - (vb[1] * s + w[0]);
    update(e, 2, res, H, e->cfg.flow_std * e->cfg.flow_std);
}

void fc_eskf_omega(const fc_eskf *e, fc_float omega[3]) {
    int i;
    for (i = 0; i < 3; i++) omega[i] = e->gyro[i] - e->bg[i];
}
