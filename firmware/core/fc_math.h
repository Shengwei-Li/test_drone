/* 基础类型和小型向量/四元数运算。整个核心库不分配内存、不依赖任何芯片外设。
 *
 * fc_float 默认是单精度(STM32F4 的硬件浮点只支持单精度);
 * 编译时加 -DFC_DOUBLE 换成双精度,用来在电脑上与 Python 版本逐位对照。
 * 约定与仿真相同:世界系 ENU,机体系 FLU,四元数 [w,x,y,z](机体→世界)。
 */
#ifndef FC_MATH_H
#define FC_MATH_H

#include <math.h>

#ifdef FC_DOUBLE
typedef double fc_float;
#define FC_SQRT sqrt
#define FC_SIN sin
#define FC_COS cos
#define FC_TANH tanh
#define FC_EXP exp
#else
typedef float fc_float;
#define FC_SQRT sqrtf
#define FC_SIN sinf
#define FC_COS cosf
#define FC_TANH tanhf
#define FC_EXP expf
#endif

#define FC_PI ((fc_float)3.14159265358979323846)

static inline fc_float fc_clamp(fc_float x, fc_float lo, fc_float hi) { return x < lo ? lo : (x > hi ? hi : x); }
static inline fc_float fc_dot3(const fc_float a[3], const fc_float b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
static inline fc_float fc_norm3(const fc_float a[3]) { return FC_SQRT(fc_dot3(a, a)); }

static inline void fc_cross3(const fc_float a[3], const fc_float b[3], fc_float out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

/* 四元数 → 旋转矩阵(机体→世界),R[行][列] */
static inline void fc_quat_to_rot(const fc_float q[4], fc_float R[3][3]) {
    fc_float w = q[0], x = q[1], y = q[2], z = q[3];
    R[0][0] = 1 - 2 * (y * y + z * z); R[0][1] = 2 * (x * y - w * z);     R[0][2] = 2 * (x * z + w * y);
    R[1][0] = 2 * (x * y + w * z);     R[1][1] = 1 - 2 * (x * x + z * z); R[1][2] = 2 * (y * z - w * x);
    R[2][0] = 2 * (x * z - w * y);     R[2][1] = 2 * (y * z + w * x);     R[2][2] = 1 - 2 * (x * x + y * y);
}

static inline void fc_quat_mul(const fc_float a[4], const fc_float b[4], fc_float out[4]) {
    out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

static inline void fc_quat_normalize(fc_float q[4]) {
    fc_float n = FC_SQRT(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    q[0] /= n; q[1] /= n; q[2] /= n; q[3] /= n;
}

/* 旋转向量(轴 × 角度)→ 四元数 */
static inline void fc_quat_from_rotvec(const fc_float r[3], fc_float q[4]) {
    fc_float angle = fc_norm3(r);
    if (angle < (fc_float)1e-9) {
        q[0] = 1; q[1] = (fc_float)0.5 * r[0]; q[2] = (fc_float)0.5 * r[1]; q[3] = (fc_float)0.5 * r[2];
    } else {
        fc_float s = FC_SIN(angle / 2) / angle;
        q[0] = FC_COS(angle / 2); q[1] = s * r[0]; q[2] = s * r[1]; q[3] = s * r[2];
    }
    fc_quat_normalize(q);
}

/* q ← q ⊗ exp(r),并归一化 */
static inline void fc_quat_apply_rotvec(fc_float q[4], const fc_float r[3]) {
    fc_float dq[4], out[4];
    fc_quat_from_rotvec(r, dq);
    fc_quat_mul(q, dq, out);
    fc_quat_normalize(out);
    q[0] = out[0]; q[1] = out[1]; q[2] = out[2]; q[3] = out[3];
}

#endif
