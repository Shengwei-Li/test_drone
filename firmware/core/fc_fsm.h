/* 飞行状态机和安全逻辑:决定电机能不能转、现在该听谁的指令。
 *
 *   上锁 ──解锁指令(且自检通过)──▶ 飞行 ──降落指令/链路丢失/低电压/越界──▶ 降落 ──触地──▶ 上锁
 *                                    │                                        │
 *                                    └──────── 翻倒 / 估计器失效 ──────────────┴──▶ 故障(电机停转,需上锁指令清除)
 *   任何状态下收到上锁指令都立即停转。
 */
#ifndef FC_FSM_H
#define FC_FSM_H

#include "fc_math.h"

typedef enum { FC_MODE_DISARMED = 0, FC_MODE_FLYING = 1, FC_MODE_LANDING = 2, FC_MODE_FAULT = 3 } fc_mode;

typedef enum {
    FC_WHY_NONE = 0,
    FC_WHY_COMMAND = 1,         /* 收到指令 */
    FC_WHY_LINK_LOST = 2,       /* 链路丢失 */
    FC_WHY_LOW_BATTERY = 3,     /* 低电压 */
    FC_WHY_GEOFENCE = 4,        /* 飞出允许范围 */
    FC_WHY_FLIPPED = 5,         /* 翻倒 */
    FC_WHY_ESTIMATOR = 6,       /* 估计器失效 */
    FC_WHY_NOT_READY = 7,       /* 解锁被拒:未校准、机体不平或电压不足 */
    FC_WHY_TOUCHDOWN = 8        /* 降落完成 */
} fc_reason;

typedef struct {
    fc_float link_timeout;      /* s,多久收不到任何帧算链路丢失 */
    fc_float v_low, v_low_time; /* 低电压阈值 (V) 和持续时间 (s) */
    fc_float flip_cos, flip_time;   /* 机体 z 轴与竖直方向夹角的余弦低于此值并持续这么久,算翻倒 */
    fc_float fence_xy, fence_z; /* m,允许飞行的范围 */
    fc_float land_speed;        /* m/s,降落的下降速度 */
    fc_float land_height, land_time;    /* 高度低于此值并持续这么久,算触地 */
    fc_float arm_min_cos;       /* 解锁时机体要基本水平 */
} fc_fsm_cfg;

typedef struct {
    fc_fsm_cfg cfg;
    fc_mode mode;
    fc_reason reason;           /* 最近一次状态变化(或解锁被拒)的原因 */
    fc_float t_link, t_batt, t_flip, t_ground;
    fc_float land_pos[3];       /* 降落时的目标点,z 逐步下降 */
} fc_fsm;

typedef struct {
    int cmd_arm, cmd_disarm, cmd_land;
    int link_rx;                /* 这一拍收到了合法的帧 */
    int calibrated, est_ok;
    fc_float vbat, cos_tilt;
    fc_float pos[3];
} fc_fsm_in;

void fc_fsm_default_cfg(fc_fsm_cfg *c);
void fc_fsm_init(fc_fsm *f, const fc_fsm_cfg *c);
void fc_fsm_step(fc_fsm *f, const fc_fsm_in *in, fc_float dt);
static inline int fc_fsm_motors_on(const fc_fsm *f) { return f->mode == FC_MODE_FLYING || f->mode == FC_MODE_LANDING; }

#endif
