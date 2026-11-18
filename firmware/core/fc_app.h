/* 固件主循环:把估计器、状态机、控制器和通信串起来。
 *
 * 板级代码(驱动层)每个控制周期调用一次 fc_app_step:
 *   填好这一拍的传感器读数和串口收到的字节 → 拿回四个电机的转速指令和要发出去的字节。
 * 这个函数本身不碰任何外设,所以可以原样放进电脑上的仿真里跑。
 */
#ifndef FC_APP_H
#define FC_APP_H

#include "fc_ctrl.h"
#include "fc_eskf.h"
#include "fc_fsm.h"
#include "fc_policy.h"
#include "fc_proto.h"

#define FC_CALIB_SAMPLES 400          /* 上电后用这么多帧静止的陀螺读数标定零偏 */
#define FC_TELEMETRY_EVERY 10         /* 每 10 拍发一帧遥测(500 Hz 控制 → 50 Hz) */

typedef struct {
    fc_float gyro[3], acc[3];
    int has_flow, has_tof;
    fc_float flow[2], tof;
    fc_float vbat;
    const uint8_t *rx;                /* 这一拍串口收到的字节 */
    int rx_len;
} fc_inputs;

typedef struct {
    fc_float motor[4];                /* 转速指令 rad/s;上锁或故障时为 0 */
    uint8_t tx[FC_PROTO_MAX_FRAME];   /* 要发出去的字节 */
    int tx_len;
} fc_outputs;

typedef struct {
    fc_eskf est;
    fc_ctrl ctrl;
    fc_policy *policy;                /* 非空且 use_policy=1 时用策略网络代替 INDI */
    int use_policy;
    fc_fsm fsm;
    fc_proto_parser rx;
    fc_reference sp;                  /* 地面发来的最新目标 */
    fc_float dt, w_hover, w_max;
    /* 陀螺零偏校准的累加器 */
    int calibrated, calib_n;
    fc_float calib_sum[3], calib_sq[3];
    uint32_t tick;
    int policy_div;
    fc_float last_motor[4];
} fc_app;

void fc_app_init(fc_app *a, const fc_vehicle *veh, const fc_eskf_cfg *est_cfg, const fc_fsm_cfg *fsm_cfg,
                 fc_float dt, fc_float w_hover, fc_float w_max);
void fc_app_step(fc_app *a, const fc_inputs *in, fc_outputs *out);

#endif
