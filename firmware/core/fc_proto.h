/* 串口通信协议(飞控 ↔ ESP32 ↔ 地面)。
 *
 * 帧格式:  0xA5 | 长度 n | 类型 | 载荷 n 字节 | CRC16 低字节 | CRC16 高字节
 * CRC16-CCITT(初值 0xFFFF,多项式 0x1021)覆盖"长度、类型、载荷"。
 * 载荷里的数都是小端 32 位浮点,与芯片内部用单精度还是双精度无关。
 *
 * 解析器逐字节喂入,不分配内存。它保留最近一帧长度的字节:校验失败时只丢掉一个字节、
 * 从下一个字节重新找帧头,所以乱码里碰巧出现的 0xA5 不会连带吃掉后面真正的帧。
 */
#ifndef FC_PROTO_H
#define FC_PROTO_H

#include <stdint.h>

#define FC_PROTO_SOF 0xA5
#define FC_PROTO_MAX_PAYLOAD 64
#define FC_PROTO_MAX_FRAME (FC_PROTO_MAX_PAYLOAD + 5)

enum {
    FC_MSG_HEARTBEAT = 1,   /* 地面 → 飞控,无载荷。飞控靠它判断链路是否还在 */
    FC_MSG_ARM = 2,         /* 解锁 */
    FC_MSG_DISARM = 3,      /* 立即上锁(电机停转) */
    FC_MSG_LAND = 4,        /* 原地降落 */
    FC_MSG_SETPOINT = 5,    /* 位置 3、速度 3、加速度 3、偏航 1,共 10 个浮点 */
    FC_MSG_TELEMETRY = 6    /* 飞控 → 地面:模式 1 字节、原因 1 字节,电压 1、位置 3、速度 3、四元数 4 个浮点 */
};

typedef struct {
    uint8_t type, len;
    uint8_t payload[FC_PROTO_MAX_PAYLOAD];
} fc_msg;

typedef struct {
    uint8_t buf[FC_PROTO_MAX_FRAME];
    uint8_t n;                        /* buf 里的字节数 */
    uint32_t n_ok, n_bad;             /* 收到的好帧 / 被丢弃的假帧头数 */
} fc_proto_parser;

void fc_proto_init(fc_proto_parser *p);
/* 喂一个字节;凑齐一帧且校验通过时返回 1,此时 *out 是这一帧 */
int fc_proto_feed(fc_proto_parser *p, uint8_t byte, fc_msg *out);
/* 组帧,返回帧的总字节数(len 超限时返回 0) */
int fc_proto_encode(uint8_t type, const uint8_t *payload, uint8_t len, uint8_t *out);
uint16_t fc_proto_crc16(const uint8_t *data, int n, uint16_t crc);

void fc_proto_put_f32(uint8_t *buf, float v);
float fc_proto_get_f32(const uint8_t *buf);

#endif
