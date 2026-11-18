#include "fc_proto.h"

#include <string.h>

static uint16_t crc_step(uint16_t crc, uint8_t b) {
    int i;
    crc ^= (uint16_t)b << 8;
    for (i = 0; i < 8; i++) crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    return crc;
}

uint16_t fc_proto_crc16(const uint8_t *data, int n, uint16_t crc) {
    int i;
    for (i = 0; i < n; i++) crc = crc_step(crc, data[i]);
    return crc;
}

void fc_proto_init(fc_proto_parser *p) {
    p->n = 0;
    p->n_ok = 0;
    p->n_bad = 0;
}

static void drop(fc_proto_parser *p, int k) {
    p->n = (uint8_t)(p->n - k);
    memmove(p->buf, p->buf + k, p->n);
}

int fc_proto_feed(fc_proto_parser *p, uint8_t byte, fc_msg *out) {
    if (p->n < FC_PROTO_MAX_FRAME) p->buf[p->n++] = byte;
    for (;;) {
        int k = 0, len, total;
        uint16_t crc;
        while (k < p->n && p->buf[k] != FC_PROTO_SOF) k++;      /* 丢掉帧头之前的字节 */
        if (k) drop(p, k);
        if (p->n < 2) return 0;
        len = p->buf[1];
        if (len > FC_PROTO_MAX_PAYLOAD) { p->n_bad++; drop(p, 1); continue; }   /* 长度不合法:假帧头 */
        total = len + 5;
        if (p->n < total) return 0;                              /* 还没收全 */
        crc = fc_proto_crc16(&p->buf[1], len + 2, 0xFFFF);
        if (p->buf[3 + len] == (crc & 0xFF) && p->buf[4 + len] == (crc >> 8)) {
            out->len = (uint8_t)len;
            out->type = p->buf[2];
            if (len) memcpy(out->payload, &p->buf[3], (size_t)len);
            drop(p, total);
            p->n_ok++;
            return 1;
        }
        p->n_bad++;                                              /* 校验不过:只丢一个字节,重新找 */
        drop(p, 1);
    }
}

int fc_proto_encode(uint8_t type, const uint8_t *payload, uint8_t len, uint8_t *out) {
    uint16_t crc;
    if (len > FC_PROTO_MAX_PAYLOAD) return 0;
    out[0] = FC_PROTO_SOF;
    out[1] = len;
    out[2] = type;
    if (len) memcpy(&out[3], payload, len);
    crc = fc_proto_crc16(&out[1], len + 2, 0xFFFF);
    out[3 + len] = (uint8_t)(crc & 0xFF);
    out[4 + len] = (uint8_t)(crc >> 8);
    return len + 5;
}

/* STM32 和 PC 都是小端、IEEE 754,直接按字节拷贝 */
void fc_proto_put_f32(uint8_t *buf, float v) { memcpy(buf, &v, 4); }
float fc_proto_get_f32(const uint8_t *buf) { float v; memcpy(&v, buf, 4); return v; }
