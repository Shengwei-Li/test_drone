"""通信协议的地面端(Python)。帧格式见 firmware/core/fc_proto.h。

这份实现是独立写的,没有调用 C 库 —— 两边能互相解出对方的帧,本身就是对协议的一次交叉检查。
"""
import struct
from dataclasses import dataclass

import numpy as np

SOF, MAX_PAYLOAD = 0xA5, 64
HEARTBEAT, ARM, DISARM, LAND, SETPOINT, TELEMETRY = 1, 2, 3, 4, 5, 6
MODES = {0: "上锁", 1: "飞行", 2: "降落", 3: "故障"}
REASONS = {0: "无", 1: "指令", 2: "链路丢失", 3: "低电压", 4: "越界", 5: "翻倒", 6: "估计器失效",
           7: "未就绪", 8: "触地"}


def crc16(data: bytes, crc: int = 0xFFFF) -> int:
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def encode(msg_type: int, payload: bytes = b"") -> bytes:
    assert len(payload) <= MAX_PAYLOAD
    body = bytes([len(payload), msg_type]) + payload
    return bytes([SOF]) + body + struct.pack("<H", crc16(body))


def setpoint(pos, vel=(0, 0, 0), acc=(0, 0, 0), yaw: float = 0.0) -> bytes:
    return encode(SETPOINT, struct.pack("<10f", *pos, *vel, *acc, yaw))


@dataclass
class Telemetry:
    mode: int
    reason: int
    vbat: float
    pos: np.ndarray
    vel: np.ndarray
    quat: np.ndarray


def parse_telemetry(payload: bytes) -> Telemetry:
    mode, reason = payload[0], payload[1]
    f = struct.unpack("<11f", payload[2:46])
    return Telemetry(mode, reason, f[0], np.array(f[1:4]), np.array(f[4:7]), np.array(f[7:11]))


class Parser:
    """逐字节解析;feed 返回这批字节里解出的 (类型, 载荷) 列表。"""

    def __init__(self):
        self.buf = bytearray()
        self.n_bad = 0

    def feed(self, data: bytes):
        self.buf += data
        out = []
        while True:
            i = self.buf.find(bytes([SOF]))
            if i < 0:
                self.buf.clear()
                break
            del self.buf[:i]
            if len(self.buf) < 3:
                break
            n = self.buf[1]
            if n > MAX_PAYLOAD:
                self.n_bad += 1
                del self.buf[:1]
                continue
            if len(self.buf) < n + 5:
                break
            body, crc = bytes(self.buf[1:3 + n]), struct.unpack("<H", self.buf[3 + n:5 + n])[0]
            if crc16(body) == crc:
                out.append((body[1], body[2:]))
                del self.buf[:n + 5]
            else:
                self.n_bad += 1
                del self.buf[:1]
        return out
