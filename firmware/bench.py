"""固件核心库各函数的运算量:在电脑上实测耗时,并按乘加次数粗估 STM32F405 上的耗时。

用法:  python -m firmware.bench

STM32 的数字是估算,不是实测:假设每次"乘加 + 循环开销"约 10 个时钟周期、主频 168 MHz、
tanh 每次约 300 周期。真实值要等板子到手(或用 Renode 这类指令级仿真器)再量。
"""
import ctypes
import time

import numpy as np

from firmware.sil import CController, CEskf, CPolicy, load
from sim.params import QuadParams
from sim.rl_policy import DEFAULT_POLICY

CYCLES_PER_MAC, F_CPU, CYCLES_TANH = 10, 168e6, 300
NE = 15
# (名称, 基准编号, 调用频率 Hz, 乘加次数, tanh 次数)
ITEMS = [
    ("ESKF 预测", 0, 500, 2 * NE**3 + 200, 0),
    ("ESKF 测距更新", 1, 50, 2 * NE**3 + 4 * NE**2, 0),
    ("ESKF 光流更新", 2, 100, 2 * NE**3 + 8 * NE**2, 0),
    ("INDI 控制器", 3, 500, 400, 0),
    ("策略网络推理", 4, 100, 28 * 64 + 64 * 64 + 64 * 4, 128),
]


def main():
    p = QuadParams()
    CEskf(p, precision="f32")
    CController(p, 0.002, True, "f32")
    CPolicy(p, 0.002, precision="f32")
    lib = load("f32")[0]
    lib.fc_sil_bench.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.fc_sil_bench.restype = None
    lib.fc_sil_state_bytes.restype = ctypes.c_int
    print(f"{'函数':<12}{'电脑实测 µs':>12}{'乘加次数':>10}{'STM32 估算 µs':>14}{'频率 Hz':>9}{'CPU 占用估算':>12}")
    total = 0.0
    for name, which, hz, mac, tanh in ITEMS:
        n = 20000
        lib.fc_sil_bench(which, 100)
        t0 = time.perf_counter()
        lib.fc_sil_bench(which, n)
        pc = (time.perf_counter() - t0) / n * 1e6
        mcu = (mac * CYCLES_PER_MAC + tanh * CYCLES_TANH) / F_CPU * 1e6
        load_pct = mcu * 1e-6 * hz * 100
        total += load_pct
        print(f"{name:<12}{pc:>12.2f}{mac:>10d}{mcu:>14.0f}{hz:>9d}{load_pct:>11.1f}%")
    print(f"{'合计':<12}{'':>12}{'':>10}{'':>14}{'':>9}{total:>11.1f}%")
    weights = sum(v.size for v in np.load(DEFAULT_POLICY).values())
    print(f"\n内存:估计器 + 控制器 + 策略的状态共 {lib.fc_sil_state_bytes() / 1024:.1f} KB(单精度),"
          f"其中策略权重 {weights * 4 / 1024:.1f} KB 可放在 Flash。")
    print("STM32F405RG 有 192 KB RAM、1 MB Flash。")


if __name__ == "__main__":
    main()
