"""飞控板 V1 原理图(1S 有刷微型四旋翼)。

运行:  python hardware/fc_v1.py
生成:  hardware/fc_v1/fc_v1.kicad_sch,并调用 kicad-cli 跑 ERC、导出 PDF/SVG。

这是第一版草稿:器件引脚以 KiCad 符号库为准,外围电路的阻容值和接法是按常规做法写的,
下单之前必须逐个对照数据手册核对(见 hardware/README.md 的清单)。
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from schgen import KICAD_SHARE, Schematic  # noqa: E402

OUT = Path(__file__).parent / "fc_v1"
CLI = KICAD_SHARE.parents[1] / "bin" / "kicad-cli.exe"

C0402, C0603, C1206 = (f"Capacitor_SMD:C_{s}" for s in ("0402_1005Metric", "0603_1608Metric", "1206_3216Metric"))
R0402 = "Resistor_SMD:R_0402_1005Metric"
LED0603 = "LED_SMD:LED_0603_1608Metric"
JST_SH8 = "Connector_JST:JST_SH_SM08B-SRSS-TB_1x08-1MP_P1.00mm_Horizontal"

s = Schematic("fc_v1", "Micro quad flight controller V1 (1S brushed)")


def cap(ref, value, x, y, top, bot="GND", fp=C0402):
    s.place("Device:C", ref, value, x, y, {"1": top, "2": bot}, footprint=fp)


def res(ref, value, x, y, top, bot):
    s.place("Device:R", ref, value, x, y, {"1": top, "2": bot}, footprint=R0402)


# ============================ 电源:电池 → 开关 → 升降压 3.3 V ============================
s.text("电源:1S 锂电 → TPS63001 升降压 → 3.3 V(开关只控制使能脚,不走大电流)", 17.78, 22.86)
s.place("Connector_Generic:Conn_01x02", "J1", "BATT 1S", 33.02, 43.18,
        {"1": "+BATT", "2": "GND"},
        footprint="Connector_Wire:SolderWire-0.25sqmm_1x02_P4.2mm_D0.65mm_OD1.7mm")   # 焊电池引线(带插头的尾线)
s.place("Switch:SW_SPDT", "SW1", "POWER", 55.88, 43.18, {"A": "REG_VINA", "B": "REG_EN", "C": "GND"},
        footprint="Button_Switch_SMD:SW_SPDT_PCM12")
s.place("Regulator_Switching:TPS63001", "U2", "TPS63001", 101.6, 48.26, {
    "VIN": "+BATT", "VINA": "REG_VINA", "EN": "REG_EN", "PS/SYNC": "REG_VINA",   # PS/SYNC 接高:关闭省电模式,纹波小
    "L1": "REG_L1", "L2": "REG_L2", "VOUT": "+3V3", "FB": "+3V3",          # 固定 3.3 V 版本:FB 直接接输出
    "GND": "GND", "PGND": "GND"}, footprint="Package_SON:Texas_DRC0010J")   # 不带 0.2 mm 散热过孔的版本,普通工艺可做
s.place("Device:L", "L1", "2.2uH Isat>=2A", 73.66, 76.2, {"1": "REG_L1", "2": "REG_L2"},
        footprint="Inductor_SMD:L_1008_2520Metric")
cap("C1", "10uF", 88.9, 86.36, "+BATT", fp=C0603)
cap("C2", "100nF", 99.06, 86.36, "REG_VINA")
res("R13", "100R", 129.54, 86.36, "+BATT", "REG_VINA")       # 手册典型电路:VIN 经 100 Ω + 100 nF 滤波后给 VINA
cap("C3", "10uF", 109.22, 86.36, "+3V3", fp=C0603)
cap("C4", "10uF", 119.38, 86.36, "+3V3", fp=C0603)
# 电池电压检测:1/2 分压进 ADC
res("R1", "100k", 132.08, 38.1, "+BATT", "VBAT_SENSE|nolabel")
res("R2", "100k", 132.08, 50.8, "VBAT_SENSE|right", "GND")
cap("C5", "100nF", 149.86, 50.8, "VBAT_SENSE")

# ============================ USB-C 与充电 ============================
s.text("USB-C 与充电 MCP73831(约 250 mA)", 17.78, 104.14)
s.place("Connector:USB_C_Receptacle_USB2.0_16P", "J2", "USB-C", 33.02, 137.16, {
    "VBUS": "VBUS", "GND": "GND", "SHIELD": "GND", "CC1": "USB_CC1", "CC2": "USB_CC2",
    "D+": "USB_DP", "D-": "USB_DM"}, nc=("SBU1", "SBU2"),
    footprint="Connector_USB:USB_C_Receptacle_HRO_TYPE-C-31-M-12")
res("R3", "5.1k", 68.58, 139.7, "USB_CC1", "GND")
res("R4", "5.1k", 78.74, 139.7, "USB_CC2", "GND")
s.place("Battery_Management:MCP73831-2-OT", "U3", "MCP73831-2", 106.68, 129.54, {
    "V_{DD}": "VBUS", "V_{SS}": "GND", "V_{BAT}": "+BATT|long", "STAT": "CHG_STAT", "PROG": "CHG_PROG"})
res("R5", "3.9k", 88.9, 160.02, "CHG_PROG", "GND")           # I = 1000 V / R ≈ 256 mA
cap("C6", "4.7uF", 99.06, 160.02, "VBUS", fp=C0603)
cap("C7", "4.7uF", 109.22, 160.02, "+BATT", fp=C0603)
res("R6", "1k", 58.42, 160.02, "VBUS", "CHG_LED_A")
s.place("Device:LED", "D1", "CHG", 78.74, 182.88, {"A": "CHG_LED_A", "K": "CHG_STAT"}, footprint=LED0603)
res("R7", "10k", 119.38, 160.02, "VBUS", "USB_VBUS_SENSE|nolabel")       # VBUS 检测,分压到 3.3 V 以内
res("R8", "20k", 119.38, 172.72, "USB_VBUS_SENSE|right", "GND")

# ============================ 主控 STM32F405RG ============================
s.text("主控 STM32F405RGT6(168 MHz,带 FPU)", 157.48, 22.86)
s.place("MCU_ST_STM32F4:STM32F405RGTx", "U1", "STM32F405RGT6", 205.74, 142.24, {
    "VBAT": "+3V3", "VDD": "+3V3", "VDDA": "+3V3A", "VSS": "GND", "VSSA": "GND",
    "VCAP_1": "VCAP_1", "VCAP_2": "VCAP_2", "NRST": "NRST", "BOOT0": "BOOT0",
    "PH0": "OSC_IN", "PH1": "OSC_OUT",
    "PA0": "EXT_TX", "PA1": "EXT_RX",                        # UART4:外部定位/摄像头模块
    "PA2": "ESP_RX", "PA3": "ESP_TX",                        # USART2:ESP32-C3 无线桥
    "PA4": "IMU_CS_ACC", "PA5": "SPI1_SCK", "PA6": "SPI1_MISO", "PA7": "SPI1_MOSI",
    "PA8": "ESP_BOOT", "PA9": "USB_VBUS_SENSE", "PA10": "PA10_PD", "PA11": "USB_DM", "PA12": "USB_DP",
    "PA13": "SWDIO", "PA14": "SWCLK",
    "PB0": "IMU_INT_GYR", "PB1": "EXT_IO", "PB2": "BOOT1", "PB3": "SWO",
    "PB4": "LED1", "PB5": "LED2", "PB6": "I2C1_SCL", "PB7": "I2C1_SDA",
    "PB12": "FLASH_CS", "PB13": "SPI2_SCK", "PB14": "SPI2_MISO", "PB15": "SPI2_MOSI",
    "PC0": "FLOW_CS", "PC2": "VBAT_SENSE", "PC3": "ESP_EN",
    "PC12": "FLOW_UTX", "PD2": "FLOW_URX", "PB10": "BOOST_EN",     # UART5:串口类光流模块;PB10 控制 5 V 升压
    "PC4": "IMU_CS_GYR", "PC5": "IMU_INT_ACC",
    "PC6": "MOTOR1_PWM", "PC7": "MOTOR2_PWM", "PC8": "MOTOR3_PWM", "PC9": "MOTOR4_PWM",   # TIM3 CH1-4
}, nc=("PA15", "PB8", "PB9", "PB11", "PC1", "PC10", "PC11", "PC13", "PC14", "PC15"))
for i in range(4):
    cap(f"C{10 + i}", "100nF", 162.56 + 10.16 * i, 50.8, "+3V3")
cap("C14", "4.7uF", 203.2, 50.8, "+3V3", fp=C0603)
s.place("Device:FerriteBead", "FB1", "600R@100MHz", 215.9, 50.8, {"1": "+3V3", "2": "+3V3A"},
        footprint="Inductor_SMD:L_0603_1608Metric")
cap("C15", "1uF", 226.06, 50.8, "+3V3A")
cap("C16", "100nF", 236.22, 50.8, "+3V3A")
cap("C22", "100nF", 246.38, 50.8, "+3V3")            # VBAT 脚(1 脚)专用
cap("C17", "2.2uF", 139.7, 185.42, "VCAP_1")
cap("C18", "2.2uF", 152.4, 185.42, "VCAP_2")
cap("C19", "100nF", 137.16, 91.44, "NRST")
res("R9", "10k", 149.86, 91.44, "BOOT0", "GND")
res("R10", "10k", 162.56, 91.44, "BOOT1", "GND")
res("R14", "10k", 175.26, 91.44, "PA10_PD", "GND")       # PA10 是下载程序的串口输入脚,不能悬空
s.place("Switch:SW_Push", "SW2", "BOOT", 149.86, 71.12, {"1": "+3V3", "2": "BOOT0"},
        footprint="Button_Switch_SMD:SW_Push_1P1T_NO_CK_KMR2")
s.place("Device:Crystal_GND24", "Y1", "8MHz CL=10pF", 149.86, 116.84, {"1": "OSC_IN", "3": "OSC_OUT", "G": "GND"},
        footprint="Crystal:Crystal_SMD_3225-4Pin_3.2x2.5mm")
cap("C20", "12pF", 137.16, 139.7, "OSC_IN")
cap("C21", "12pF", 162.56, 139.7, "OSC_OUT")
res("R11", "1k", 251.46, 190.5, "LED1", "LED1_A")
res("R12", "1k", 271.78, 190.5, "LED2", "LED2_A")
s.place("Device:LED", "D2", "BLUE", 251.46, 210.82, {"A": "LED1_A", "K": "GND"}, footprint=LED0603)
s.place("Device:LED", "D3", "RED", 271.78, 210.82, {"A": "LED2_A", "K": "GND"}, footprint=LED0603)

# ============================ 传感器 ============================
s.text("IMU BMI088(SPI1)", 269.24, 22.86)
s.place("Sensor_Motion:BMI088", "U4", "BMI088", 294.64, 50.8, {
    "~{CSB1}": "IMU_CS_ACC", "~{CSB2}": "IMU_CS_GYR", "SCK/SCL": "SPI1_SCK", "SDI/SDA": "SPI1_MOSI",
    "SDO1": "SPI1_MISO", "SDO2": "SPI1_MISO", "PS": "GND",          # PS 接地 = SPI 模式
    "INT1": "IMU_INT_ACC", "INT3": "IMU_INT_GYR",
    "VDD": "+3V3", "VDDIO": "+3V3", "GNDA": "GND", "GNDIO": "GND"}, nc=("INT2", "INT4"))
cap("C30", "100nF", 325.12, 48.26, "+3V3")
cap("C31", "100nF", 335.28, 48.26, "+3V3")

s.text("气压计 BMP280(I2C1,地址 0x76)", 269.24, 83.82)
s.place("Sensor_Pressure:BMP280", "U5", "BMP280", 294.64, 104.14, {
    "CSB": "+3V3", "SDO": "GND", "SDI": "I2C1_SDA", "SCK": "I2C1_SCL",   # CSB 接高 = I2C;SDO 接地 = 0x76
    "VDD": "+3V3", "VDDIO": "+3V3", "GND": "GND"})
cap("C32", "100nF", 322.58, 104.14, "+3V3")           # VDD
cap("C36", "100nF", 312.42, 104.14, "+3V3")           # VDDIO(手册要求两个电源脚各一个)
res("R20", "2.2k", 332.74, 104.14, "+3V3", "I2C1_SCL")
res("R21", "2.2k", 342.9, 104.14, "+3V3", "I2C1_SDA")

s.text("日志 Flash W25Q128(SPI2)", 355.6, 22.86)
s.place("Memory_Flash:W25Q128JVS", "U7", "W25Q128JVS", 381.0, 50.8, {
    "~{CS}": "FLASH_CS", "CLK": "SPI2_SCK", "DI/IO_{0}": "SPI2_MOSI", "DO/IO_{1}": "SPI2_MISO",
    "~{WP}/IO_{2}": "FLASH_WP", "~{HOLD}/~{RESET}/IO_{3}": "FLASH_HOLD", "VCC": "+3V3", "GND": "GND"})
res("R33", "10k", 325.12, 205.74, "+3V3", "FLASH_CS|right")    # 上电和复位期间保持 Flash 不被选中
res("R35", "10k", 337.82, 205.74, "+3V3", "FLASH_WP|right")    # 用上拉而不是直连:IQ 型号这两个脚可能变成输出
res("R36", "10k", 350.52, 205.74, "+3V3", "FLASH_HOLD|right")
res("R34", "10k", 363.22, 205.74, "+3V3", "ESP_BOOT|right")    # IO9 启动配置脚,保证正常启动
cap("C35", "100nF", 403.86, 48.26, "+3V3")

s.text("无线 ESP32-C3(串口桥)", 355.6, 83.82)
s.place("RF_Module:ESP32-C3-WROOM-02", "U8", "ESP32-C3-WROOM-02", 378.46, 119.38, {
    "3V3": "+3V3", "GND": "GND", "EN": "ESP_EN", "IO9": "ESP_BOOT", "IO8": "ESP_IO8", "IO2": "ESP_IO2",
    "IO21/TXD": "ESP_TX", "IO20/RXD": "ESP_RX"},
    nc=("IO0", "IO1", "IO3", "IO4", "IO5", "IO6", "IO7", "IO10", "IO18", "IO19"))
cap("C40", "10uF", 355.6, 180.34, "+3V3", fp=C0603)
cap("C41", "100nF", 365.76, 180.34, "+3V3")
res("R30", "10k", 375.92, 180.34, "+3V3", "ESP_EN|nolabel")
cap("C42", "1uF", 375.92, 193.04, "ESP_EN|right")
res("R31", "10k", 388.62, 180.34, "+3V3", "ESP_IO8")           # 启动配置脚,需上拉
res("R32", "10k", 398.78, 180.34, "+3V3", "ESP_IO2")           # 启动配置脚,需上拉

# ============================ 电机驱动 ×4 ============================
s.text("有刷电机驱动 ×4:N 沟道 MOS 低边开关 + 续流二极管 + 栅极下拉", 17.78, 215.9)
for i in range(4):
    n, x = i + 1, 35.56 + 60.96 * i
    s.place("Transistor_FET:AO3400A", f"Q{n}", "AO3400A", x, 254.0,
            {"G": f"MOTOR{n}_PWM", "D": f"MOTOR{n}_N", "S": "GND"})
    res(f"R{40 + n}", "10k", x + 17.78, 261.62, f"MOTOR{n}_PWM", "GND")
    s.place("Device:D_Schottky", f"D{10 + n}", "SS14-class", x + 25.4, 233.68,
            {"K": "+BATT", "A": f"MOTOR{n}_N"}, footprint="Diode_SMD:D_SOD-323")
    s.place("Connector_Generic:Conn_01x02", f"J{2 + n}", f"MOTOR{n}", x + 38.1, 254.0,
            {"1": "+BATT", "2": f"MOTOR{n}_N"},
            footprint="Connector_Wire:SolderWire-0.1sqmm_1x02_P3.6mm_D0.4mm_OD1mm")      # 电机线直接焊在臂上
    cap(f"C{51 + n}", "10uF", x + 45.72, 238.76, "+BATT", fp=C0603)   # 每条臂上的本地储能:开关瞬间的脉冲电流就地取
cap("C50", "47uF", 30.48, 78.74, "+BATT", fp=C1206)
cap("C51", "47uF", 43.18, 78.74, "+BATT", fp=C1206)

# ============================ 接口 ============================
s.text("调试 SWD", 274.32, 220.98)
s.place("Connector_Generic:Conn_01x06", "J7", "SWD", 289.56, 236.22,
        {"1": "+3V3", "2": "SWDIO", "3": "SWCLK", "4": "SWO", "5": "NRST", "6": "GND"},
        footprint="Connector_PinHeader_1.27mm:PinHeader_1x06_P1.27mm_Vertical_SMD_Pin1Left")
s.text("扩展口", 304.8, 220.98)
s.place("Connector_Generic:Conn_01x08", "J8", "EXT", 320.04, 236.22,
        {"1": "+BATT", "2": "+3V3", "3": "EXT_TX", "4": "EXT_RX", "5": "I2C1_SCL", "6": "I2C1_SDA",
         "7": "EXT_IO", "8": "GND"}, footprint=JST_SH8)
# 外接光流/测距模块的两种接口,型号未定,两类都能直接插:
#   J9  SPI + I2C、3.3 V 供电(Bitcraze Flow deck v2 这一类:PMW3901 走 SPI,VL53L1X 走 I2C)
#   J10 串口、5 V 供电(Matek 3901-L0X、MicoAir MTF-01 这一类,串口电平 3.3 V)
s.text("模块口 A:SPI + I2C", 340.36, 220.98, size=2.0)
s.place("Connector_Generic:Conn_01x08", "J9", "FLOW SPI/I2C", 360.68, 236.22,
        {"1": "+3V3", "2": "FLOW_SCK", "3": "FLOW_MISO", "4": "FLOW_MOSI", "5": "FLOW_CS", "6": "I2C1_SCL",
         "7": "I2C1_SDA", "8": "GND"}, footprint=JST_SH8)
s.text("口 B:串口 + 5 V", 383.54, 220.98, size=2.0)
s.place("Connector_Generic:Conn_01x04", "J10", "FLOW UART", 401.32, 236.22,
        {"1": "+5V", "2": "FLOW_UTX", "3": "FLOW_URX", "4": "GND"},
        footprint="Connector_JST:JST_SH_SM04B-SRSS-TB_1x04-1MP_P1.00mm_Horizontal")
# SPI2 与板上的 Flash 共用。接口这一段经 33 Ω 串联电阻引出,外接线缆的反射和负载不直接挂在 Flash 的总线上
res("R37", "33R", 302.26, 185.42, "SPI2_SCK", "FLOW_SCK|right")
res("R38", "33R", 314.96, 185.42, "SPI2_MISO", "FLOW_MISO|right")
res("R39", "33R", 327.66, 185.42, "SPI2_MOSI", "FLOW_MOSI|right")

# ============================ 给串口类模块的 5 V 升压 ============================
s.text("5 V 升压 TPS61222(只给口 B 的模块用,默认关闭)", 269.24, 132.08)
s.place("Regulator_Switching:TPS61222DCK", "U9", "TPS61222", 294.64, 152.4, {
    "VIN": "+BATT", "L": "BOOST_L", "EN": "BOOST_EN",          # EN 不能悬空:下拉默认关,由主控拉高打开
    "VOUT": "+5V", "FB": "+5V|long", "GND": "GND"})             # 固定 5 V 版本:FB 接 VOUT
s.place("Device:L", "L2", "4.7uH Isat>=0.5A", 320.04, 152.4, {"1": "+BATT", "2": "BOOST_L"},
        footprint="Inductor_SMD:L_1008_2520Metric")
cap("C43", "10uF", 330.2, 152.4, "+BATT", fp=C0603)
cap("C44", "10uF", 340.36, 152.4, "+5V", fp=C0603)
res("R15", "100k", 350.52, 152.4, "BOOST_EN", "GND")

# ============================ 电源标志(ERC 用) ============================
s.text("PWR_FLAG:仅供 ERC 识别电源来源", 17.78, 187.96, size=1.8)
for k, net in enumerate(sorted(s.undriven_power_nets())):
    s.flag(net, 22.86 + 25.4 * (k % 5), 198.12 + 12.7 * (k // 5))


def main():
    OUT.mkdir(exist_ok=True)
    sch = OUT / "fc_v1.kicad_sch"
    s.write(sch)
    pro = OUT / "fc_v1.kicad_pro"
    if not pro.exists():
        pro.write_text('{\n  "meta": {\n    "filename": "fc_v1.kicad_pro",\n    "version": 3\n  }\n}\n',
                       encoding="utf-8")
    nets = {k: v for k, v in s.net_pins.items()}
    import json
    (OUT / "netlist.json").write_text(json.dumps({
        "parts": s.parts,
        "nets": {k: [[r, n] for r, n, _ in v if not r.startswith("#")] for k, v in nets.items()},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"器件 {len(s.refs)} 个,网络 {len(nets)} 个 → {sch}")
    if not CLI.exists():
        print("没找到 kicad-cli,跳过 ERC。")
        return
    for args in (["sch", "erc", "--severity-all", "--exit-code-violations", "-o", str(OUT / "erc.rpt"), str(sch)],
                 ["sch", "export", "pdf", "-o", str(OUT / "fc_v1.pdf"), str(sch)],
                 ["sch", "export", "svg", "-o", str(OUT), str(sch)],
                 ["sch", "export", "netlist", "-o", str(OUT / "fc_v1.net"), str(sch)]):
        r = subprocess.run([str(CLI), *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(f"kicad-cli {' '.join(args[:3])}: 返回 {r.returncode}  {(r.stdout + r.stderr).strip()[-300:]}")


if __name__ == "__main__":
    main()
