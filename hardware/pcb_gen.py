"""生成飞控板 V1 的 PCB 文件:板框 + 所有封装摆到位 + 网络(飞线),不含走线和铺铜。

运行(要用 KiCad 自带的 Python,因为要用它的 pcbnew 模块):
    D:\\kicad\\bin\\python.exe hardware\\pcb_gen.py
前置:先运行 python hardware/fc_v1.py(产生 netlist.json)和 python -m mech.model(产生板框)。
输出:hardware/fc_v1/fc_v1.kicad_pcb,以及 DRC 报告和渲染图。

坐标:下面的摆位表用"俯视图坐标"(u 向右,v 向下,单位 mm,原点在板中心,机头朝上即 −v)。
底面的器件也按俯视看到的位置写,脚本负责翻面。角度是逆时针为正。
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import pcbnew

HERE = Path(__file__).parent
OUT = HERE / "fc_v1"
SHARE = Path(pcbnew.__file__).resolve()
for _ in range(8):
    SHARE = SHARE.parent
    if (SHARE / "share" / "kicad" / "footprints").exists():
        break
FP_DIR = SHARE / "share" / "kicad" / "footprints"
CLI = SHARE / "bin" / "kicad-cli.exe"
CX, CY = 100.0, 100.0                     # 板中心在图纸上的位置
MM = pcbnew.FromMM

# ------------------------------------------------------------------ 摆位表
# 位号: (u, v, 角度, 面)   面:"F" 顶面,"B" 底面
P = {
    # 无线模块:天线一端伸出板前沿约 6 mm,天线下方和两侧没有板子
    "U8": (0.0, -12.6, 0, "F"),
    # 电源电容和使能脚的 RC 紧挨模块左上角的 1 脚(3V3)和 2 脚(EN)
    "C41": (-11.3, -18.6, 180, "F"), "C40": (-14.0, -18.6, 180, "F"),
    "R30": (-11.3, -17.1, 0, "F"), "C42": (-11.3, -15.9, 180, "F"),
    # 启动配置脚的上拉(不敏感)留在模块后沿
    "R31": (3.2, -4.6, 0, "F"), "R32": (5.4, -4.6, 0, "F"), "R34": (7.6, -4.6, 0, "F"),
    # IMU 在板正中;气压计在左侧
    "U4": (0.0, -0.6, 0, "F"), "C30": (-4.0, -0.6, 90, "F"), "C31": (4.0, -0.6, 90, "F"),
    "U5": (-9.0, -1.0, 0, "F"), "C32": (-9.0, 1.3, 0, "F"), "C36": (-6.3, -1.6, 90, "F"),
    "R20": (-12.3, -3.2, 0, "F"), "R21": (-12.3, -2.0, 0, "F"),
    # 主控转 180°:IMU 用的 SPI1 引脚朝前,USB 引脚朝左,晶振引脚朝右
    "U1": (0.0, 10.5, 180, "F"),
    # 每个电容对着它的引脚(主控转 180° 后的位置):
    #   前侧:31 脚 VCAP_1、32 脚 VDD、19 脚 VDD      左侧:47 脚 VCAP_2、48 脚 VDD
    #   右侧:13 脚 VDDA、1 脚 VBAT                    后侧:64 脚 VDD
    "C17": (-2.77, 2.6, 0, "F"), "C11": (-5.0, 2.6, 0, "F"), "C12": (2.5, 2.6, 0, "F"),
    "C18": (-7.5, 14.4, 90, "F"), "C10": (-7.5, 16.6, 90, "F"),
    "C16": (7.8, 8.3, 0, "F"), "C15": (7.8, 6.9, 0, "F"), "FB1": (8.9, 5.3, 0, "F"),
    "C13": (4.2, 18.3, 0, "F"), "C14": (-5.2, 18.3, 0, "F"), "C19": (6.6, 18.3, 0, "F"),
    "R9": (-8.0, 8.0, 90, "F"), "R10": (-8.0, 12.0, 90, "F"), "R14": (-8.0, 10.0, 90, "F"),
    # 晶振:输入焊盘在左下,它的电容 C20 放下面;输出焊盘在右上,电容 C21 放上面,两根线不交叉
    "Y1": (9.5, 12.2, 0, "F"), "C20": (8.6, 14.9, 90, "F"), "C21": (10.4, 9.45, 270, "F"),
    # 前左:日志 Flash。前右:BOOT 键和指示灯
    "U7": (-15.0, -10.5, 90, "F"), "C35": (-18.9, -14.1, 90, "F"),      # 电容在 8 脚 (VCC) 旁
    "R33": (-17.0, -4.6, 0, "F"), "R35": (-14.8, -4.6, 0, "F"), "R36": (-12.6, -4.6, 0, "F"),
    "SW2": (15.0, -8.0, 0, "F"),
    "D2": (15.0, -12.0, 0, "F"), "R11": (18.2, -12.0, 0, "F"),
    "D3": (15.0, -14.0, 0, "F"), "R12": (18.2, -14.0, 0, "F"),
    # 后右:升降压电源、开关、电池电压检测
    # 电源芯片转 180°:VIN、电感、VOUT 这一排功率引脚朝右(板边),电感贴着它们,远离晶振;
    # 输入电容在 VIN 脚上方,输出电容在 VOUT 脚下方,VINA 的滤波在左侧
    "U2": (13.5, 7.0, 180, "F"), "L1": (17.3, 7.0, 270, "F"),
    "C1": (14.6, 4.2, 180, "F"), "C3": (14.6, 9.8, 180, "F"), "C4": (14.6, 11.4, 180, "F"),
    "C2": (10.4, 7.2, 180, "F"), "R13": (11.0, 4.0, 0, "F"),
    "SW1": (15.5, 16.6, 0, "F"),
    "C5": (10.1, 16.1, 0, "F"), "R1": (10.1, 17.3, 0, "F"), "R2": (10.1, 18.5, 0, "F"),
    "C22": (7.6, 16.4, 90, "F"),
    "C50": (13.4, -4.2, 180, "F"), "C51": (13.4, -1.4, 180, "F"),      # 大电容紧挨电池焊盘,远离 IMU
    "J1": (18.3, -4.2, 270, "F"),
    # 后左:USB-C(开口朝左)和充电
    "J2": (-16.3, 10.5, 270, "F"),
    "R3": (-10.0, 6.5, 90, "F"), "R4": (-10.0, 14.5, 90, "F"),
    "U3": (-15.5, 18.0, 0, "F"), "R5": (-12.3, 17.2, 0, "F"),
    "C7": (-18.9, 18.0, 90, "F"), "C6": (-10.2, 17.4, 90, "F"),          # C7 在 3 脚 (VBAT) 旁
    "D1": (-15.0, 3.5, 0, "F"), "R6": (-18.3, 3.5, 0, "F"),
    "R7": (-11.5, 3.5, 0, "F"), "R8": (-9.2, 3.5, 0, "F"),
    # 底面:接口(手工焊或单独贴)
    "J7": (-5.6, 15.4, 90, "B"), "J8": (5.0, 16.3, 0, "B"), "J9": (12.5, -15.4, 0, "B"),
    "J10": (-12.5, -15.4, 0, "B"),
    # 模块口 A 的串联电阻:在顶面,正对底面的 J9
    "R37": (11.7, -15.3, 0, "F"), "R38": (11.7, -14.1, 0, "F"), "R39": (11.7, -12.9, 0, "F"),
    # 5 V 升压:放在左侧中部,离 IMU 约 16 mm,离天线和晶振也远
    "U9": (-15.8, 0.8, 0, "F"), "L2": (-15.8, -2.2, 0, "F"),
    "C43": (-18.6, 0.8, 270, "F"), "C44": (-13.0, 0.9, 90, "F"), "R15": (-18.6, -2.2, 90, "F"),
}
# 电机驱动放在臂上:沿臂方向排 [二极管 | 下拉电阻] → MOS 管 → 电机线焊盘
ARM_ANGLE = {1: 45, 2: 315, 3: 225, 4: 135}        # M1 右前、M2 右后、M3 左后、M4 左前(逆时针,从 +u 起)


def on_arm(n, d, lateral, rot):
    a = math.radians(ARM_ANGLE[n])
    return (d * math.cos(a) + lateral * math.sin(a), -d * math.sin(a) + lateral * math.cos(a),
            (ARM_ANGLE[n] + rot) % 360, "F")


for n in ARM_ANGLE:
    # MOS 管的栅极和源极在臂中线的 + 侧,漏极在 − 侧。下拉电阻跟栅极同侧、二极管跟漏极同侧,
    # 这样栅极线和电机负端的线各走一边,不交叉;储能电容夹在中间。
    P[f"R{40 + n}"] = on_arm(n, 30.4, 1.6, 0)
    P[f"C{51 + n}"] = on_arm(n, 30.4, -0.2, 0)
    P[f"D{10 + n}"] = on_arm(n, 30.4, -2.3, 0)
    P[f"Q{n}"] = on_arm(n, 33.9, 0.0, 90)
    u, v, rot, side = on_arm(n, 36.9, 0.0, 90)        # 电机座的插座从离中心 38.5 mm 处开始,焊孔要在它之内      # 焊线封装的原点在 1 号孔,挪半个孔距让两孔对称地跨在臂中线两侧
    phi = math.radians(rot)
    P[f"J{2 + n}"] = (u - 1.8 * math.cos(phi), v + 1.8 * math.sin(phi), rot, side)

PAD_ALIAS = {("J2", "S1"): "GND"}                   # USB 外壳焊盘在封装里叫 S1,原理图里叫 SH

NET_CLASSES = [
    # 名称, 线宽, 间距, 过孔外径, 过孔孔径, 匹配的网络
    ("Default", 0.15, 0.15, 0.6, 0.3, []),
    ("Power", 0.4, 0.15, 0.6, 0.3, ["+3V3", "+3V3A", "VBUS", "REG_L1", "REG_L2", "+5V", "BOOST_L"]),
    ("Motor", 1.0, 0.2, 0.8, 0.4, ["+BATT", "MOTOR*_N"]),
]


def to_page(u, v):
    return pcbnew.VECTOR2I(MM(CX + u), MM(CY + v))


def build():
    netlist = json.loads((OUT / "netlist.json").read_text(encoding="utf-8"))
    design = json.loads((HERE.parent / "mech" / "out" / "design.json").read_text(encoding="utf-8"))
    parts, nets = netlist["parts"], netlist["nets"]
    missing = sorted(set(parts) - set(P))
    extra = sorted(set(P) - set(parts))
    assert not missing and not extra, f"摆位表缺少 {missing},多出 {extra}"

    board = pcbnew.BOARD()
    board.SetCopperLayerCount(4)
    board.GetDesignSettings().SetBoardThickness(MM(1.0))

    # 板框:机架模型给的轮廓(机体系 x 前、y 左)→ 俯视图 (u, v) = (−y, −x)
    pts = [(-y, -x) for x, y in design["board_outline_mm"]]
    for (u1, v1), (u2, v2) in zip(pts, pts[1:] + pts[:1]):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(to_page(u1, v1))
        seg.SetEnd(to_page(u2, v2))
        seg.SetLayer(pcbnew.Edge_Cuts)
        seg.SetWidth(MM(0.1))
        board.Add(seg)

    # 臂端的定位孔(穿销子锁住电机座):在板框层画圆,板厂会当成内孔加工
    for x, y in design["pin_holes_mm"]:
        hole = pcbnew.PCB_SHAPE(board)
        hole.SetShape(pcbnew.SHAPE_T_CIRCLE)
        hole.SetCenter(to_page(-y, -x))
        hole.SetEnd(to_page(-y + design["pin_hole_diameter_mm"] / 2, -x))
        hole.SetLayer(pcbnew.Edge_Cuts)
        hole.SetWidth(MM(0.1))
        board.Add(hole)

    net_items = {}
    for name in nets:
        ni = pcbnew.NETINFO_ITEM(board, name)
        board.Add(ni)
        net_items[name] = ni
    pad_net = {(ref, pin): name for name, pins in nets.items() for ref, pin in pins}
    pad_net.update(PAD_ALIAS)

    unconnected_pads = []
    for ref, info in parts.items():
        lib, name = info["footprint"].split(":")
        fp = pcbnew.FootprintLoad(str(FP_DIR / f"{lib}.pretty"), name)
        assert fp is not None, info["footprint"]
        fp.SetReference(ref)
        fp.SetValue(info["value"])
        fp.SetFPID(pcbnew.LIB_ID(lib, name))
        fp.SetPath(pcbnew.KIID_PATH(f"/{info['uuid']}"))      # 关联到原理图符号,之后"从原理图更新 PCB"能对上
        u, v, rot, side = P[ref]
        board.Add(fp)
        fp.SetPosition(to_page(u, v))
        if side == "B":
            fp.Flip(to_page(u, v), pcbnew.FLIP_DIRECTION_LEFT_RIGHT)
        fp.SetOrientationDegrees(rot)
        for pad in fp.Pads():
            key = (ref, pad.GetNumber())
            if key in pad_net:
                pad.SetNet(net_items[pad_net[key]])
            elif pad.GetNumber() and pad.GetNumber() != "MP":
                unconnected_pads.append(key)
    pcb = OUT / "fc_v1.kicad_pcb"
    pcbnew.SaveBoard(str(pcb), board)
    return pcb, len(parts), len(nets), sorted(set(unconnected_pads))


def write_project():
    """工程文件:网络分类(线宽规则)和基本设计规则。"""
    classes = []
    patterns = []
    for name, width, clr, via_d, via_h, pats in NET_CLASSES:
        classes.append({"name": name, "track_width": width, "clearance": clr, "via_diameter": via_d,
                        "via_drill": via_h, "microvia_diameter": 0.3, "microvia_drill": 0.1,
                        "diff_pair_width": 0.2, "diff_pair_gap": 0.2, "diff_pair_via_gap": 0.25,
                        "bus_width": 12, "wire_width": 6, "line_style": 0,
                        "pcb_color": "rgba(0, 0, 0, 0.000)", "schematic_color": "rgba(0, 0, 0, 0.000)"})
        patterns += [{"netclass": name, "pattern": p} for p in pats]
    pro = {
        "meta": {"filename": "fc_v1.kicad_pro", "version": 3},
        "board": {"design_settings": {"rules": {
            "min_clearance": 0.127, "min_track_width": 0.127, "min_via_diameter": 0.45,
            "min_through_hole_diameter": 0.2, "min_via_annular_width": 0.1,
            "min_copper_edge_clearance": 0.3, "min_hole_clearance": 0.25, "min_hole_to_hole": 0.25}}},
        "net_settings": {"classes": classes, "netclass_patterns": patterns, "meta": {"version": 4}},
    }
    (OUT / "fc_v1.kicad_pro").write_text(json.dumps(pro, indent=2), encoding="utf-8")


def cli(*args):
    r = subprocess.run([str(CLI), *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout + r.stderr).strip()


def check_and_render(pcb):
    rpt = OUT / "drc.json"
    cli("pcb", "drc", "--format", "json", "--severity-all", "-o", str(rpt), str(pcb))
    drc = json.loads(rpt.read_text(encoding="utf-8"))
    kinds = {}
    for v in drc.get("violations", []):
        kinds.setdefault(v["type"], []).append(v)
    print("DRC(未布线,所以不看未连接项):")
    for k, vs in sorted(kinds.items()):
        print(f"  {k}: {len(vs)}")
    for v in kinds.get("drill_out_of_range", [])[:3]:
        print("    [drill_out_of_range]", v["description"], "|", " | ".join(i["description"] for i in v["items"]))
    for k in ("courtyards_overlap", "pth_inside_courtyard", "npth_inside_courtyard", "copper_edge_clearance",
              "shorting_items", "clearance", "solder_mask_bridge", "hole_clearance", "malformed_courtyard",
              "missing_courtyard"):
        for v in kinds.get(k, []):
            items = " | ".join(i["description"] for i in v["items"])
            print(f"    [{k}] {items}")
    print(f"  未连接的飞线数: {len(drc.get('unconnected_items', []))}")
    for side in ("top", "bottom"):
        cli("pcb", "render", "--side", side, "-w", "1400", "--height", "1400", "--quality", "basic",
            "-o", str(OUT / f"render_{side}.png"), str(pcb))
    cli("pcb", "export", "pdf", "--layers", "F.Fab,F.CrtYd,F.SilkS,Edge.Cuts", "--mode-single",
        "-o", str(OUT / "placement_top.pdf"), str(pcb))
    cli("pcb", "export", "pdf", "--layers", "B.Fab,B.CrtYd,B.SilkS,Edge.Cuts", "--mirror", "--mode-single",
        "-o", str(OUT / "placement_bottom.pdf"), str(pcb))


if __name__ == "__main__":
    write_project()
    pcb, n_parts, n_nets, loose = build()
    print(f"{n_parts} 个封装、{n_nets} 个网络 → {pcb}")
    if loose:
        print("没有分配网络的焊盘:", loose)
    if "--no-check" not in sys.argv:
        check_and_render(pcb)
