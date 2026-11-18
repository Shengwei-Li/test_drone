"""整机装配:把 KiCad 导出的 PCB 三维模型、电机座、电机、桨、电池、光流子板放到一起。

运行:  python -m mech.assembly
输出:  out/assembly.html(浏览器里可旋转查看;需要联网加载 three.js),并在终端打印干涉检查结果。
前置:  python hardware/fc_v1.py、python -m mech.model、python -m mech.motor_mount、
       D:\\kicad\\bin\\python.exe hardware\\pcb_gen.py 都运行过。

装配不是在 CAD 软件里做的:各零件的位置都由参数算出来(电机轴位置、插舌、电池配平位置等),
PCB 模型由 kicad-cli 导出,电机座是 motor_mount.py 生成的网格,电机、桨、电池是简单的圆柱和方块。
机体系:x 前、y 左、z 上,原点在板中心、PCB 厚度的中面,单位 mm。
"""
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from mech import motor_mount as mm
from mech.model import (ARM, ARM_W, BATT, MODULE, MOTOR_D, MOTOR_L, PCB_T, PROP_D, TAB_END, balanced, module_z,
                        mass_properties, motor_positions)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
PCB = ROOT / "hardware" / "fc_v1" / "fc_v1.kicad_pcb"
CLI = Path(r"D:\kicad\bin\kicad-cli.exe")
PCB_ORIGIN = 100.0          # 板中心在 KiCad 图纸上的坐标 (mm)
PROP_Z = 8.5                # 桨盘中面的高度
ESP_TOP = PCB_T / 2 + 3.2   # 无线模块顶面的高度


def export_pcb():
    OUT.mkdir(exist_ok=True)
    for args in (["pcb", "export", "glb", "--subst-models", "--include-pads", "-f",
                  "--user-origin", f"{PCB_ORIGIN}x{PCB_ORIGIN}mm", "-o", str(OUT / "fc_v1.glb"), str(PCB)],
                 ["pcb", "export", "pos", "--format", "csv", "--units", "mm", "--side", "both",
                  "-o", str(PCB.parent / "positions.csv"), str(PCB)]):
        r = subprocess.run([str(CLI), *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert r.returncode == 0, r.stdout + r.stderr


def placements():
    """元件位置 → 机体系 (x, y)。KiCad 位置文件里 X 向右、Y 向上;俯视图里机头朝上。"""
    out = {}
    with open(PCB.parent / "positions.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u, v_up = float(row["PosX"]) - PCB_ORIGIN, float(row["PosY"]) + PCB_ORIGIN
            out[row["Ref"]] = dict(x=v_up, y=-u, rot=float(row["Rot"]), side=row["Side"], package=row["Package"])
    return out


def mount_transform(k):
    """第 k 个电机座:平移到电机轴,绕 z 转到"局部 x 指向板中心"。返回 (位置, 转角)。"""
    x, y = motor_positions()[k]
    return (x, y), math.atan2(-y, -x)


def checks(place, batt_x):
    """按实际摆位做的几何检查。返回 [(项目, 数值说明, 是否通过)]。"""
    res = []
    socket_start = ARM - mm.X_FRONT                           # 电机座的插座从离中心这么远开始
    arm_refs = [r for r in place if r[0] in "QDRCJ" and math.hypot(place[r]["x"], place[r]["y"]) > 27.5]
    far = max(arm_refs, key=lambda r: math.hypot(place[r]["x"], place[r]["y"]))
    d_far = math.hypot(place[far]["x"], place[far]["y"])
    reach = d_far + (1.0 if far.startswith("J") else 1.8)     # 元件自身的半长(焊孔按焊盘半径 + 余量)
    res.append(("臂上最外侧的元件到电机座插座的间隙", f"{far} 外沿在 {reach:.1f} mm,插座从 {socket_start:.1f} mm 开始,"
                f"间隙 {socket_start - reach:+.1f} mm", socket_start - reach > 0.2))
    res.append(("插舌插入插座的深度", f"{TAB_END - socket_start:.1f} mm(插座深 {mm.X_FRONT - mm.X_SLOT:.1f} mm)",
                abs((TAB_END - socket_start) - (mm.X_FRONT - mm.X_SLOT)) < 0.01))
    res.append(("插槽对插舌的间隙", f"宽度方向 {mm.SLOT_W - ARM_W:.2f} mm,厚度方向 {mm.SLOT_T - PCB_T:.2f} mm",
                mm.SLOT_W > ARM_W and mm.SLOT_T > PCB_T))
    # 桨盘:与相邻桨、与无线模块
    side = ARM * math.sqrt(2)
    res.append(("相邻两桨的桨尖间隙", f"{side - PROP_D:.1f} mm", side - PROP_D > 3))
    esp = place["U8"]
    tip = [(esp["x"] + 13.1, s * 9.0) for s in (1, -1)]       # 模块前端两个角(天线端)
    gap_xy = min(math.hypot(tx - mx, ty - my) for tx, ty in tip for mx, my in motor_positions()) - PROP_D / 2
    gap_z = PROP_Z - 0.5 - ESP_TOP
    res.append(("桨盘与无线模块", f"俯视方向间隙 {gap_xy:+.1f} mm,高度方向桨在模块顶面上方 {gap_z:.1f} mm",
                gap_xy > 0 or gap_z > 2))
    # 板底:电池与底面接口、光流子板、脚的高度
    bx0, bx1, by = batt_x - BATT[0] / 2, batt_x + BATT[0] / 2, BATT[1] / 2
    # half:封装在 0° 时沿机体 x(前后)和 y(左右)方向的半尺寸;转 90° 时对调
    for ref, name, half in (("J7", "调试口", (4.4, 1.6)), ("J8", "扩展口", (3.3, 5.9)), ("J9", "子板接口", (3.3, 5.9))):
        p = place[ref]
        hx, hy = half if p["rot"] % 180 == 0 else (half[1], half[0])
        dx = max(bx0 - (p["x"] + hx), (p["x"] - hx) - bx1)
        dy = abs(p["y"]) - hy - by
        gap = max(dx, dy)
        res.append((f"电池与底面{name} {ref}", f"间隙 {gap:+.1f} mm", gap > 0.3))
    mod_bottom = module_z()[1]
    res.append(("外接模块的预留空间", f"{MODULE[0]:.0f} × {MODULE[1]:.0f} × {MODULE[2]:.0f} mm,在电池正下方,"
                f"底面离地 {mod_bottom - mm.Z_FOOT:.1f} mm", mod_bottom - mm.Z_FOOT > 1.5))
    for ref, name in (("J9", "模块口 A"), ("J10", "模块口 B")):
        p = place[ref]
        res.append((f"{name} {ref} 到模块空间前沿的走线距离", f"{abs(p['x'] - (batt_x + MODULE[0] / 2)):.0f} mm"
                    f"(接口在板底 x = {p['x']:+.1f},y = {p['y']:+.1f})", True))
    batt_bottom = -(PCB_T / 2 + 1.0 + BATT[2])
    res.append(("脚架高度", f"脚底在板下 {-mm.Z_FOOT:.1f} mm,电池底面在板下 {-batt_bottom:.1f} mm",
                batt_bottom - mm.Z_FOOT > 1.5))
    res.append(("电机在电机座里的夹持长度", f"{mm.Z_TOP - mm.Z_SEAT:.1f} mm(电机长 {MOTOR_L:.0f} mm)",
                mm.Z_TOP - mm.Z_SEAT > 0.6 * MOTOR_L))
    return res


def main():
    export_pcb()
    place = placements()
    batt_x, parts = balanced()
    _, cg, _ = mass_properties(parts)
    tris = [t for p in mm.build().values() for t in p]
    data = dict(
        mount=np.round(np.array(tris).reshape(-1, 9), 3).tolist(),
        mounts=[dict(pos=list(mount_transform(k)[0]), rot=mount_transform(k)[1]) for k in range(4)],
        motor=dict(r=MOTOR_D / 2, z0=mm.Z_SEAT, len=MOTOR_L), prop=dict(r=PROP_D / 2, z=PROP_Z),
        battery=dict(size=list(BATT), x=batt_x, z=-(PCB_T / 2 + 1.0 + BATT[2] / 2)),
        flow=dict(size=list(MODULE), x=batt_x, z=module_z()[0]),
        cg=[float(v) for v in cg], foot_z=mm.Z_FOOT,
    )
    html = (Path(__file__).parent / "assembly_template.html").read_text(encoding="utf-8")
    (OUT / "assembly.html").write_text(html.replace("/*DATA*/", json.dumps(data, separators=(",", ":"))),
                                       encoding="utf-8")
    print("干涉与配合检查(按 PCB 的实际摆位):")
    ok_all = True
    for name, text, ok in checks(place, batt_x):
        ok_all &= ok
        print(f"  [{'通过' if ok else '不通过'}] {name}:{text}")
    print(f"装配页面: {OUT / 'assembly.html'}")
    return ok_all


if __name__ == "__main__":
    main()
