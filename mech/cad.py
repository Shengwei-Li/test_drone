"""实体 CAD:电机座的实体模型、整机装配的 STEP 文件、实体干涉检查。

运行:  python -m mech.cad
输出:  mech/out/motor_mount.step、motor_mount.stl   电机座(一个完整的实体,打印用这个 STL)
       out/assembly.step                             整机装配(可用 Fusion 360 / FreeCAD 打开)
并在终端打印各零件与 PCB(含板上所有元件)的实体相交体积。

用的是 build123d(底层是 OpenCascade 几何内核)。电机座的尺寸参数仍然来自 motor_mount.py。
机体系:x 前、y 左、z 上,原点在板中心、PCB 厚度的中面,单位 mm。
"""
import math
import subprocess
from pathlib import Path

from mech import motor_mount as mm
from mech._b123d import bd
from mech.model import (BATT, MODULE, MOTOR_D, MOTOR_L, PCB_T, PIN_LAT, PROP_D, balanced, module_z,
                        motor_positions)

ROOT = Path(__file__).resolve().parents[1]
OUT, MOUT = ROOT / "out", Path(__file__).parent / "out"
PCB = ROOT / "hardware" / "fc_v1" / "fc_v1.kicad_pcb"
CLI = Path(r"D:\kicad\bin\kicad-cli.exe")
MIN = (bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)
PROP_Z = 8.5


def cyl(r, z0, z1):
    return bd.Pos(0, 0, z0) * bd.Cylinder(r, z1 - z0, align=MIN)


def box(x0, x1, y0, y1, z0, z1):
    return bd.Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * bd.Box(x1 - x0, y1 - y0, z1 - z0)


def motor_mount():
    """电机座(局部坐标:原点在电机轴线与 PCB 中面的交点,x 指向板中心,z 向上)。"""
    half = mm.SLOT_W / 2 + mm.SIDE_T
    zs = mm.SLOT_T / 2
    body = cyl(mm.WALL_R, mm.Z_FOOT, mm.Z_TOP) + box(mm.X_BACK, mm.X_FRONT, -half, half, -zs - mm.PLATE_T, zs + mm.PLATE_T)
    try:                                               # 插座外侧的四条竖棱倒圆,减少应力集中
        edges = [e for e in body.edges().filter_by(bd.Axis.Z) if e.center().X > mm.X_FRONT - 0.1]
        body = bd.fillet(edges, 0.8)
    except Exception:                                  # noqa: BLE001  倒圆失败不影响功能
        pass
    cut = (cyl(mm.BORE_R, mm.Z_SEAT, mm.Z_TOP + 1)                             # 电机孔
           + cyl(mm.WIRE_HOLE_R, mm.Z_SEAT - mm.FLOOR_T - 0.1, mm.Z_SEAT + 0.1)   # 底板中间的走线孔
           + cyl(mm.WALL_R - mm.LEG_WALL, mm.Z_FOOT - 1, mm.Z_SEAT - mm.FLOOR_T)   # 脚是空心的薄壁圆筒
           + box(0, mm.WALL_R + 1, -1.25, 1.25, mm.Z_FOOT - 1, mm.Z_SEAT - mm.FLOOR_T - 1.0)   # 脚朝板中心一侧开槽,电机线从这里出来
           + box(mm.X_SLOT, mm.X_FRONT + 1, -mm.SLOT_W / 2, mm.SLOT_W / 2, -zs, zs))   # 插槽
    for s in (1, -1):                                                           # 销孔(圆孔,贯穿上下片)
        cut += bd.Pos(mm.PIN_X, s * PIN_LAT, 0) * cyl(mm.PIN_SQ / 2, -zs - mm.PLATE_T - 1, zs + mm.PLATE_T + 1)
    return body - cut


def export_pcb_step():
    OUT.mkdir(exist_ok=True)
    path = OUT / "fc_v1.step"
    r = subprocess.run([str(CLI), "pcb", "export", "step", "--subst-models", "-f", "--user-origin", "100x100mm",
                        "-o", str(path), str(PCB)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert path.exists(), r.stdout + r.stderr
    return path


def load_pcb():
    """KiCad 的 STEP:X 向右、Y 向图纸上方、板底在 z = 0。转成机体系:x 前(= STEP 的 Y)、y 左、z 以板中面为 0。"""
    pcb = bd.import_step(str(export_pcb_step()))
    return bd.Pos(0, 0, -PCB_T / 2) * bd.Rot(0, 0, -90) * pcb


def assembly_parts():
    batt_x, _ = balanced()
    mount = motor_mount()
    motor = cyl(MOTOR_D / 2, mm.Z_SEAT, mm.Z_SEAT + MOTOR_L)
    parts = {}
    for k, (x, y) in enumerate(motor_positions(), 1):
        place = bd.Pos(x, y, 0) * bd.Rot(0, 0, math.degrees(math.atan2(-y, -x)))
        parts[f"电机座 {k}"] = place * mount
        parts[f"电机 {k}"] = place * motor
        parts[f"桨 {k}"] = bd.Pos(x, y, PROP_Z - 0.5) * bd.Cylinder(PROP_D / 2, 1.0, align=MIN)
    parts["电池"] = bd.Pos(batt_x, 0, -(PCB_T / 2 + 1.0 + BATT[2] / 2)) * bd.Box(*BATT)
    parts["光流测距模块(预留空间)"] = bd.Pos(batt_x, 0, module_z()[0]) * bd.Box(*MODULE)
    return mount, parts


def bounds(shape):
    bb = shape.bounding_box()
    return (bb.min.X, bb.min.Y, bb.min.Z), (bb.max.X, bb.max.Y, bb.max.Z)


def overlaps(a, b, margin=0.0):
    (a0, a1), (b0, b1) = bounds(a), bounds(b)
    return all(a0[i] <= b1[i] + margin and b0[i] <= a1[i] + margin for i in range(3))


def interference(parts, pcb):
    """每个零件与 PCB 模型里各个实体(板子本身和每个元件)做布尔交,返回 {零件: 相交体积 mm³}。"""
    solids = list(pcb.solids())
    out = {}
    for name, part in parts.items():
        vol = 0.0
        for s in solids:
            if not overlaps(part, s):
                continue
            try:
                common = part & s
                vol += sum(c.volume for c in common.solids()) if common is not None else 0.0
            except Exception:                              # noqa: BLE001
                vol += float("nan")
        out[name] = vol
    return out, len(solids)


def main():
    MOUT.mkdir(exist_ok=True)
    mount, parts = assembly_parts()
    bd.export_step(mount, str(MOUT / "motor_mount.step"))
    bd.export_stl(mount, str(MOUT / "motor_mount.stl"), tolerance=0.01, angular_tolerance=0.1)
    bb = mount.bounding_box()
    print(f"电机座:体积 {mount.volume:.0f} mm³,PLA 实心约 {mount.volume * mm.DENSITY_PLA:.2f} g,"
          f"外形 {bb.size.X:.1f} × {bb.size.Y:.1f} × {bb.size.Z:.1f} mm,"
          f"{'是' if mount.is_valid else '不是'}合法实体,{len(mount.solids())} 个实体")
    pcb = load_pcb()
    pb = pcb.bounding_box()
    print(f"PCB 模型:x {pb.min.X:.1f}..{pb.max.X:.1f},y {pb.min.Y:.1f}..{pb.max.Y:.1f},z {pb.min.Z:.1f}..{pb.max.Z:.1f} mm")
    tall = max(pcb.solids(), key=lambda s: bounds(s)[1][2])            # 顶面最高的元件应该是无线模块,在机头一侧
    (tx0, ty0, _), (tx1, ty1, tz1) = bounds(tall)
    print(f"  最高的元件:x {tx0:.1f}..{tx1:.1f},y {ty0:.1f}..{ty1:.1f},顶面 z = {tz1:.1f} mm(应为无线模块,x 为正)")
    assert tx0 > 0 and abs(ty0 + ty1) < 1, "PCB 模型的朝向不对"
    res, n = interference(parts, pcb)
    print(f"实体干涉检查(各零件与 PCB 模型的 {n} 个实体逐一求交):")
    bad = 0
    for name, vol in res.items():
        flag = "无干涉" if vol < 0.01 else f"相交 {vol:.2f} mm³"      # 0.01 mm³ 以下是数值误差
        bad += vol >= 0.01 or vol != vol
        print(f"  {name}:{flag}")
    # 零件之间
    names = list(parts)
    pair_bad = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = parts[names[i]], parts[names[j]]
            if overlaps(a, b):
                c = a & b
                v = sum(s.volume for s in c.solids()) if c is not None else 0.0
                if v > 1e-3:
                    pair_bad.append((names[i], names[j], v))
    for a, b, v in pair_bad:
        print(f"  {a} 与 {b}:相交 {v:.2f} mm³")
    if not pair_bad:
        print("  零件之间:无干涉(电机与电机座之间留有 0.05 mm 的配合间隙)")
    asm = bd.Compound(children=[pcb] + list(parts.values()), label="quad_v1")
    bd.export_step(asm, str(OUT / "assembly.step"))
    print(f"已写出 {MOUT / 'motor_mount.step'}、{MOUT / 'motor_mount.stl'}、{OUT / 'assembly.step'}")
    return bad + len(pair_bad)


if __name__ == "__main__":
    main()
