"""电机座:3D 打印件,卡在 PCB 臂端的插舌上,套住 7 mm 空心杯电机,向下伸出当脚架。

运行:  python -m mech.motor_mount
输出:  mech/out/motor_mount.png(三视图)。这里的网格只用于网页装配图和预览;
       打印用的 STL 和 STEP 由 python -m mech.cad 生成(同一套尺寸参数,完整实体)。

结构(局部坐标:原点在电机轴线与 PCB 中面的交点,x 指向板中心,z 向上):
  套筒   套住电机,内径比电机大 0.1 mm;底部有一圈底板托住电机,中间留孔走线
  插座   套筒朝板中心一侧伸出的扁盒子,中间一条水平的槽,PCB 插舌从 x 方向插进去
  销孔   插座上下两片各有两个方孔,对着插舌上的两个定位孔,穿一根 1 mm 钢丝(回形针)锁住

打印:轴线竖直放,套筒开口朝上;插槽是 7 mm 宽的短桥,不需要支撑。

这个模型由一组互相贴合/重叠的简单实体(长方体和圆环柱)拼成,切片软件会自动合并。
所有尺寸都是按图纸算的,没有打印验证过:配合间隙(电机孔、插槽、销孔)大概率要试打一两次再调。
"""
import math
from pathlib import Path

import numpy as np

from mech.model import ARM, ARM_W, MOTOR_D, PAD_R, PCB_T, PIN_D, PIN_HOLE, PIN_LAT, TAB_END

OUT = Path(__file__).parent / "out"

BORE_R = MOTOR_D / 2 + 0.05     # 电机孔半径(过盈配合靠打印件的弹性,太紧就把这里加大)
WALL_R = PAD_R                  # 套筒外半径
Z_TOP, Z_SEAT, Z_FOOT = 1.8, -10.0, -16.0      # 套筒顶、电机底面的位置、脚底
# 脚底到板下 16 mm:电池(7 mm)下面还要挂外接的光流/测距模块(预留 5 mm 厚),再留 2 mm 离地
FLOOR_T = 1.5                   # 托住电机的底板厚度
LEG_WALL = 1.0                  # 底板以下的脚是薄壁圆筒,壁厚
WIRE_HOLE_R = 2.0               # 底板中间的走线孔
SLOT_T = PCB_T + 0.15           # 插槽高度(板厚 + 间隙)
SLOT_W = ARM_W + 0.3            # 插槽宽度(臂宽 + 间隙)
PLATE_T = 1.2                   # 插座上下片的厚度
SIDE_T = 1.1                    # 插座两侧壁厚
X_BACK = BORE_R + 0.1           # 插座靠套筒一侧的起点(不能伸进电机孔)
X_SLOT = ARM - TAB_END          # 插舌末端顶到的位置
X_FRONT = X_SLOT + 5.5          # 插座朝板中心一侧的开口
PIN_X = ARM - PIN_D             # 销孔位置
PIN_SQ = PIN_HOLE + 0.2         # 方形销孔的边长
DENSITY_PLA = 1.24e-3           # g/mm³


def box(x0, x1, y0, y1, z0, z1):
    v = np.array([[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], float)
    f = [(0, 2, 1), (1, 2, 3), (4, 5, 6), (5, 7, 6), (0, 1, 4), (1, 5, 4),
         (2, 6, 3), (3, 6, 7), (0, 4, 2), (2, 4, 6), (1, 3, 5), (3, 7, 5)]
    return [v[list(t)] for t in f]


def ring(r_in, r_out, z0, z1, n=64):
    """圆环柱(r_in = 0 时是实心圆柱)。"""
    tris = []
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        c0, s0, c1, s1 = math.cos(a0), math.sin(a0), math.cos(a1), math.sin(a1)

        def p(r, c, s, z):
            return np.array([r * c, r * s, z])
        o00, o01, o10, o11 = p(r_out, c0, s0, z0), p(r_out, c0, s0, z1), p(r_out, c1, s1, z0), p(r_out, c1, s1, z1)
        tris += [np.array([o00, o10, o11]), np.array([o00, o11, o01])]                 # 外壁
        if r_in > 0:
            i00, i01, i10, i11 = p(r_in, c0, s0, z0), p(r_in, c0, s0, z1), p(r_in, c1, s1, z0), p(r_in, c1, s1, z1)
            tris += [np.array([i00, i11, i10]), np.array([i00, i01, i11])]             # 内壁
            tris += [np.array([o01, o11, i11]), np.array([o01, i11, i01])]             # 顶面
            tris += [np.array([o00, i10, o10]), np.array([o00, i00, i10])]             # 底面
        else:
            ct, cb = np.array([0, 0, z1]), np.array([0, 0, z0])
            tris += [np.array([ct, o01, o11]), np.array([cb, o10, o00])]
    return tris


def plate_with_pin_holes(z0, z1):
    """插座的上/下片:长方形,在两个销孔处留出方孔(把长方形切成小格,跳过孔所在的格子)。"""
    half = SLOT_W / 2 + SIDE_T
    xs = [X_BACK, PIN_X - PIN_SQ / 2, PIN_X + PIN_SQ / 2, X_FRONT]
    ys = [-half, -PIN_LAT - PIN_SQ / 2, -PIN_LAT + PIN_SQ / 2, PIN_LAT - PIN_SQ / 2, PIN_LAT + PIN_SQ / 2, half]
    tris = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            if i == 1 and j in (1, 3):
                continue                                   # 销孔
            tris += box(xs[i], xs[i + 1], ys[j], ys[j + 1], z0, z1)
    return tris


def build():
    half = SLOT_W / 2 + SIDE_T
    zs = SLOT_T / 2
    parts = {
        "套筒": ring(BORE_R, WALL_R, Z_SEAT, Z_TOP),
        "底板": ring(WIRE_HOLE_R, WALL_R, Z_SEAT - FLOOR_T, Z_SEAT),
        "脚": ring(WALL_R - LEG_WALL, WALL_R, Z_FOOT, Z_SEAT - FLOOR_T),
        "插座下片": plate_with_pin_holes(-zs - PLATE_T, -zs),
        "插座上片": plate_with_pin_holes(zs, zs + PLATE_T),
        "插座侧壁 1": box(X_BACK, X_FRONT, SLOT_W / 2, half, -zs, zs),
        "插座侧壁 2": box(X_BACK, X_FRONT, -half, -SLOT_W / 2, -zs, zs),
        "插座后壁": box(X_BACK, X_SLOT, -SLOT_W / 2, SLOT_W / 2, -zs, zs),
    }
    return parts


def volume(tris):
    """封闭三角网格的体积(有向四面体求和)。"""
    return sum(float(np.dot(t[0], np.cross(t[1], t[2]))) for t in tris) / 6.0


def write_stl(path, tris):
    lines = ["solid motor_mount"]
    for t in tris:
        n = np.cross(t[1] - t[0], t[2] - t[0])
        n = n / (np.linalg.norm(n) or 1.0)
        lines.append(f"facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n outer loop")
        lines += [f"  vertex {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}" for v in t]
        lines.append(" endloop\nendfacet")
    lines.append("endsolid motor_mount")
    Path(path).write_text("\n".join(lines), encoding="ascii")


def preview(path, tris):
    """三视图线框(俯视、侧视、从板中心方向看),并画出 PCB 插舌和电机的轮廓。"""
    from PIL import Image, ImageDraw
    s, w, h = 22, 1500, 560
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    views = [("top", 0, 1, 250, 260), ("side", 0, 2, 750, 250), ("front (from board centre)", 1, 2, 1250, 250)]
    for name, a, b, ox, oy in views:
        for t in tris:
            pts = [(ox + t[k][a] * s, oy - t[k][b] * s) for k in range(3)]
            d.polygon(pts, outline=(60, 70, 80))
        d.text((ox - 110, 20), name, fill="black")
    # PCB 插舌(绿色)和电机(橙色)在侧视图里的位置
    ox, oy = 750, 250
    d.rectangle([ox + X_SLOT * s, oy - PCB_T / 2 * s, ox + (X_FRONT + 4) * s, oy + PCB_T / 2 * s], outline=(20, 140, 60), width=3)
    d.rectangle([ox - MOTOR_D / 2 * s, oy - (Z_SEAT + 16) * s, ox + MOTOR_D / 2 * s, oy - Z_SEAT * s], outline=(220, 120, 0), width=3)
    img.save(path)


def main():
    parts = build()
    tris = [t for p in parts.values() for t in p]
    for name, p in parts.items():
        assert volume(p) > 0, f"{name} 的面朝向反了"
    # 各实体之间有重叠(套筒与插座),直接相加会多算;重叠部分单独估一下再扣掉
    vol = sum(volume(p) for p in parts.values())
    overlap = 0.0
    half = SLOT_W / 2 + SIDE_T
    n = 60
    for x in np.linspace(X_BACK, WALL_R, n):
        y_max = math.sqrt(max(WALL_R**2 - x * x, 0.0))
        overlap += 2 * min(y_max, half) * (WALL_R - X_BACK) / n * (SLOT_T + 2 * PLATE_T)
    vol -= overlap
    OUT.mkdir(exist_ok=True)
    preview(OUT / "motor_mount.png", tris)      # 打印用的 STL/STEP 由 mech/cad.py 生成(完整实体)
    print(f"三角面 {len(tris)} 个;体积约 {vol:.0f} mm³;PLA 实心打印约 {vol * DENSITY_PLA:.2f} g")
    print(f"外形:套筒外径 {2 * WALL_R:.1f} mm、高 {Z_TOP - Z_FOOT:.1f} mm;插座伸出到离轴线 {X_FRONT:.1f} mm")
    print(f"脚底在 PCB 中面下方 {-Z_FOOT:.1f} mm;电机底面在下方 {-Z_SEAT:.1f} mm")
    print(f"已写出 {OUT / 'motor_mount.png'}(打印文件请运行 python -m mech.cad)")


if __name__ == "__main__":
    main()
