"""机架的参数化模型:零件清单 → 质量、重心、转动惯量 → 仿真参数。

运行:  python -m mech.model
输出:  质量预算表、重心、惯量、推重比;
       mech/out/design.json(给仿真和 3D 回放用)、mech/out/top_view.svg(俯视图)。

方案:PCB 本身就是机架(Crazyflie 的做法)。中间是方形主板,四条臂从四角伸出,
臂端装空心杯电机;电池在板子下方,电机座向下伸出当脚架。

坐标:机体系 FLU(x 前、y 左、z 上),原点在四个电机的几何中心、PCB 厚度的中面。单位 mm、g。

零件质量大多是凭经验估的(见各行注释),买到实物后要逐个称重回填。
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sim.params import QuadParams

OUT = Path(__file__).parent / "out"

# ---------------- 设计参数(改这里) ----------------
ARM = 48.0            # 中心到电机轴的距离
BODY = 40.0           # 主板中间方形部分的边长
ARM_W = 7.0           # 臂宽
PAD_R = 4.45          # 电机座套筒的外半径
TAB_END = ARM - 4.0   # 臂端插舌的末端到板中心的距离(插进电机座的插槽里)
PIN_D = ARM - 6.8     # 定位孔到板中心的距离
PIN_LAT = 2.0         # 定位孔偏离臂中线的距离(左右各一个)
PIN_HOLE = 1.1        # 定位孔直径,穿 1.0 mm 的钢丝或回形针做销子
PCB_T = 1.0           # 板厚
PCB_DENSITY = 2.1e-3  # g/mm³,FR4 约 1.85,加上 4 层铜和阻焊的折算值
PROP_D = 46.0         # 桨直径
MOTOR_D, MOTOR_L = 7.0, 16.0
MOUNT_MASS = 0.72     # g,电机座(由 mech/motor_mount.py 的体积按 PLA 算出,见该脚本的输出)
MOTOR_FMAX = 0.149    # N,单个电机的最大推力(Crazyflie 7×16 电机 + 45 mm 桨的量级,未实测)
BATT = (20.0, 30.0, 7.0)   # 电池 x、y、z 尺寸(1S 300 mAh 左右)
# 外接的光流/测距模块:型号未定,先按"最大可能"预留空间。装在电池正下方,朝下。
# 30 × 36 × 5 mm 能放下 Bitcraze Flow deck v2 (21 × 28 × 4) 和 Matek 3901-L0X (36 × 12);
# MicoAir MTF-01 高 14.5 mm,放不进这个空间。
MODULE = (30.0, 36.0, 5.0)        # x、y、z 方向的预留尺寸
MODULE_MASS = 2.0                 # g,按较重的 Matek 模块估
MODULE_GAP = 0.5                  # 电池底面到模块之间的垫层/托架厚度


def module_z():
    """模块预留空间的中心高度和底面高度。"""
    top = -(PCB_T / 2 + 1.0 + BATT[2] + MODULE_GAP)
    return top - MODULE[2] / 2, top - MODULE[2]


@dataclass
class Part:
    name: str
    mass: float                       # g
    shape: str                        # "box" | "cyl"(轴沿 z)| "plate"(多边形薄板)
    size: tuple                       # box: (lx, ly, lz);cyl: (半径, 高);plate: (厚度,)
    pos: tuple = (0.0, 0.0, 0.0)      # 质心位置
    color: str = "#2b3238"
    outline: list = field(default_factory=list)   # plate 的轮廓点
    prop: bool = False                # 是否是桨(回放里半透明显示)

    def inertia(self) -> np.ndarray:
        """绕自身质心的惯量张量 (g·mm²)。"""
        m = self.mass
        if self.shape == "box":
            lx, ly, lz = self.size
            return m / 12.0 * np.diag([ly**2 + lz**2, lx**2 + lz**2, lx**2 + ly**2])
        if self.shape == "cyl":
            r, h = self.size
            return m * np.diag([(3 * r * r + h * h) / 12.0, (3 * r * r + h * h) / 12.0, r * r / 2.0])
        if self.shape == "plate":
            a, _, (ixx, iyy, ixy) = polygon_moments(self.outline)
            k = m / a                                 # 面密度
            t = self.size[0]
            return np.array([[k * ixx + m * t * t / 12, -k * ixy, 0.0],
                             [-k * ixy, k * iyy + m * t * t / 12, 0.0],
                             [0.0, 0.0, k * (ixx + iyy)]])
        raise ValueError(self.shape)


def polygon_moments(pts):
    """多边形的面积、形心、绕形心的面积二次矩 (∫y²dA, ∫x²dA, ∫xy dA)。"""
    p = np.asarray(pts, float)
    x0, y0 = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    c = x0 * y1 - x1 * y0
    a = c.sum() / 2.0
    cx = ((x0 + x1) * c).sum() / (6 * a)
    cy = ((y0 + y1) * c).sum() / (6 * a)
    ixx = ((y0**2 + y0 * y1 + y1**2) * c).sum() / 12.0 - a * cy**2
    iyy = ((x0**2 + x0 * x1 + x1**2) * c).sum() / 12.0 - a * cx**2
    ixy = ((x0 * y1 + 2 * x0 * y0 + 2 * x1 * y1 + x1 * y0) * c).sum() / 24.0 - a * cx * cy
    return a, (cx, cy), (ixx, iyy, ixy)


def board_outline(n_arc: int = 16):
    """PCB 外形(逆时针):方形主板 + 四条对角的臂。臂端是一段直的插舌,插进电机座。"""
    h, w = BODY / 2, ARM_W / 2
    cut = w * np.sqrt(2)                              # 臂与方形边相交处到角的距离
    u, n = np.array([1, 1]) / np.sqrt(2), np.array([-1, 1]) / np.sqrt(2)
    c = 0.8                                           # 插舌末端倒角,方便插入
    base = [(h, h - cut),                             # 第一象限:沿右边向上走到臂根
            tuple((TAB_END - c) * u - w * n), tuple(TAB_END * u - (w - c) * n),
            tuple(TAB_END * u + (w - c) * n), tuple((TAB_END - c) * u + w * n),
            (h - cut, h)]
    pts = []
    for k in range(4):                                # 旋转 90° 得到其余三个象限
        co, s = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
        pts += [(co * x - s * y, s * x + co * y) for x, y in base]
    return [(round(float(x), 3), round(float(y), 3)) for x, y in pts]


def pin_holes():
    """四条臂上的定位孔中心(机体系 x、y),每条臂两个。"""
    out = []
    for k in range(4):
        a = np.pi / 4 + k * np.pi / 2
        u, n = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        out += [tuple(float(v) for v in PIN_D * u + s * PIN_LAT * n) for s in (1, -1)]
    return out


def motor_positions():
    d = ARM / np.sqrt(2)
    return [(d, -d), (-d, -d), (-d, d), (d, d)]       # M1 右前、M2 右后、M3 左后、M4 左前


def build(batt_x: float = 0.0):
    """零件清单。batt_x 是电池中心的前后位置,用来配平重心。"""
    top, bot = PCB_T / 2, -PCB_T / 2
    outline = board_outline()
    area = polygon_moments(outline)[0]
    parts = [
        Part("PCB(4 层,1.0 mm)", area * PCB_T * PCB_DENSITY, "plate", (PCB_T,), color="#1d6b3a", outline=outline),
        # 以下位置取自 PCB 的实际摆位(hardware/pcb_gen.py),质量仍是估计值
        Part("STM32F405(LQFP-64)", 0.35, "box", (10, 10, 1.4), (-10.5, 0, top + 0.7)),
        Part("ESP32-C3 模块", 1.4, "box", (20, 18, 3.2), (15.5, 0, top + 1.6), color="#8a939b"),    # 天线端伸出板前沿
        Part("USB-C 座", 0.3, "box", (9, 7.5, 3.2), (-10.5, 16.3, top + 1.6), color="#b0b6bb"),      # 左侧,开口朝左
        Part("Flash(SOIC-8)", 0.15, "box", (8, 5.3, 1.8), (10.5, 15.0, top + 0.9)),
        Part("电源开关", 0.25, "box", (3, 7, 1.5), (-16.6, -15.5, top + 0.75), color="#b0b6bb"),
        Part("电感", 0.1, "box", (2, 2.5, 1.2), (-7.0, -17.3, top + 0.6)),
        Part("底面接口 ×3", 0.6, "box", (6, 22, 3), (-16.0, 0, bot - 1.5), color="#e8e2d0"),          # 扩展口和调试口在后沿
        Part("其余元件和焊锡", 1.5, "box", (BODY - 4, BODY - 4, 0.6), (0, 0, top + 0.3)),            # 估计值
        Part("光流测距模块(外接)", MODULE_MASS, "box", MODULE, (batt_x, 0, module_z()[0]), color="#3a4a8a"),   # 预留空间
        Part("电池 1S 300 mAh", 7.5, "box", BATT, (batt_x, 0, bot - 1.0 - BATT[2] / 2), color="#c9ced3"),
        Part("导线和绑带", 1.0, "box", (10, 10, 2), (batt_x, 0, bot - 1.0)),                         # 估计值
    ]
    for i, (x, y) in enumerate(motor_positions(), 1):
        parts += [
            Part(f"电机 {i}(7×16)", 2.7, "cyl", (MOTOR_D / 2, MOTOR_L), (x, y, -2.0), color="#b0b6bb"),
            Part(f"电机座 {i}", MOUNT_MASS, "cyl", (PAD_R, 17.8), (x, y, -7.1), color="#20262b"),  # 打印件,兼作脚架
            Part(f"桨 {i}", 0.25, "cyl", (PROP_D / 2, 1.0), (x, y, 8.5),
                 color="#d6332a" if x > 0 else "#59636b", prop=True),
        ]
    return parts


def mass_properties(parts):
    m = sum(p.mass for p in parts)
    cg = sum(p.mass * np.array(p.pos) for p in parts) / m
    inertia = np.zeros((3, 3))
    for p in parts:
        d = np.array(p.pos) - cg
        inertia += p.inertia() + p.mass * (d @ d * np.eye(3) - np.outer(d, d))   # 平行轴定理
    return m, cg, inertia


def balanced():
    """解出让重心前后位置为 0 的电池位置(质量对 batt_x 是线性的,一步到位)。"""
    _, cg0, _ = mass_properties(build(0.0))
    _, cg1, _ = mass_properties(build(1.0))
    x = -cg0[0] / (cg1[0] - cg0[0])
    return x, build(x)


def design_params(base: QuadParams | None = None) -> QuadParams:
    """由机架模型得到的仿真参数(其余参数沿用 base,目前是 Crazyflie 的占位值)。"""
    from dataclasses import replace
    base = base or QuadParams()
    _, parts = balanced()
    m, _, inertia = mass_properties(parts)
    mass = m * 1e-3
    return replace(base, mass=mass, arm=ARM * 1e-3,
                   inertia=tuple(float(v) * 1e-9 for v in np.diag(inertia)),
                   thrust_to_weight=4 * MOTOR_FMAX / (mass * base.g))


def viz_model(parts, cg):
    """给 3D 回放用的零件几何(单位 m,原点移到重心)。"""
    out = []
    for p in parts:
        d = dict(shape=p.shape, size=[v * 1e-3 for v in p.size], color=p.color, prop=p.prop,
                 pos=[(a - b) * 1e-3 for a, b in zip(p.pos, cg)])
        if p.shape == "plate":
            d["outline"] = [[(x - cg[0]) * 1e-3, (y - cg[1]) * 1e-3] for x, y in p.outline]
        out.append(d)
    return out


def top_view_svg(parts, cg, batt_x) -> str:
    s = 4.0                                            # 像素/mm

    def X(v):
        return 240 - v[1] * s                          # 机体 y(左)画在图的左边

    def Y(v):
        return 240 - v[0] * s                          # 机体 x(前)画在图的上方

    el = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 480" font-family="sans-serif" font-size="11">',
          '<rect width="480" height="480" fill="#f7f8f9"/>']
    for p in parts:
        if p.shape == "plate":
            pts = " ".join(f"{X(q):.1f},{Y(q):.1f}" for q in p.outline)
            el.append(f'<polygon points="{pts}" fill="{p.color}" fill-opacity="0.85" stroke="#0d3a1f"/>')
    for p in parts:
        if p.shape == "box":
            lx, ly, _ = p.size
            below = p.pos[2] < 0
            el.append(f'<rect x="{X(p.pos) - ly * s / 2:.1f}" y="{Y(p.pos) - lx * s / 2:.1f}" width="{ly * s:.1f}" '
                      f'height="{lx * s:.1f}" fill="{"none" if below else p.color}" fill-opacity="0.9" '
                      f'stroke="#20262b" stroke-dasharray="{"5 3" if below else "none"}"/>')
        elif p.shape == "cyl" and p.prop:
            el.append(f'<circle cx="{X(p.pos):.1f}" cy="{Y(p.pos):.1f}" r="{p.size[0] * s:.1f}" fill="{p.color}" '
                      f'fill-opacity="0.18" stroke="{p.color}"/>')
        elif p.shape == "cyl" and "电机 " in p.name:
            el.append(f'<circle cx="{X(p.pos):.1f}" cy="{Y(p.pos):.1f}" r="{p.size[0] * s:.1f}" fill="{p.color}" stroke="#20262b"/>')
            el.append(f'<text x="{X(p.pos) - 8:.1f}" y="{Y(p.pos) - 18:.1f}">M{p.name[3]}</text>')
    el.append(f'<circle cx="{X(cg):.1f}" cy="{Y(cg):.1f}" r="4" fill="#e07b00"/>')
    el.append(f'<text x="{X(cg) + 7:.1f}" y="{Y(cg) + 4:.1f}" fill="#e07b00">重心</text>')
    el.append('<text x="12" y="20">俯视图(上方是机头 x,左方是 y);虚线框在板子下面</text>')
    el.append(f'<text x="12" y="468">电机对角距 {2 * ARM:.0f} mm,桨径 {PROP_D:.0f} mm,电池中心 x = {batt_x:+.1f} mm</text>')
    el.append('<line x1="240" y1="60" x2="240" y2="30" stroke="#d6332a" stroke-width="2"/>'
              '<polygon points="235,34 245,34 240,24" fill="#d6332a"/></svg>')
    return "\n".join(el)


def main():
    batt_x, parts = balanced()
    m, cg, inertia = mass_properties(parts)
    p = design_params()
    old = QuadParams()
    print(f"{'零件':<22}{'质量 g':>8}")
    groups = {}
    for part in parts:
        key = part.name
        if key[:2] in ("电机", "桨 "):
            key = key.replace("(7×16)", "").rstrip("1234 ") + " ×4"
        groups[key] = groups.get(key, 0.0) + part.mass
    for k, v in groups.items():
        print(f"{k:<22}{v:>8.2f}")
    print(f"{'合计':<22}{m:>8.2f}")
    print(f"\n电池中心 x = {batt_x:+.2f} mm(配平后重心 x = {cg[0]:+.3f}, y = {cg[1]:+.3f}, z = {cg[2]:+.2f} mm)")
    print(f"板底:电池下方预留 {MODULE[0]:.0f} × {MODULE[1]:.0f} × {MODULE[2]:.0f} mm 给外接光流/测距模块,"
          f"底面在板下 {-module_z()[1] :.1f} mm")
    print(f"板底:电池后沿 x = {batt_x - BATT[0] / 2:+.1f} mm(主板后沿在 {-BODY / 2:+.1f} mm)")
    side = ARM * np.sqrt(2)
    print(f"相邻电机间距 {side:.1f} mm,桨尖间隙 {side - PROP_D:.1f} mm")
    print(f"\n{'参数':<16}{'本设计':>14}{'原占位值':>14}")
    print(f"{'质量 kg':<16}{p.mass:>14.4f}{old.mass:>14.4f}")
    print(f"{'臂长 m':<16}{p.arm:>14.4f}{old.arm:>14.4f}")
    for i, ax in enumerate("xyz"):
        print(f"{'惯量 J' + ax + ' kg·m²':<16}{p.inertia[i]:>14.3e}{old.inertia[i]:>14.3e}")
    print(f"{'推重比':<16}{p.thrust_to_weight:>14.2f}{old.thrust_to_weight:>14.2f}")
    off = np.abs(inertia - np.diag(np.diag(inertia))).max() / np.diag(inertia).min()
    print(f"惯量张量的非对角项最大为对角项的 {off * 100:.1f}%(仿真里按 0 处理)")

    OUT.mkdir(exist_ok=True)
    (OUT / "design.json").write_text(json.dumps({
        "mass_g": m, "cg_mm": cg.tolist(), "inertia_g_mm2": inertia.tolist(), "battery_x_mm": batt_x,
        "params": {"mass": p.mass, "arm": p.arm, "inertia": p.inertia, "thrust_to_weight": p.thrust_to_weight},
        "board_outline_mm": board_outline(), "motors_mm": motor_positions(),
        "pin_holes_mm": pin_holes(), "pin_hole_diameter_mm": PIN_HOLE,
        "parts": [{"name": q.name, "mass_g": q.mass, "shape": q.shape, "size_mm": q.size, "pos_mm": q.pos}
                  for q in parts],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "top_view.svg").write_text(top_view_svg(parts, cg, batt_x), encoding="utf-8")
    print(f"\n已写出 {OUT / 'design.json'} 和 {OUT / 'top_view.svg'}")


if __name__ == "__main__":
    main()
