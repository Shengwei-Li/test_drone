"""用 Python 生成 KiCad 原理图 (.kicad_sch) 的小工具。

思路:每个器件摆在指定坐标,每个引脚引出一小段导线,末端放网络标签或电源符号。
同名标签即相连,所以不需要画长导线;连接关系全部写在 fc_v1.py 的引脚→网络表里,
改设计就是改那张表,然后重新生成。

符号直接从 KiCad 自带的符号库读取并嵌入原理图,引脚编号和位置以库为准。
"""
import copy
import os
import uuid
from pathlib import Path

from sexpr import Str, dump, find, find_all, load

KICAD_SHARE = Path(os.environ.get("KICAD_SHARE", r"D:\kicad\share\kicad"))
STUB = 2.54
POWER_NETS = {"GND": "down", "+3V3": "up", "+BATT": "up", "VBUS": "up", "+5V": "up"}   # 电源符号的朝向
OUT_DIR = {0: (-1, 0), 180: (1, 0), 90: (0, 1), 270: (0, -1)}             # 引脚角度 → 向外的方向(图纸坐标,y 向下)
LABEL_STYLE = {(-1, 0): (180, "right"), (1, 0): (0, "left"), (0, -1): (90, "left"), (0, 1): (270, "right")}


def r2(v):
    return round(v + 0.0, 4)


class Schematic:
    def __init__(self, project: str, title: str, paper: str = "A3"):
        self.project, self.title, self.paper = project, title, paper
        self._libs, self._lib_syms = {}, {}
        self.items = []
        self.root_uuid = self._uid("root")
        self._n = 0
        self._pwr_n = 0
        self.net_pins = {}          # 网络 → [(位号, 引脚号, 电气类型)]
        self.refs = set()
        self.parts = {}             # 位号 → 数值、封装、原理图符号的 uuid(生成 PCB 时用)
        self._wire_ends = {}

    # ---- 基础 ----
    def _uid(self, key=None) -> Str:
        if key is None:
            self._n += 1
            key = f"item{self._n}"
        return Str(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.project}/{key}")))

    def _symbol(self, lib_id: str):
        """从库里取出符号(展开 extends),返回 (嵌入用的定义, 引脚表, 属性表)。"""
        if lib_id in self._lib_syms:
            return self._lib_syms[lib_id]
        lib, name = lib_id.split(":")
        if lib not in self._libs:
            root = load(KICAD_SHARE / "symbols" / f"{lib}.kicad_sym")
            self._libs[lib] = {s[1]: s for s in find_all(root, "symbol")}
        src = self._libs[lib][name]
        ext = find(src, "extends")
        if ext:
            parent = self._libs[lib][ext[1]]
            sym = copy.deepcopy(parent)
            for sub in find_all(sym, "symbol"):
                sub[1] = Str(sub[1].replace(ext[1], name, 1))
            own = {p[1]: p for p in find_all(src, "property")}
            sym[:] = [own.pop(c[1], c) if isinstance(c, list) and c[0] == "property" else c for c in sym]
            sym.extend(copy.deepcopy(list(own.values())))
        else:
            sym = copy.deepcopy(src)
        sym[1] = Str(lib_id)
        pins = []
        for sub in find_all(sym, "symbol"):
            for pin in find_all(sub, "pin"):
                at = find(pin, "at")
                pins.append(dict(number=str(find(pin, "number")[1]), name=str(find(pin, "name")[1]),
                                 type=pin[1], x=float(at[1]), y=float(at[2]), angle=int(float(at[3]))))
        props = {p[1]: p for p in find_all(sym, "property")}
        self._lib_syms[lib_id] = (sym, pins, props)
        return self._lib_syms[lib_id]

    # ---- 图元 ----
    def wire(self, x1, y1, x2, y2):
        self.items.append(["wire", ["pts", ["xy", r2(x1), r2(y1)], ["xy", r2(x2), r2(y2)]],
                           ["stroke", ["width", 0], ["type", "default"]], ["uuid", self._uid()]])

    def junction(self, x, y):
        self.items.append(["junction", ["at", r2(x), r2(y)], ["diameter", 0], ["color", 0, 0, 0, 0],
                           ["uuid", self._uid()]])

    def no_connect(self, x, y):
        self.items.append(["no_connect", ["at", r2(x), r2(y)], ["uuid", self._uid()]])

    def text(self, s, x, y, size=2.5):
        self.items.append(["text", Str(s), ["exclude_from_sim", "no"], ["at", r2(x), r2(y), 0],
                           ["effects", ["font", ["size", size, size]], ["justify", "left", "bottom"]],
                           ["uuid", self._uid()]])

    def label(self, net, x, y, direction):
        angle, just = LABEL_STYLE[direction]
        self.items.append(["label", Str(net), ["at", r2(x), r2(y), angle],
                           ["effects", ["font", ["size", 1.27, 1.27]], ["justify", just, "bottom"]],
                           ["uuid", self._uid()]])

    def global_label(self, net, x, y, direction):
        angle, just = LABEL_STYLE[direction]
        self.items.append(["global_label", Str(net), ["shape", "passive"], ["at", r2(x), r2(y), angle],
                           ["effects", ["font", ["size", 1.27, 1.27]], ["justify", just]],
                           ["uuid", self._uid()],
                           ["property", Str("Intersheetrefs"), Str("${INTERSHEET_REFS}"), ["at", 0, 0, 0],
                            ["effects", ["font", ["size", 1.27, 1.27]], ["hide", "yes"]]]])

    def power(self, net, x, y):
        self._pwr_n += 1
        self._instance(f"power:{net}", f"#PWR{self._pwr_n:03d}", net, x, y, None)

    def _instance(self, lib_id, ref, value, x, y, footprint):
        _, pins, props = self._symbol(lib_id)
        node = ["symbol", ["lib_id", Str(lib_id)], ["at", r2(x), r2(y), 0], ["unit", 1],
                ["exclude_from_sim", "no"], ["in_bom", "yes"], ["on_board", "yes"], ["dnp", "no"],
                ["uuid", self._uid(f"sym/{ref}")]]
        for key in ("Reference", "Value", "Footprint", "Datasheet", "Description"):
            if key not in props:
                continue
            p = copy.deepcopy(props[key])
            if key == "Reference":
                p[2] = Str(ref)
            elif key == "Value":
                p[2] = Str(value)
            elif key == "Footprint" and footprint is not None:
                p[2] = Str(footprint)
            at = find(p, "at")
            at[1], at[2] = r2(x + float(at[1])), r2(y - float(at[2]))
            node.append(p)
        for num in sorted({p["number"] for p in pins}):
            node.append(["pin", Str(num), ["uuid", self._uid(f"pin/{ref}/{num}")]])
        node.append(["instances", ["project", Str(self.project),
                                   ["path", Str(f"/{self.root_uuid}"), ["reference", Str(ref)], ["unit", 1]]]])
        self.items.append(node)
        return pins, props

    # ---- 放置器件 ----
    def place(self, lib_id, ref, value, x, y, nets: dict, nc=(), footprint=None):
        """nets:{引脚名或引脚号: 网络名};nc:悬空引脚(名或号)。每个引脚都必须出现在其中之一。"""
        assert ref not in self.refs, f"位号重复 {ref}"
        self.refs.add(ref)
        pins, props = self._instance(lib_id, ref, value, x, y, footprint)
        fp = footprint if footprint is not None else str(props["Footprint"][2])
        assert fp, f"{ref} 没有封装"
        lib, name = fp.split(":")
        assert (KICAD_SHARE / "footprints" / f"{lib}.pretty" / f"{name}.kicad_mod").exists(), f"封装不存在 {fp}"
        self.parts[ref] = dict(value=value, footprint=fp, uuid=str(self._uid(f"sym/{ref}")))

        names = {p["name"] for p in pins}
        used = set()
        points = {}                                   # (x, y) → (网络或 None, 角度)
        for p in pins:
            if p["type"] == "no_connect":
                continue
            key = p["name"] if p["name"] in nets or p["name"] in nc else p["number"]
            if key == p["number"] and p["number"] in names and p["name"] != p["number"]:
                raise ValueError(f"{ref}: '{key}' 既是引脚名又是另一个引脚的编号")
            if key in nc:
                net = None
            elif key in nets:
                net, _, opt = nets[key].partition("|")
                self.net_pins.setdefault(net, []).append((ref, p["number"], p["type"]))
            else:
                raise ValueError(f"{ref} ({lib_id}) 引脚 {p['number']} '{p['name']}' 没有指定网络")
            used.add(key)
            pos = (r2(x + p["x"]), r2(y - p["y"]))
            if pos in points and points[pos][0] != net:
                raise ValueError(f"{ref}: 重叠引脚接了不同网络 {points[pos][0]} / {net}")
            points[pos] = (net, p["angle"], opt if net else "")
        unknown = (set(nets) | set(nc)) - used
        assert not unknown, f"{ref}: 表里有库中不存在的引脚 {unknown}"

        occupied = set(points)
        groups = {}                                   # 同方向、同电源网络、相邻的竖直引脚合并成一个符号
        for (px, py), (net, angle, opt) in sorted(points.items()):
            if net is None:
                self.no_connect(px, py)
                continue
            dx, dy = OUT_DIR[angle]
            want = POWER_NETS.get(net)
            blocked = (want and dy == 0
                       and (px, r2(py + (-2.54 if want == "up" else 2.54))) in occupied)
            length = STUB * (6 if blocked else 3 if opt == "long" else 1)   # 全局标签引远一点,避开相邻引脚的标签
            ex, ey = r2(px + dx * length), r2(py + dy * length)
            self.wire(px, py, ex, ey)
            if net not in POWER_NETS:
                if opt == "right":                      # 竖直引脚上的横排标签(上下两个器件直接对接时用)
                    self.label(net, ex, ey, (1, 0))
                elif opt != "nolabel":
                    self.label(net, ex, ey, (dx, dy))
                continue
            if dy != 0:
                if (dy < 0) != (want == "up"):
                    raise ValueError(f"{ref}: {net} 接在了方向相反的引脚上,请调整器件")
                groups.setdefault((net, ey), []).append(ex)
            else:
                # 侧面的电源引脚:符号会占用上方(电源)或下方(地)的位置,那里有相邻引脚时改用全局标签
                if blocked:
                    self.global_label(net, ex, ey, (dx, dy))
                else:
                    self.power(net, ex, ey)
        for (net, ey), xs in groups.items():
            xs.sort()
            runs = [[xs[0]]]
            for v in xs[1:]:
                if v - runs[-1][-1] <= 5.08 + 1e-6:
                    runs[-1].append(v)
                else:
                    runs.append([v])
            for run in runs:
                for a, b in zip(run, run[1:]):
                    self.wire(a, ey, b, ey)
                for v in run[1:-1]:
                    self.junction(v, ey)
                self.power(net, run[0], ey)

    def flag(self, net, x, y):
        """给没有电源输出引脚驱动的网络加 PWR_FLAG(告诉 ERC:这个网络确实有电源)。"""
        self._pwr_n += 1
        self._instance("power:PWR_FLAG", f"#FLG{self._pwr_n:03d}", "PWR_FLAG", x, y, None)
        self.wire(x, y, x + 10.16, y)
        if net in POWER_NETS:
            self.power(net, x + 10.16, y)
        else:
            self.label(net, x + 10.16, y, (1, 0))
        self.net_pins.setdefault(net, []).append(("#FLG", "1", "power_out"))

    def undriven_power_nets(self):
        out = []
        for net, pins in self.net_pins.items():
            types = {t for _, _, t in pins}
            if "power_in" in types and "power_out" not in types:
                out.append(net)
        return out

    def check(self):
        for net, pins in self.net_pins.items():
            if net not in POWER_NETS and len(pins) < 2:
                raise ValueError(f"网络 {net} 只接了一个引脚: {pins}")

    # ---- 输出 ----
    def write(self, path):
        self.check()
        libs = ["lib_symbols"] + [self._lib_syms[k][0] for k in sorted(self._lib_syms)]
        root = ["kicad_sch", ["version", 20250610], ["generator", Str("fc_schgen")],
                ["generator_version", Str("1.0")], ["uuid", self.root_uuid], ["paper", Str(self.paper)],
                ["title_block", ["title", Str(self.title)], ["rev", Str("0.1")]],
                libs, *self.items,
                ["sheet_instances", ["path", Str("/"), ["page", Str("1")]]],
                ["embedded_fonts", "no"]]
        Path(path).write_text(dump(root) + "\n", encoding="utf-8")
