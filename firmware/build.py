"""把固件核心库编译成软件在环用的动态库(单精度和双精度各一份)。

用法:  python firmware/build.py
输出:  firmware/build/fc_core_f32.dll、fc_core_f64.dll

Windows 上用 Visual Studio 生成工具里的 cl(通过 vswhere 找);其他系统用 cc。
核心库本身是标准 C99,不依赖编译器,上板时用 arm-none-eabi-gcc 编译同一批源文件。
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
BUILD = HERE / "build"
SOURCES = [*sorted((HERE / "core").glob("*.c")), HERE / "sil" / "fc_sil.c"]
EXT = ".dll" if os.name == "nt" else ".so"


def lib_path(precision: str) -> Path:
    return BUILD / f"fc_core_{precision}{EXT}"


def _vcvars() -> Path | None:
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.exists():
        return None
    out = subprocess.run([str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
                         capture_output=True, text=True).stdout.strip()
    bat = Path(out) / "VC/Auxiliary/Build/vcvars64.bat"
    return bat if out and bat.exists() else None


def build(precision: str) -> Path:
    BUILD.mkdir(exist_ok=True)
    out = lib_path(precision)
    define = ["FC_DOUBLE"] if precision == "f64" else []
    srcs = [str(s) for s in SOURCES]
    if os.name == "nt":
        bat = _vcvars()
        if bat is None:
            raise RuntimeError("没找到 Visual Studio 的 C 编译器")
        obj = BUILD / f"obj_{precision}"
        obj.mkdir(exist_ok=True)
        # /fp:precise:不做会改变结果的浮点优化,保证与 Python 对照时可重复
        cl = (f'cl /nologo /utf-8 /O2 /fp:precise /W3 /LD {" ".join("/D" + d for d in define)} '
              f'/Fo"{obj}\\\\" /Fe"{out}" ' + " ".join(f'"{s}"' for s in srcs))
        r = subprocess.run(f'cmd /c ""{bat}" >nul && {cl}"', capture_output=True, text=True, shell=True,
                           encoding="utf-8", errors="replace")
    else:
        cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        r = subprocess.run([cc, "-O2", "-std=c99", "-shared", "-fPIC", "-Wall", *("-D" + d for d in define),
                            "-o", str(out), *srcs, "-lm"], capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"编译失败:\n{r.stdout}\n{r.stderr}")
    if "warning" in r.stdout + r.stderr:
        print(r.stdout + r.stderr)
    return out


def ensure(precision: str) -> Path:
    """库不存在或比源文件旧时重新编译。"""
    out = lib_path(precision)
    newest = max(p.stat().st_mtime for p in (HERE / "core").glob("*")) if out.exists() else 0
    newest = max(newest, (HERE / "sil" / "fc_sil.c").stat().st_mtime)
    if not out.exists() or out.stat().st_mtime < newest:
        build(precision)
    return out


if __name__ == "__main__":
    for prec in ("f32", "f64"):
        print("已编译", build(prec))
    sys.exit(0)
