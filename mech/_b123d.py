"""导入 build123d(Python 的 CAD 库,底层是 OpenCascade 几何内核)。

这台机器上 scikit-learn 的一个 DLL 被 Windows 的应用程序控制策略拦截,而 build123d 启动时
会导入它(只用于"从 STL 反推实体"这一个功能,本项目用不到)。这里不去改系统设置,
而是在导入前放一个空的占位模块,让 build123d 跳过那个功能。其余功能(实体建模、布尔运算、
STEP 导入导出)不受影响。如果以后 scikit-learn 能正常加载,这个占位不会生效。
"""
import sys
import types

try:
    import sklearn.cluster  # noqa: F401
except Exception:                                   # noqa: BLE001
    for name in [m for m in sys.modules if m == "sklearn" or m.startswith("sklearn.")]:
        del sys.modules[name]
    stub = types.ModuleType("sklearn")
    cluster = types.ModuleType("sklearn.cluster")
    cluster.DBSCAN = None
    stub.cluster = cluster
    sys.modules["sklearn"], sys.modules["sklearn.cluster"] = stub, cluster

import build123d as bd  # noqa: E402

__all__ = ["bd"]
