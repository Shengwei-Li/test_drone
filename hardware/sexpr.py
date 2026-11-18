"""KiCad 文件用的 S 表达式:最小的解析器和输出器。

解析结果是嵌套列表;带引号的字符串用 Str 类型保存,输出时还原引号。
"""
from pathlib import Path


class Str(str):
    """原文件里带引号的字符串。"""


def parse(text: str):
    pos, n = 0, len(text)
    stack, cur = [], []
    while pos < n:
        c = text[pos]
        if c == "(":
            stack.append(cur)
            cur = []
            pos += 1
        elif c == ")":
            done, cur = cur, stack.pop()
            cur.append(done)
            pos += 1
        elif c == '"':
            pos += 1
            buf = []
            while text[pos] != '"':
                if text[pos] == "\\":
                    buf.append(text[pos:pos + 2])
                    pos += 2
                else:
                    buf.append(text[pos])
                    pos += 1
            cur.append(Str("".join(buf)))
            pos += 1
        elif c.isspace():
            pos += 1
        else:
            end = pos
            while end < n and not text[end].isspace() and text[end] not in "()":
                end += 1
            cur.append(text[pos:end])
            pos = end
    return cur[0]


def dump(node, indent: int = 0) -> str:
    if isinstance(node, Str):
        return f'"{node}"'
    if not isinstance(node, list):
        if isinstance(node, float):
            return f"{node:.4f}".rstrip("0").rstrip(".")
        return str(node)
    simple = all(not isinstance(c, list) for c in node)
    if simple:
        return "(" + " ".join(dump(c) for c in node) + ")"
    pad = "\t" * (indent + 1)
    k = 0
    while k < len(node) and not isinstance(node[k], list):
        k += 1
    head = [dump(c) for c in node[:k]]                      # 开头的原子写在同一行
    body = [pad + dump(c, indent + 1) for c in node[k:]]    # 其余保持原顺序,各占一行
    return "(" + " ".join(head) + "\n" + "\n".join(body) + "\n" + "\t" * indent + ")"


def find(node, tag: str):
    """第一个以 tag 开头的子表。"""
    for c in node:
        if isinstance(c, list) and c and c[0] == tag:
            return c
    return None


def find_all(node, tag: str):
    return [c for c in node if isinstance(c, list) and c and c[0] == tag]


def load(path) -> list:
    return parse(Path(path).read_text(encoding="utf-8"))
