"""脚本共用：把工程根目录加入 sys.path，并统一控制台编码。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def setup() -> Path:
    """让 `python scripts/xxx.py` 能 import routerbench，并避免 Windows 控制台编码报错。"""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    _force_utf8_console()
    return ROOT


def _force_utf8_console() -> None:
    """Windows 默认 GBK(cp936) 控制台遇到 ✓/⚠ 会抛 UnicodeEncodeError。

    统一改成 UTF-8 并允许替换不可编码字符：
      * Windows Terminal / VS Code 终端可直接正常显示中文；
      * 传统 cmd 需要先执行 `chcp 65001`（README 里已注明）。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # 被重定向到不支持改编码的对象时静默跳过
            pass
