"""无 pytest 环境下的极简测试运行器：发现并执行 tests/test_*.py 里的 test_* 函数。

用法：
    python tests/run_tests.py [-v]
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _prepare() -> None:
    for p in (str(ROOT), str(HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)


def discover() -> list[str]:
    return sorted(p.stem for p in HERE.glob("test_*.py"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    _prepare()

    total = passed = 0
    failures: list[tuple[str, str]] = []

    for mod_name in discover():
        mod = importlib.import_module(mod_name)
        names = [n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
        for name in names:
            fn = getattr(mod, name)
            total += 1
            try:
                fn()
                passed += 1
                if args.verbose:
                    print(f"  PASS  {mod_name}::{name}")
            except Exception:
                failures.append((f"{mod_name}::{name}", traceback.format_exc()))
                print(f"  FAIL  {mod_name}::{name}")

    print(f"\n{passed}/{total} passed")
    for name, tb in failures:
        print("\n" + "=" * 70)
        print(name)
        print("-" * 70)
        print(tb)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
