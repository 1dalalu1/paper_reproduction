"""Step 0：生成 / 复用合成替代数据。

用法：
    python scripts/step0_make_surrogate.py [--force]
"""

from __future__ import annotations

import argparse

import _bootstrap

_bootstrap.setup()

from routerbench.config import load_config  # noqa: E402
from routerbench.pipeline import build_surrogate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 surrogate 数据")
    ap.add_argument("--force", action="store_true", help="已存在也重新生成")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    info = build_surrogate(cfg, force=args.force)
    print(f"[step0] {info['status']}: {info['path']}")
    print(f"[step0] rows = {info['n_rows']:,}")
    if "data_card" in info:
        print(f"[step0] data card -> {info['data_card']}")
    print("[step0] 注意：这是合成数据，仅用于打通与自测流程。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
