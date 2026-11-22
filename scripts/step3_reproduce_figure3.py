"""Step 3：复现 Figure 3（cost-quality 散点 + Oracle 选择分布）并计算 AIQ。

用法：
    python scripts/step3_reproduce_figure3.py [--source surrogate|real]
"""

from __future__ import annotations

import argparse

import _bootstrap

_bootstrap.setup()

from routerbench.config import load_config  # noqa: E402
from routerbench.pipeline import step_figure3  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="复现 Figure 3")
    ap.add_argument("--source", choices=["surrogate", "real"], default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.source:
        cfg.source = args.source
    res = step_figure3(cfg)
    for f in res["figures"]:
        print(f"[step3] figure -> {f}")
    print("[step3] Zero router vs 最好单模型（AIQ）：")
    for row in res["zero_beats_best_single"]:
        flag = "✓" if row["zero_beats_best_single"] else "✗"
        print(
            f"        {flag} {row['dataset']:<12} zero={row['zero_aiq']:.3f}"
            f"  best_single={row['best_single_aiq']:.3f} ({row['best_single']})"
        )
    print("[step3] 相近质量下的成本倍数：")
    for row in res["cost_ratio"]:
        print(
            f"        {row['dataset']:<12} quality>={row['quality_threshold']:.3f}"
            f"  {row['cheapest']} -> {row['priciest']}  ×{row['cost_ratio']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
