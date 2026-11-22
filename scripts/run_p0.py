"""一键跑完 P0 基线（Phase 0–4）。

用法：
    python scripts/run_p0.py [--source surrogate|real] [--force-surrogate]
"""

from __future__ import annotations

import argparse
import time

import _bootstrap

_bootstrap.setup()

from routerbench.config import load_config  # noqa: E402
from routerbench.pipeline import run_all  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="P0 基线一键复现")
    ap.add_argument("--source", choices=["surrogate", "real"], default=None)
    ap.add_argument("--force-surrogate", action="store_true", help="重新生成合成数据")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.source:
        cfg.source = args.source

    t0 = time.time()
    print("=" * 72)
    print(f"RouterBench P0 基线复现   source={cfg.source}   root={cfg.root}")
    print("=" * 72)

    out = run_all(cfg, force_surrogate=args.force_surrogate)

    s1, s2, s3 = out["step1_validate"], out["step2_table1"], out["step3_figure3"]
    print(f"\n[1/3] 数据校验：{s1['report']['n_rows']:,} 行，errors={len(s1['report']['errors'])}")
    print(f"      报告 -> {s1['markdown']}")

    sm = s2["summary"]
    print(f"\n[2/3] Table 1：Perf {sm['perf_pass']}/{sm['n_cells']}  "
          f"Cost {sm['cost_pass']}/{sm['n_cells']}  maxΔperf={sm['max_abs_perf_diff']:.4f}")
    print(f"      表格 -> {s2['table1_csv']}")

    print("\n[3/3] Figure 3：")
    for f in s3["figures"]:
        print(f"      -> {f}")

    print(f"\n总耗时 {time.time() - t0:.1f}s")
    if cfg.source == "surrogate":
        print("\n⚠ 本次使用 surrogate 合成数据：Figure/Table 的数值一致性来自构造，")
        print("  只证明代码链路正确，不构成对论文结论的复现证据。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
