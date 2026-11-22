"""Step 2：复现 Table 1 并与论文逐格对比（Phase 2/3 的验收关卡）。

用法：
    python scripts/step2_reproduce_table1.py [--source surrogate|real]
"""

from __future__ import annotations

import argparse

import _bootstrap

_bootstrap.setup()

from routerbench.config import load_config  # noqa: E402
from routerbench.pipeline import step_table1  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="复现 Table 1")
    ap.add_argument("--source", choices=["surrogate", "real"], default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.source:
        cfg.source = args.source
    res = step_table1(cfg)
    s = res["summary"]
    print(f"[step2] table -> {res['table1_csv']}")
    print(f"[step2] Perf 通过 {s['perf_pass']}/{s['n_cells']}（{s['perf_pass_rate']:.1%}）")
    print(f"[step2] Cost 通过 {s['cost_pass']}/{s['n_cells']}（{s['cost_pass_rate']:.1%}）")
    print(f"[step2] 最大 |Perf 偏差| = {s['max_abs_perf_diff']:.4f}")
    print(f"[step2] 最大相对 Cost 偏差 = {s['max_rel_cost_diff']:.2%}")
    o = s["oracle_row"]
    print(f"[step2] Oracle 行：Perf {o['perf_pass']}/7，Cost {o['cost_pass']}/7")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
