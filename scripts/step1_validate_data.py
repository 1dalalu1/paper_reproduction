"""Step 1：数据校验（Phase 1 的验收关卡）。

用法：
    python scripts/step1_validate_data.py [--source surrogate|real]
"""

from __future__ import annotations

import argparse

import _bootstrap

_bootstrap.setup()

from routerbench.config import load_config  # noqa: E402
from routerbench.pipeline import step_validate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="校验长格式数据")
    ap.add_argument("--source", choices=["surrogate", "real"], default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.source:
        cfg.source = args.source
    res = step_validate(cfg)
    rep = res["report"]
    print(f"[step1] file   = {res['path']}")
    print(f"[step1] source = {res['source']}")
    print(f"[step1] rows   = {rep['n_rows']:,}  datasets={rep['n_datasets']}  models={rep['n_models']}")
    for row in res["per_dataset"]:
        print(
            f"        - {row['dataset']:<12} n={row['n_samples']:>6}  rows={row['n_rows']:>7}"
            f"  perf={row['perf_mean']:.3f}  cost={row['cost_sum']:.3f}"
        )
    print(f"[step1] report -> {res['markdown']}")
    if rep["errors"]:
        print("[step1] FAILED")
        for e in rep["errors"]:
            print("   !", e)
        return 1
    print("[step1] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
