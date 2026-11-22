"""把 HuggingFace 上的 RouterBench 原始宽表转成本仓库的长格式。

用法：
    python scripts/step0b_convert_real.py [--force] [--config configs/real.json]

产出：
    data/real/routerbench_long.pkl        长表缓存（下一步直接读它）
    data/derived*/real_data_card.json    转换参数与每个数据集的规模
"""

from __future__ import annotations

import argparse
import json

import _bootstrap

_bootstrap.setup()

from routerbench import realdata, report  # noqa: E402
from routerbench.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="转换 HF 原始宽表 -> 长表")
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="忽略缓存重新转换")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.ensure_dirs()

    zero = cfg.raw_dir / "routerbench_0shot.pkl"
    five = cfg.raw_dir / "routerbench_5shot.pkl"
    for p in (zero, five):
        if not p.exists():
            print(f"缺少原始文件：{p}")
            print("先跑：python scripts/download_data.py")
            return 2

    data_cfg = cfg.raw.get("data", {})
    shot_map = data_cfg.get("shot_map") or realdata.DEFAULT_SHOT_MAP
    shot_map = {k: int(v) for k, v in shot_map.items()}
    split_filter = data_cfg.get("split_filter", realdata.DEFAULT_SPLIT_FILTER)

    long = realdata.load_or_build(
        zero,
        five,
        cache_path=cfg.real_path,
        force=args.force,
        shot_map=shot_map,
        split_filter=split_filter,
    )
    summary = realdata.summarize(long)

    card = {
        "kind": "real",
        "source": "huggingface.co/datasets/withmartian/routerbench",
        "files": [zero.name, five.name],
        "shot_map": shot_map,
        "split_filter": split_filter,
        "aggregate_mode": cfg.aggregate_mode,
        "n_rows": int(len(long)),
        "per_dataset": summary.to_dict("records"),
        "note": (
            "直接由公开 pkl 转换，未经任何重采样；"
            "RAG 子集与论文自建的 800 条 Martian 查询不同，默认丢弃。"
        ),
    }
    card_path = report.write_text(
        cfg.derived_dir / "real_data_card.json",
        json.dumps(card, ensure_ascii=False, indent=2, default=str),
    )

    print(summary.to_string(index=False))
    print(f"\n长表 -> {cfg.real_path}（{len(long):,} 行）")
    print(f"数据卡 -> {card_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
