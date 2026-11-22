"""把 HuggingFace 上的 RouterBench 原始宽表转成仓库统一的长格式。

原始文件（`withmartian/routerbench`）有两个版本：

* `routerbench_0shot.pkl`  —— 0-shot 生成
* `routerbench_5shot.pkl`  —— 5-shot 生成

两者都是**宽表**：一行一个 prompt，`sample_id` / `prompt` / `eval_name` 之外，
每个模型占三列——质量分（列名就是模型名）、`<model>|model_response`、
`<model>|total_cost`（该次推理的美元成本）。没有 `dataset` / `performance` / `cost` 这样的列，
所以必须做一次「宽转长 + 数据集归并 + 版本选择」的转换，才能进入本仓库的流水线。

论文的推理设置是：MMLU / HellaSwag / Winogrande / GSM8K / ARC 用 5-shot，
MBPP / MT-Bench 用 0-shot（RAG 子集论文用自建查询，公开数据里对应的是另一批中文/商业 prompt，
不在 Table 1 里，默认丢弃）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from . import constants as C

# --------------------------------------------------------------------------
# 列名映射：HF 宽表里的模型列名 → 本仓库的短名（与 constants.MODELS 一致）
# --------------------------------------------------------------------------
MODEL_COLUMNS: Dict[str, str] = {
    "WizardLM/WizardLM-13B-V1.2": "WizardLM 13B",
    "mistralai/mistral-7b-chat": "Mistral 7B",
    "mistralai/mixtral-8x7b-chat": "Mixtral 8x7B",
    "meta/code-llama-instruct-34b-chat": "Code Llama 34B",
    "zero-one-ai/Yi-34B-Chat": "Yi 34B",
    "gpt-3.5-turbo-1106": "GPT-3.5",
    "claude-instant-v1": "Claude Instant V1",
    "meta/llama-2-70b-chat": "Llama 70B",
    "claude-v1": "Claude V1",
    "claude-v2": "Claude V2",
    "gpt-4-1106-preview": "GPT-4",
}

# 每个数据集用哪个版本（论文的推理设置）
DEFAULT_SHOT_MAP: Dict[str, int] = {
    "mmlu": 5,
    "hellaswag": 5,
    "winogrande": 5,
    "gsm8k": 5,
    "arc": 5,
    "mbpp": 0,
    "mt_bench": 0,
}

# ARC 官方 test 划分是 1172 题，与论文 Table 1 的样本数一致；公开 5-shot 文件里还混了 312 行 val
DEFAULT_SPLIT_FILTER: Dict[str, Optional[str]] = {"arc": "test"}

# 归并后保留哪些数据集（RAG 子集不在论文 Table 1 里）
DEFAULT_KEEP = tuple(C.DATASETS)


def map_eval_name(eval_name: str) -> str:
    """把 HF 的 86 个 eval_name 归并成 7 个数据集名（+ rag）。"""
    if eval_name.startswith("mmlu-"):
        return "mmlu"
    if eval_name == "hellaswag":
        return "hellaswag"
    if eval_name == "winogrande":
        return "winogrande"
    if eval_name == "arc-challenge":
        return "arc"
    if eval_name == "grade-school-math":
        return "gsm8k"
    if eval_name == "mbpp":
        return "mbpp"
    if eval_name.startswith("mtbench"):
        return "mt_bench"
    return "rag"


_SPLIT_RE = re.compile(r"\.(val|dev|test|s1)\.\d+$")


def split_of(sample_id: str) -> str:
    """`grade-school-math.dev.42` → `dev`（原数据集自带的划分，不是 RouterBench 的 train/test）。"""
    m = _SPLIT_RE.search(sample_id)
    return m.group(1) if m else "unknown"


def _wide_to_long(
    wide: pd.DataFrame,
    shot: int,
    keep: Iterable[str] = DEFAULT_KEEP,
    split_filter: Optional[Dict[str, Optional[str]]] = None,
) -> pd.DataFrame:
    """单个宽表 → 长表（一个样本 × 11 行）。"""
    split_filter = DEFAULT_SPLIT_FILTER if split_filter is None else split_filter
    df = wide.copy()
    df["dataset"] = df["eval_name"].map(map_eval_name)
    df["split"] = df["sample_id"].map(split_of)

    model_cols = [c for c in MODEL_COLUMNS if c in df.columns]
    missing = [c for c in MODEL_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"{shot}-shot 文件缺少模型列：{missing}")
    if list(df["dataset"].map(lambda d: d in keep)).count(False) and shot == 0:
        pass  # 只做提示，不打断

    # 丢掉整行（所有模型）都为 NaN 的记录：公开 5-shot 文件里有 14 行这样的重复样本
    all_nan = df[model_cols].isna().all(axis=1)
    n_dropped = int(all_nan.sum())
    df = df[~all_nan].copy()

    rows: List[pd.DataFrame] = []
    for col in model_cols:
        cost_col = f"{col}|total_cost"
        part = pd.DataFrame(
            {
                "sample_id": df["sample_id"].values,
                "dataset": df["dataset"].values,
                "eval_name": df["eval_name"].values,
                "split": df["split"].values,
                "model": MODEL_COLUMNS[col],
                "performance": df[col].astype("float64").fillna(0.0).to_numpy(dtype=float),
                "cost": df[cost_col].astype("float64").fillna(0.0).to_numpy(dtype=float),
                "shot": shot,
            }
        )
        rows.append(part)
    long = pd.concat(rows, ignore_index=True)

    # 数据集级过滤：保留哪个 split、是否丢弃 RAG
    mask = long["dataset"].isin(list(keep))
    for ds, split in (split_filter or {}).items():
        if split is None:
            continue
        mask &= ~((long["dataset"] == ds) & (long["split"] != split))
    long = long[mask].copy()

    # 同一 (dataset, sample_id) 理论上只出现一次；若重复保留第一次
    long = long.drop_duplicates(subset=["dataset", "sample_id", "model"], keep="first")
    long["_dropped_all_nan"] = n_dropped  # 仅作信息，不参与后续计算
    return long.drop(columns=["_dropped_all_nan"])


def build_longframe(
    zero_path: str | Path,
    five_path: str | Path,
    shot_map: Optional[Dict[str, int]] = None,
    keep: Iterable[str] = DEFAULT_KEEP,
    split_filter: Optional[Dict[str, Optional[str]]] = None,
) -> pd.DataFrame:
    """读两个版本的宽表，按论文的 shot 设置拼成一张长表。"""
    shot_map = DEFAULT_SHOT_MAP if shot_map is None else shot_map
    five = pd.read_pickle(Path(five_path))
    zero = pd.read_pickle(Path(zero_path))

    parts: List[pd.DataFrame] = []
    for shot, wide in ((5, five), (0, zero)):
        datasets = [d for d, s in shot_map.items() if s == shot and d in keep]
        if not datasets:
            continue
        sub = wide[wide["eval_name"].map(map_eval_name).isin(datasets)].copy()
        if sub.empty:
            continue
        parts.append(
            _wide_to_long(sub, shot=shot, keep=datasets, split_filter=split_filter)
        )

    long = pd.concat(parts, ignore_index=True)
    # 固定顺序，便于复现与 diff
    ds_order = {d: i for i, d in enumerate(C.DATASETS)}
    long["_d"] = long["dataset"].map(ds_order)
    long = long.sort_values(["_d", "dataset", "sample_id", "model"]).drop(columns=["_d"])
    return long.reset_index(drop=True)


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    """转换后每个数据集的行数、样本数、模型数、性能均值、成本总和。"""
    rows = []
    for ds, g in long.groupby("dataset", sort=False):
        rows.append(
            {
                "dataset": ds,
                "shot": int(g["shot"].iloc[0]),
                "n_samples": int(g["sample_id"].nunique()),
                "n_models": int(g["model"].nunique()),
                "n_rows": int(len(g)),
                "perf_mean": float(g["performance"].mean()),
                "cost_sum": float(g["cost"].sum()),
            }
        )
    return pd.DataFrame(rows)


def load_or_build(
    zero_path: str | Path,
    five_path: str | Path,
    cache_path: str | Path,
    force: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """有缓存就用缓存（pkl），否则转换并落盘。"""
    cache = Path(cache_path)
    if cache.exists() and not force:
        return pd.read_pickle(cache)
    long = build_longframe(zero_path, five_path, **kwargs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    long.to_pickle(cache)
    return long


if __name__ == "__main__":  # 手工排查用
    import sys

    root = Path(__file__).resolve().parents[1]
    df = build_longframe(
        root / "data/raw/routerbench_0shot.pkl",
        root / "data/raw/routerbench_5shot.pkl",
    )
    print(summarize(df).to_string(index=False))
    sys.exit(0)
