"""长格式数据的读写、校验、聚合与切分。

数据约定（全工程唯一格式）
--------------------------
长格式表，一行 = 一个 (样本, 模型) 组合：
    sample_id   str    样本唯一标识（含数据集前缀）
    dataset     str    数据集名，取值见 constants.DATASETS
    model       str    模型名，取值见 constants.MODELS
    performance float  质量，[0, 1]；exact-match 数据集为 0/1，裁判数据集为连续值
    cost        float  该次推理的成本（美元）

注意：**按行切分训练/测试集会泄露**（同一样本的 11 个模型会被拆到两边），
必须按 `sample_id` 切分，见 `split_by_sample`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

from . import constants as C

REQUIRED_COLUMNS = ("sample_id", "dataset", "model", "performance", "cost")

#: 真实数据的列名可能不同，载入时统一改名
COLUMN_ALIASES: Dict[str, str] = {
    "id": "sample_id",
    "sample": "sample_id",
    "question_id": "sample_id",
    "dataset_name": "dataset",
    "task": "dataset",
    "model_name": "model",
    "llm": "model",
    "score": "performance",
    "quality": "performance",
    "performance_score": "performance",
    "price": "cost",
    "cost_usd": "cost",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: COLUMN_ALIASES[c] for c in df.columns if c in COLUMN_ALIASES}
    return df.rename(columns=rename)


def load_longframe(path: str | Path) -> pd.DataFrame:
    """读取长格式表，支持 csv / pkl / jsonl / parquet。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"数据文件不存在：{path}\n"
            "  * 若要跑真实数据：把 RouterBench 的 parquet/pkl 放进 data/raw/，"
            "再把 configs/default.json 的 data.source 改成 \"real\"；\n"
            "  * 若只想打通流程：先运行 scripts/00_make_surrogate.py。"
        )
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".pkl", ".pickle"):
        df = pd.read_pickle(path)
    elif suffix in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True)
    elif suffix == ".parquet":
        try:
            df = pd.read_parquet(path)
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的提示
            raise ImportError(
                "读取 parquet 需要 pyarrow 或 fastparquet：pip install pyarrow；"
                "或改用 .pkl 版本的数据文件。"
            ) from exc
    else:
        raise ValueError(f"不支持的文件类型：{suffix}")
    return _normalize_columns(df)


def save_longframe(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".csv":
        raise ValueError("统一保存为 csv，便于人工抽查")
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------
def validate_longframe(
    df: pd.DataFrame,
    models: Sequence[str] = C.MODELS,
    datasets: Sequence[str] = C.DATASETS,
    strict: bool = True,
) -> Dict[str, object]:
    """校验长格式表，返回报告字典；硬错误在 strict 下直接抛异常。"""
    errors: list[str] = []
    warnings: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必需列：{missing}；现有列：{list(df.columns)}")

    out = df.copy()
    out["performance"] = pd.to_numeric(out["performance"], errors="coerce")
    out["cost"] = pd.to_numeric(out["cost"], errors="coerce")
    out["sample_id"] = out["sample_id"].astype(str)
    out["dataset"] = out["dataset"].astype(str)
    out["model"] = out["model"].astype(str)

    if out[["performance", "cost"]].isna().any().any():
        errors.append("performance / cost 存在空值或非数值")

    if (out["cost"] < 0).any():
        errors.append(f"存在负成本：{(out['cost'] < 0).sum()} 行")
    if ((out["performance"] < 0) | (out["performance"] > 1)).any():
        bad = int(((out["performance"] < 0) | (out["performance"] > 1)).sum())
        errors.append(f"performance 越界 [0,1]：{bad} 行")

    dup = out.duplicated(subset=["sample_id", "model"]).sum()
    if dup:
        errors.append(f"(sample_id, model) 重复：{dup} 行")

    known_ds = set(datasets)
    unknown_ds = sorted(set(out["dataset"]) - known_ds)
    if unknown_ds:
        errors.append(f"出现未登记的数据集：{unknown_ds}")
    unknown_model = sorted(set(out["model"]) - set(models))
    if unknown_model:
        errors.append(f"出现未登记的模型：{unknown_model}")

    # 每个数据集内：样本数 × 模型数必须等于行数（不允许某模型缺样本）
    per_ds: Dict[str, Dict[str, int]] = {}
    for ds, g in out.groupby("dataset"):
        n_samples = g["sample_id"].nunique()
        counts = g.groupby("model").size()
        if len(counts) != len(models):
            errors.append(f"[{ds}] 模型数 {len(counts)} != {len(models)}")
        if counts.nunique() > 1:
            errors.append(f"[{ds}] 各模型行数不一致：{counts.to_dict()}")
        if int(counts.min()) != n_samples:
            errors.append(f"[{ds}] 行数 {int(counts.min())} != 样本数 {n_samples}")
        per_ds[ds] = {"n_samples": int(n_samples), "n_rows": int(len(g))}

    report: Dict[str, object] = {
        "n_rows": int(len(out)),
        "n_datasets": int(out["dataset"].nunique()),
        "n_models": int(out["model"].nunique()),
        "per_dataset": per_ds,
        "errors": errors,
        "warnings": warnings,
    }
    if errors and strict:
        raise ValueError("数据校验失败：\n  - " + "\n  - ".join(errors))
    return report


# --------------------------------------------------------------------------
# 聚合
# --------------------------------------------------------------------------
def aggregate(
    df: pd.DataFrame,
    models: Sequence[str] = C.MODELS,
    datasets: Sequence[str] = C.DATASETS,
    mode: str = "pooled",
) -> pd.DataFrame:
    """按 (dataset, model) 聚合出 Table 1 的两个量：perf 均值、cost 总和。

    mode="pooled"：对该数据集所有样本直接求均值（样本加权）。
    mode="macro" ：先按 eval_name 求均值，再对 eval_name 求平均（子任务等权）。
      论文官方实现（`evaluation/eval.py`）用的是后者：MMLU 有 57 个学科，
      各学科题数差异很大，等权平均与样本加权会差几个点。缺少 eval_name 列时自动退回 pooled。
    """
    if mode == "macro" and "eval_name" in df.columns:
        per = (
            df.groupby(["dataset", "eval_name", "model"], sort=False)
            .agg(n=("sample_id", "nunique"), p=("performance", "mean"), c=("cost", "sum"))
            .reset_index()
        )
        agg = (
            per.groupby(["dataset", "model"], sort=False)
            .agg(n=("n", "sum"), perf_mean=("p", "mean"), cost_sum=("c", "sum"))
            .reset_index()
        )
    elif mode not in ("pooled", "macro"):
        raise ValueError(f"未知的聚合模式：{mode}")
    else:
        agg = (
            df.groupby(["dataset", "model"], sort=False)
            .agg(n=("sample_id", "nunique"), perf_mean=("performance", "mean"), cost_sum=("cost", "sum"))
            .reset_index()
        )
    agg["cost_mean"] = agg["cost_sum"] / agg["n"]
    # 固定输出顺序，方便与论文逐格对照
    ds_order = {d: i for i, d in enumerate(datasets)}
    m_order = {m: i for i, m in enumerate(models)}
    agg["_d"] = agg["dataset"].map(ds_order)
    agg["_m"] = agg["model"].map(m_order)
    agg = agg.sort_values(["_d", "_m"]).drop(columns=["_d", "_m"])
    return agg.reset_index(drop=True)


def to_wide(
    df: pd.DataFrame,
    dataset: str,
    models: Sequence[str] = C.MODELS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """取出某数据集的 (performance, cost) 两张 n_samples × n_models 宽表。"""
    sub = df[df["dataset"] == dataset]
    if sub.empty:
        raise ValueError(f"数据集中没有 {dataset}")
    perf = sub.pivot(index="sample_id", columns="model", values="performance")
    cost = sub.pivot(index="sample_id", columns="model", values="cost")
    perf = perf.reindex(columns=list(models))
    cost = cost.reindex(columns=list(models))
    if perf.isna().any().any() or cost.isna().any().any():
        raise ValueError(f"[{dataset}] 宽表存在缺失，说明样本×模型不完整")
    return perf, cost


# --------------------------------------------------------------------------
# 切分与 Oracle
# --------------------------------------------------------------------------
def split_by_sample(
    perf: pd.DataFrame,
    cost: pd.DataFrame,
    test_size: float = C.TEST_SIZE,
    seed: int = 0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """按 sample_id（行）切分 70/30，返回 {"train": (Xp, Xc, ...)} 形式的位置索引。

    按 sample_id 切分而不是按行切分，是为了避免同一样本的 11 个模型跨集泄露。
    """
    rng = np.random.default_rng(seed)
    n = len(perf)
    idx = rng.permutation(n)
    n_test = int(round(n * test_size))
    test_idx, train_idx = np.sort(idx[:n_test]), np.sort(idx[n_test:])
    return {
        "train": (perf.values[train_idx], cost.values[train_idx], train_idx),
        "test": (perf.values[test_idx], cost.values[test_idx], test_idx),
    }


def oracle_select(perf: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Oracle 路由：每样本在「达到最高质量」的模型里选最便宜的那个。

    若某样本所有模型都答错（最高质量为 0），等价于选最便宜的那个。
    """
    perf = np.asarray(perf, dtype=float)
    cost = np.asarray(cost, dtype=float)
    best = perf.max(axis=1, keepdims=True)
    masked = np.where(perf >= best - 1e-12, cost, np.inf)
    return np.argmin(masked, axis=1)


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """每个数据集的样本数与总行数，用于数据卡。"""
    rows = []
    for ds, g in df.groupby("dataset", sort=False):
        rows.append(
            {
                "dataset": ds,
                "n_samples": int(g["sample_id"].nunique()),
                "n_models": int(g["model"].nunique()),
                "n_rows": int(len(g)),
                "perf_mean": float(g["performance"].mean()),
                "cost_sum": float(g["cost"].sum()),
            }
        )
    return pd.DataFrame(rows)
