"""表格对比与 Markdown 序列化（不依赖 tabulate）。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from . import constants as C


def markdown_table(
    df: pd.DataFrame,
    floatfmt: str = "{:.3f}",
    index: bool = False,
) -> str:
    """把 DataFrame 渲染成 GitHub 风格的 Markdown 表。"""
    cols = list(df.columns)
    header = ([""] if index else []) + [str(c) for c in cols]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for idx, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else floatfmt.format(float(v)))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(([str(idx)] if index else []) + cells) + " |")
    return "\n".join(lines)


def matrix_markdown(
    mat: pd.DataFrame,
    floatfmt: str = "{:.3f}",
    index_name: str = "Model",
    caption: str | None = None,
) -> str:
    out = []
    if caption:
        out.append(caption)
        out.append("")
    body = mat.copy()
    body.index.name = index_name
    out.append(markdown_table(body, floatfmt=floatfmt, index=True))
    return "\n".join(out)


# --------------------------------------------------------------------------
# Table 1 对比
# --------------------------------------------------------------------------
def compare_table1(
    agg: pd.DataFrame,
    perf_tol: float = 0.01,
    z_tol: float = 3.0,
    cost_abs_tol: float = 0.01,
    cost_rel_tol: float = 0.01,
) -> pd.DataFrame:
    """逐格对比复现值与论文 Table 1。

    判定规则（写进报告，避免"肉眼看差不多"）：
      * perf：|diff| <= perf_tol，**或** |diff| 在二项分布标准误的 z_tol 倍以内
        （小数据集上 1% 的绝对偏差本来就是抽样噪声）；
      * cost：|diff| <= max(cost_abs_tol, cost_rel_tol * paper)，论文只保留 3 位小数。
    """
    paper_perf = C.table1_perf()
    paper_cost = C.table1_cost()
    oracle_perf = C.oracle_perf()
    oracle_cost = C.oracle_cost()
    rows = []
    for _, r in agg.iterrows():
        model, dataset = r["model"], r["dataset"]
        if model == "Oracle":  # Table 1 最后一行
            if dataset not in oracle_perf:
                continue
            p_paper, c_paper = oracle_perf[dataset], oracle_cost[dataset]
        else:
            key = (model, dataset)
            if key not in paper_perf:
                continue
            p_paper, c_paper = paper_perf[key], paper_cost[key]
        n = int(r["n"])
        p = float(r["perf_mean"])
        se = np.sqrt(max(p_paper * (1 - p_paper), 1e-12) / max(n, 1))
        z = abs(p - p_paper) / se
        rows.append(
            {
                "dataset": C.DATASET_LABELS[r["dataset"]],
                "model": r["model"],
                "n": n,
                "perf_repro": p,
                "perf_paper": p_paper,
                "perf_diff": p - p_paper,
                "perf_z": z,
                "perf_pass": bool(abs(p - p_paper) <= perf_tol or z <= z_tol),
                "cost_repro": float(r["cost_sum"]),
                "cost_paper": c_paper,
                "cost_diff": float(r["cost_sum"]) - c_paper,
                "cost_pass": bool(
                    abs(float(r["cost_sum"]) - c_paper)
                    <= max(cost_abs_tol, cost_rel_tol * abs(c_paper))
                ),
            }
        )
    return pd.DataFrame(rows)


def comparison_summary(cmp: pd.DataFrame) -> Dict[str, object]:
    n = len(cmp)
    perf_pass = int(cmp["perf_pass"].sum())
    cost_pass = int(cmp["cost_pass"].sum())
    worst = cmp.reindex(cmp["perf_diff"].abs().sort_values(ascending=False).index).head(5)
    return {
        "n_cells": n,
        "perf_pass": perf_pass,
        "cost_pass": cost_pass,
        "perf_pass_rate": perf_pass / n if n else 0.0,
        "cost_pass_rate": cost_pass / n if n else 0.0,
        "max_abs_perf_diff": float(cmp["perf_diff"].abs().max()) if n else 0.0,
        "max_rel_cost_diff": float(
            (cmp["cost_diff"].abs() / cmp["cost_paper"].abs().clip(lower=1e-12)).max()
        )
        if n
        else 0.0,
        "worst_cells": worst[["dataset", "model", "perf_diff", "perf_z"]].to_dict("records"),
    }


def build_matrices(agg: pd.DataFrame, value: str) -> pd.DataFrame:
    """把长表转成 行=模型、列=数据集 的矩阵。"""
    mat = agg.pivot(index="model", columns="dataset", values=value)
    mat = mat.reindex(index=list(C.MODELS), columns=list(C.DATASETS))
    mat.columns = [C.DATASET_LABELS[c] for c in mat.columns]
    return mat


def write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
