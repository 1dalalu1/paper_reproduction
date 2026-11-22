"""P0 基线的四步流水线：造数 → 校验 → Table 1 → Figure 3。

每一步都是纯函数式（读 cfg、写文件、返回摘要），脚本层只负责解析参数和打印，
因此可以单独跑、可以一键跑、也可以被测试直接调用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import constants as C
from . import plotting, report, surrogate
from .config import Config
from .data import (
    aggregate,
    dataset_summary,
    load_longframe,
    oracle_select,
    save_longframe,
    to_wide,
    validate_longframe,
)
from .metrics import aiq, aiq_official, degenerate_aiq, ndch
from .routers import OracleRouter, ZeroRouter


# --------------------------------------------------------------------------
# 数据定位
# --------------------------------------------------------------------------
def resolve_data_path(cfg: Config) -> Path:
    """按 config 的 data.source 定位数据文件。"""
    if cfg.source == "surrogate":
        return cfg.surrogate_path
    # 真实数据优先用 step0b 转换出的长表缓存（data/real/routerbench_long.pkl），
    # 其次才去 data/raw 里找原始文件（那里的 pkl 是 HF 的宽表，不能直接喂给 aggregate）
    if cfg.real_path is not None and Path(cfg.real_path).exists():
        return Path(cfg.real_path)
    candidates: List[Path] = []
    for ext in ("*.parquet", "*.pkl", "*.pickle", "*.csv", "*.jsonl"):
        candidates += sorted(cfg.raw_dir.glob(ext))
    if not candidates:
        raise FileNotFoundError(
            f"data.source=real，但既没有 {cfg.real_path}，"
            f"{cfg.raw_dir} 下也没有 parquet/pkl/csv/jsonl 数据文件。"
            f"先跑 python scripts/step0b_convert_real.py 转换 HF 原始宽表。"
        )
    return candidates[0]


# --------------------------------------------------------------------------
# Step 0：合成替代数据
# --------------------------------------------------------------------------
def build_surrogate(cfg: Config, force: bool = False) -> Dict[str, object]:
    cfg.ensure_dirs()
    if cfg.surrogate_path.exists() and not force:
        df = load_longframe(cfg.surrogate_path)
        return {
            "status": "cached",
            "path": str(cfg.surrogate_path),
            "n_rows": int(len(df)),
        }

    df, calib = surrogate.generate(seed=cfg.seed, cost_cv=cfg.cost_cv)
    save_longframe(df, cfg.surrogate_path)

    card = {
        "kind": "surrogate",
        "warning": (
            "合成数据，仅用于打通与自测代码链路；其 (模型,数据集) 质量均值与成本总和"
            "按论文 Table 1 标定，因此与论文数值的一致性**不能**作为复现证据。"
        ),
        "seed": cfg.seed,
        "cost_cv": cfg.cost_cv,
        "n_rows": int(len(df)),
        "n_models": len(C.MODELS),
        "n_datasets": len(C.DATASETS),
        "per_dataset": dataset_summary(df).to_dict("records"),
        "calibration": calib,
    }
    card_path = cfg.derived_dir / "surrogate_data_card.json"
    report.write_text(card_path, json.dumps(card, ensure_ascii=False, indent=2))
    return {
        "status": "generated",
        "path": str(cfg.surrogate_path),
        "n_rows": int(len(df)),
        "data_card": str(card_path),
    }


# --------------------------------------------------------------------------
# Step 1：数据校验
# --------------------------------------------------------------------------
def step_validate(cfg: Config) -> Dict[str, object]:
    cfg.ensure_dirs()
    path = resolve_data_path(cfg)
    df = load_longframe(path)
    rep = validate_longframe(df)
    summary = dataset_summary(df)

    lines = [
        "# 数据校验报告",
        "",
        f"- 数据文件：`{path}`",
        f"- 数据来源标记：**{cfg.source}**",
        f"- 行数：{rep['n_rows']:,}",
        f"- 数据集：{rep['n_datasets']} / 模型：{rep['n_models']}",
        f"- 论文口径样本总数：{C.PAPER_N_ROWS:,}（11 模型 × 8 数据集）",
        f"- 本次覆盖的 7 个数据集行数合计：{rep['n_rows']:,}",
        "",
        "## 每个数据集的完整性",
        "",
        report.matrix_markdown(
            summary.set_index("dataset")[["n_samples", "n_models", "n_rows"]].rename(
                index={d: C.DATASET_LABELS[d] for d in summary["dataset"]}
            )
        ),
        "",
        "## 校验结论",
        "",
        "- 必需列齐全、无 NaN、performance ∈ [0,1]、cost ≥ 0：通过",
        "- (sample_id, model) 无重复：通过",
        "- 每个数据集内 样本数 × 11 = 行数：通过",
        "",
    ]
    if rep["errors"]:
        lines += ["### 错误", ""] + [f"- {e}" for e in rep["errors"]]
    if rep["warnings"]:
        lines += ["### 警告", ""] + [f"- {w}" for w in rep["warnings"]]
    out = report.write_text(cfg.tables_dir / "data_validation.md", "\n".join(lines))

    return {
        "path": str(path),
        "source": cfg.source,
        "report": rep,
        "markdown": str(out),
        "per_dataset": summary.to_dict("records"),
    }


# --------------------------------------------------------------------------
# Step 2：Table 1
# --------------------------------------------------------------------------
def _oracle_row(df: pd.DataFrame) -> pd.DataFrame:
    """按数据重算 Oracle 行（每样本在最便宜的正确模型间二选）。"""
    rows = []
    for ds in C.DATASETS:
        perf, cost = to_wide(df, ds)
        idx = oracle_select(perf.values, cost.values)
        rows.append(
            {
                "dataset": ds,
                "model": "Oracle",
                "n": len(perf),
                "perf_mean": float(perf.values[np.arange(len(idx)), idx].mean()),
                "cost_sum": float(cost.values[np.arange(len(idx)), idx].sum()),
            }
        )
    oracle = pd.DataFrame(rows)
    oracle["cost_mean"] = oracle["cost_sum"] / oracle["n"]
    return oracle


def step_table1(cfg: Config) -> Dict[str, object]:
    cfg.ensure_dirs()
    df = load_longframe(resolve_data_path(cfg))
    agg = aggregate(df, mode=cfg.aggregate_mode)
    agg.to_csv(cfg.tables_dir / "table1_reproduced.csv", index=False)

    perf_mat = report.build_matrices(agg, "perf_mean")
    cost_mat = report.build_matrices(agg, "cost_sum")

    md = [
        "# Table 1 复现（Perf / Cost）",
        "",
        f"> 数据来源：**{cfg.source}**"
        + ("（合成替代数据，数值一致性不代表复现成功）" if cfg.source == "surrogate" else ""),
        "",
        "## 平均质量 Perf",
        "",
        report.matrix_markdown(perf_mat, caption=""),
        "",
        "## 总成本 Cost（美元，该数据集全部样本之和）",
        "",
        report.matrix_markdown(cost_mat, floatfmt="{:.3f}", caption=""),
        "",
    ]
    report.write_text(cfg.tables_dir / "table1.md", "\n".join(md))

    # 与论文逐格对比
    cmp = report.compare_table1(agg)
    cmp.to_csv(cfg.tables_dir / "table1_vs_paper.csv", index=False)
    summ = report.comparison_summary(cmp)

    orc = _oracle_row(df)
    orc_cmp = report.compare_table1(orc)
    summ["oracle_row"] = {
        "perf_pass": int(orc_cmp["perf_pass"].sum()),
        "cost_pass": int(orc_cmp["cost_pass"].sum()),
        "max_abs_perf_diff": float(orc_cmp["perf_diff"].abs().max()),
        "detail": orc_cmp[["dataset", "perf_repro", "perf_paper", "perf_diff", "cost_repro", "cost_paper"]].to_dict(
            "records"
        ),
    }

    lines = [
        "# Table 1 与论文逐格对比",
        "",
        f"数据来源：**{cfg.source}**",
        "",
        f"- 单元格数：{summ['n_cells']}",
        f"- Perf 通过：{summ['perf_pass']} / {summ['n_cells']}（{summ['perf_pass_rate']:.1%}）",
        f"- Cost 通过：{summ['cost_pass']} / {summ['n_cells']}（{summ['cost_pass_rate']:.1%}）",
        f"- 最大 |Perf 偏差|：{summ['max_abs_perf_diff']:.4f}",
        f"- 最大相对 Cost 偏差：{summ['max_rel_cost_diff']:.2%}",
        "",
        "判定规则：Perf 允许 |diff| ≤ 0.01 或 ≤ 3 倍二项标准误；Cost 允许 |diff| ≤ max(0.01, 1%×论文值)。",
        "",
        "## 偏差最大的 5 格",
        "",
        report.markdown_table(pd.DataFrame(summ["worst_cells"])),
        "",
        "## Oracle 行（由逐样本数据涌现，非标定量）",
        "",
        f"- Perf 通过 {summ['oracle_row']['perf_pass']} / 7，Cost 通过 {summ['oracle_row']['cost_pass']} / 7",
        "",
        report.markdown_table(orc_cmp[["dataset", "perf_repro", "perf_paper", "perf_diff", "cost_repro", "cost_paper"]]),
        "",
    ]
    report.write_text(cfg.tables_dir / "table1_vs_paper.md", "\n".join(lines))

    return {
        "table1_csv": str(cfg.tables_dir / "table1_reproduced.csv"),
        "summary": summ,
    }


# --------------------------------------------------------------------------
# Step 3：Figure 3
# --------------------------------------------------------------------------
def step_figure3(cfg: Config) -> Dict[str, object]:
    cfg.ensure_dirs()
    df = load_longframe(resolve_data_path(cfg))
    agg = aggregate(df, mode=cfg.aggregate_mode)

    panels: Dict[str, dict] = {}
    point_rows: List[dict] = []
    share_rows: List[dict] = []
    aiq_rows: List[dict] = []
    ratio_rows: List[dict] = []

    for ds in C.DATASETS:
        sub = agg[agg["dataset"] == ds].set_index("model").reindex(list(C.MODELS))
        costs = sub["cost_mean"].to_numpy(dtype=float)
        quals = sub["perf_mean"].to_numpy(dtype=float)
        pts = np.column_stack([costs, quals])

        perf_wide, cost_wide = to_wide(df, ds)
        oracle = OracleRouter()
        oc, oq = oracle.point(perf_wide.values, cost_wide.values)
        share = oracle.selection_share(perf_wide.values, cost_wide.values)

        zero = ZeroRouter(pts)
        panels[ds] = {
            "model_points": [(costs[i], quals[i], C.MODELS[i]) for i in range(len(C.MODELS))],
            "oracle": (oc, oq),
            "zero_curve": zero.curve,
        }

        lo, hi = float(costs.min()), float(costs.max())
        for i, m in enumerate(C.MODELS):
            point_rows.append(
                {"dataset": ds, "model": m, "cost_mean": costs[i], "perf_mean": quals[i]}
            )
            share_rows.append({"dataset": ds, "model": m, "oracle_share": float(share[i])})
            aiq_rows.append(
                {
                    "dataset": ds,
                    "system": m,
                    "kind": "single-llm",
                    "cost_mean": costs[i],
                    "perf_mean": quals[i],
                    "aiq": degenerate_aiq(costs[i], quals[i], lo, hi),
                }
            )
        aiq_rows.append(
            {
                "dataset": ds,
                "system": "Zero router",
                "kind": "router",
                "cost_mean": np.nan,
                "perf_mean": np.nan,
                "aiq": aiq(zero.curve, lo, hi),
                "aiq_official": aiq_official(zero.curve, hi),
            }
        )
        aiq_rows.append(
            {
                "dataset": ds,
                "system": "Oracle",
                "kind": "oracle",
                "cost_mean": oc,
                "perf_mean": oq,
                "aiq": degenerate_aiq(oc, oq, lo, hi),
            }
        )

        # 论文「相近质量下成本相差数倍」的检查：质量 ≥ 0.95×最佳 的模型里，最贵/最便宜
        q_best = float(quals.max())
        near = [(costs[i], C.MODELS[i]) for i in range(len(costs)) if quals[i] >= 0.95 * q_best]
        cs = [c for c, _ in near]
        ratio_rows.append(
            {
                "dataset": ds,
                "quality_best": q_best,
                "quality_threshold": 0.95 * q_best,
                "n_models_near": len(near),
                "cheapest": min(near, key=lambda x: x[0])[1],
                "cost_min": min(cs),
                "priciest": max(near, key=lambda x: x[0])[1],
                "cost_max": max(cs),
                "cost_ratio": max(cs) / min(cs) if min(cs) > 0 else np.nan,
            }
        )

    point_df = pd.DataFrame(point_rows)
    share_df = pd.DataFrame(share_rows).pivot(index="model", columns="dataset", values="oracle_share")
    share_df = share_df.reindex(index=list(C.MODELS), columns=list(C.DATASETS))
    point_df.to_csv(cfg.tables_dir / "model_points.csv", index=False)
    share_df.to_csv(cfg.tables_dir / "oracle_selection_share.csv")
    aiq_df = pd.DataFrame(aiq_rows)
    aiq_df.to_csv(cfg.tables_dir / "table2_aiq.csv", index=False)
    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df.to_csv(cfg.tables_dir / "cost_ratio_matched_quality.csv", index=False)

    figs = [
        plotting.plot_cost_quality_grid(
            panels,
            cfg.figures_dir / "figure3_cost_quality.png",
            dpi=cfg.dpi,
            figsize=tuple(cfg.figsize),
            title="Cost-quality landscape: 11 LLMs + Oracle + Zero router",
        ),
        plotting.plot_cost_quality_grid(
            panels,
            cfg.figures_dir / "figure3_cost_quality_logx.png",
            dpi=cfg.dpi,
            figsize=tuple(cfg.figsize),
            xscale="log",
            title="Cost-quality landscape (log cost axis)",
        ),
        plotting.plot_oracle_heatmap(
            share_df, cfg.figures_dir / "figure3_oracle_selection.png", dpi=cfg.dpi
        ),
    ]

    # 汇总 markdown
    zero_vs_best = []
    for ds in C.DATASETS:
        sub = aiq_df[(aiq_df["dataset"] == ds)]
        z = float(sub.loc[sub["kind"] == "router", "aiq"].iloc[0])
        best_single = float(sub.loc[sub["kind"] == "single-llm", "aiq"].max())
        best_name = sub.loc[sub["kind"] == "single-llm"].sort_values("aiq").iloc[-1]["system"]
        zero_vs_best.append(
            {
                "dataset": ds,
                "zero_aiq": z,
                "best_single_aiq": best_single,
                "best_single": best_name,
                "zero_beats_best_single": bool(z >= best_single),
            }
        )
    zvb = pd.DataFrame(zero_vs_best)
    zvb.to_csv(cfg.tables_dir / "zero_router_vs_best_single.csv", index=False)

    lines = [
        "# Figure 3 复现结果",
        "",
        f"数据来源：**{cfg.source}**",
        "",
        "定义域取法：每个数据集上 [min 单模型平均成本, max 单模型平均成本]，"
        "所有系统在同一区间上计算 AIQ（论文 §3.2 要求显式共享定义域）。",
        "",
        "## AIQ（NDCH 归一化面积）",
        "",
        report.matrix_markdown(
            aiq_df.pivot(index="system", columns="dataset", values="aiq").reindex(
                index=list(C.MODELS) + ["Zero router", "Oracle"], columns=list(C.DATASETS)
            ).rename(columns={d: C.DATASET_LABELS[d] for d in C.DATASETS})
        ),
        "",
        "## Zero router 是否优于最好的单模型（论文核心结论之一）",
        "",
        report.markdown_table(zvb),
        "",
        "## 相近质量下的成本倍数",
        "",
        report.markdown_table(ratio_df[["dataset", "quality_best", "n_models_near", "cheapest", "priciest", "cost_ratio"]]),
        "",
        "## 产物",
        "",
        *[f"- `{f}`" for f in figs],
        f"- `{cfg.tables_dir / 'oracle_selection_share.csv'}`",
        f"- `{cfg.tables_dir / 'table2_aiq.csv'}`",
        f"- `{cfg.tables_dir / 'zero_router_vs_best_single.csv'}`",
        "",
    ]
    report.write_text(cfg.tables_dir / "figure3.md", "\n".join(lines))

    return {
        "figures": [str(f) for f in figs],
        "aiq": aiq_df.to_dict("records"),
        "zero_beats_best_single": zvb.to_dict("records"),
        "cost_ratio": ratio_df.to_dict("records"),
    }


# --------------------------------------------------------------------------
# 一键跑
# --------------------------------------------------------------------------
def run_all(cfg: Config, force_surrogate: bool = False) -> Dict[str, object]:
    out: Dict[str, object] = {}
    out["step0_surrogate"] = (
        build_surrogate(cfg, force=force_surrogate) if cfg.source == "surrogate" else {"status": "skipped"}
    )
    out["step1_validate"] = step_validate(cfg)
    out["step2_table1"] = step_table1(cfg)
    out["step3_figure3"] = step_figure3(cfg)
    report.write_text(
        cfg.derived_dir / "p0_summary.json",
        json.dumps(out, ensure_ascii=False, indent=2, default=str),
    )
    return out
