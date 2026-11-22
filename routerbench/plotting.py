"""出图：Figure 3 的左半（cost-quality 散点 + Oracle + Zero 前沿）与右半（Oracle 选择分布）。

风格约定：白底、去掉上/右边框、细网格、模型配色全局固定，
图内文字一律英文（避免字体缺字导致的方框）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # 无显示环境

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import constants as C  # noqa: E402

#: 11 个模型的固定配色
MODEL_COLORS: Dict[str, tuple] = {
    m: plt.cm.tab20(i / 11.0) for i, m in enumerate(C.MODELS)
}
ACCENT = "#111111"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def plot_cost_quality_grid(
    panels: Dict[str, dict],
    out_path: str | Path,
    dpi: int = 300,
    figsize: Tuple[float, float] = (18.0, 7.6),
    xscale: str = "linear",
    title: str | None = None,
) -> Path:
    """2×4 网格，每个数据集一格。

    panels[dataset] = {
        "model_points": [(cost, quality, model_name), ...],
        "oracle": (cost, quality),
        "zero_curve": (k, 2) ndarray,
    }
    """
    _style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=figsize, squeeze=False)
    order = [d for d in C.DATASETS if d in panels]
    for k, ds in enumerate(order):
        ax = axes[k // 4][k % 4]
        panel = panels[ds]
        pts = panel["model_points"]
        for cost, quality, name in pts:
            ax.scatter(
                cost,
                quality,
                s=34,
                color=MODEL_COLORS[name],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
        curve = np.asarray(panel["zero_curve"], dtype=float)
        ax.plot(
            curve[:, 0],
            curve[:, 1],
            color="#7f7f7f",
            lw=1.1,
            ls="--",
            zorder=2,
            label="Zero (NDCH)",
        )
        oc, oq = panel["oracle"]
        ax.scatter(
            oc,
            oq,
            marker="*",
            s=150,
            color=ACCENT,
            zorder=4,
            label="Oracle",
        )
        if xscale == "log":
            ax.set_xscale("log")
        ax.set_title(C.DATASET_LABELS[ds])
        ax.set_xlabel("cost per sample (USD)")
        ax.set_ylabel("quality")
        ax.margins(x=0.08, y=0.08)
        # 每样本成本常在 1e-5 量级，十进制刻度会挤成一团，改用科学计数法共享指数
        x_max = max(c for c, _, _ in pts)
        if xscale != "log" and x_max < 1e-2:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)

    # 最后一格放图例
    ax = axes[1][3]
    ax.axis("off")
    handles = [
        plt.Line2D(
            [], [], marker="o", ls="", markersize=6, color=MODEL_COLORS[m], label=m
        )
        for m in C.MODELS
    ]
    handles += [
        plt.Line2D([], [], marker="*", ls="", markersize=11, color=ACCENT, label="Oracle"),
        plt.Line2D([], [], color="#7f7f7f", ls="--", lw=1.1, label="Zero router (NDCH)"),
    ]
    ax.legend(handles=handles, loc="center", fontsize=9, ncol=1, handletextpad=0.4)

    if title:
        fig.suptitle(title, fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96 if title else 1.0))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_oracle_heatmap(
    share: pd.DataFrame,
    out_path: str | Path,
    dpi: int = 300,
    figsize: Tuple[float, float] = (10.0, 6.0),
) -> Path:
    """Oracle 在各数据集上的选择占比（行=模型，列=数据集）。"""
    _style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mat = share.reindex(index=list(C.MODELS), columns=[d for d in C.DATASETS if d in share.columns])
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat.values, aspect="auto", cmap="Blues", vmin=0.0, vmax=float(np.nanmax(mat.values)))
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([C.DATASET_LABELS[c] for c in mat.columns], rotation=25, ha="right")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v > 0.55 * np.nanmax(mat.values) else "#333333",
            )
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("share of samples where model is the Oracle choice")
    ax.set_title("Oracle routing distribution (Figure 3, right)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
