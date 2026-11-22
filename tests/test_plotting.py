"""绘图冒烟测试：只要图能落盘且非空即可（防止 matplotlib API 变动悄悄炸掉）。

注：输出放在工程内的 `_tmp/`，不使用系统临时目录（本机沙箱禁止写系统 Temp）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from routerbench import constants as C
from routerbench import plotting

TMP = Path(__file__).resolve().parents[1] / "_tmp"


def _panels():
    panels = {}
    rng = np.random.default_rng(0)
    for ds in C.DATASETS:
        costs = rng.uniform(0.01, 1.0, 3)
        quals = rng.uniform(0.3, 0.9, 3)
        curve = np.column_stack([costs, quals])
        panels[ds] = {
            "model_points": [(costs[i], quals[i], C.MODELS[i]) for i in range(3)],
            "oracle": (float(costs.min()), 0.99),
            "zero_curve": curve,
        }
    return panels


def _out(name: str) -> Path:
    TMP.mkdir(parents=True, exist_ok=True)
    return TMP / name


def test_plot_cost_quality_grid_writes_file():
    out = plotting.plot_cost_quality_grid(_panels(), _out("smoke_fig3.png"), dpi=80)
    assert out.exists() and out.stat().st_size > 1000


def test_plot_cost_quality_grid_logx():
    out = plotting.plot_cost_quality_grid(
        _panels(), _out("smoke_fig3_logx.png"), dpi=80, xscale="log"
    )
    assert out.exists() and out.stat().st_size > 1000


def test_plot_oracle_heatmap():
    share = pd.DataFrame(
        np.full((len(C.MODELS), len(C.DATASETS)), 1.0 / len(C.MODELS)),
        index=list(C.MODELS),
        columns=list(C.DATASETS),
    )
    out = plotting.plot_oracle_heatmap(share, _out("smoke_heatmap.png"), dpi=80)
    assert out.exists() and out.stat().st_size > 1000
