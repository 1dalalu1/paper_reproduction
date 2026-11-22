"""Oracle 路由器：每样本在「答对」的模型里选最便宜的。

论文 Table 1 的 Oracle 行就是这个策略；全错时退化为选最便宜的模型
（此时所有模型都是「最高质量」，argmin 成本自然取到最便宜的）。
"""

from __future__ import annotations

import numpy as np

from .base import Router


class OracleRouter(Router):
    name = "oracle"

    def select(self, perf: np.ndarray, cost: np.ndarray) -> np.ndarray:
        perf = np.asarray(perf, dtype=float)
        cost = np.asarray(cost, dtype=float)
        best = perf.max(axis=1, keepdims=True)
        masked = np.where(perf >= best - 1e-12, cost, np.inf)
        return np.argmin(masked, axis=1)

    def selection_share(self, perf: np.ndarray, cost: np.ndarray) -> np.ndarray:
        """每个模型被 Oracle 选中的比例，用于复现 Figure 3 右侧的分布图。"""
        idx = self.select(perf, cost)
        return np.bincount(idx, minlength=perf.shape[1]) / len(idx)
