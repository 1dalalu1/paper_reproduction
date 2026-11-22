"""路由器基类。

论文的视角：路由器 R 是 x -> LLM_i 的映射，它在 cost-quality 平面上对应一个点
    c_R(D) = E[c(R(x))],  q_R(D) = E[q(R(x))]
多个路由器按概率混合时，可达点集是它们坐标的凸包。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Router(ABC):
    name: str = "router"

    @abstractmethod
    def select(self, perf: np.ndarray, cost: np.ndarray) -> np.ndarray:
        """返回每个样本选中的模型下标，形状 (n_samples,)。"""

    def point(self, perf: np.ndarray, cost: np.ndarray) -> tuple[float, float]:
        """路由器在 (平均成本, 平均质量) 平面上的坐标。"""
        idx = self.select(perf, cost)
        rows = np.arange(len(idx))
        return float(cost[rows, idx].mean()), float(perf[rows, idx].mean())

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<{type(self).__name__} name={self.name!r}>"


class PointRouter(Router):
    """固定在某个模型上的「路由」（即单个 LLM 本身）。

    这是一个退化的路由器：曲线只有一个点，定义域退化成一点。
    """

    def __init__(self, model_index: int, name: str | None = None):
        self.model_index = int(model_index)
        self.name = name or f"model_{model_index}"

    def select(self, perf: np.ndarray, cost: np.ndarray) -> np.ndarray:
        return np.full(len(perf), self.model_index, dtype=int)
