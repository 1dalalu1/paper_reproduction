"""Zero router（R_zero）：不做任何预测，直接对已有 LLM 做概率混合。

论文的做法：把 11 个 LLM 作为 11 个点，其**期望混合**可达集就是这些点的上凸包，
因此 Zero router 的曲线 = NDCH(11 个点)。它等价于「随机地按某个比例把请求
分给不同模型」，但不需要任何额外的模型/训练成本。

它是论文最重要的基线：KNN / MLP 在多数数据集上打不过它。
"""

from __future__ import annotations

import numpy as np

from ..metrics import ndch, value_at


class ZeroRouter:
    """只看点集、不看样本的路由器，用曲线（而非逐样本选择）描述。"""

    name = "zero"

    def __init__(self, points: np.ndarray, name: str | None = None):
        self.points = np.asarray(points, dtype=float).reshape(-1, 2)
        if name:
            self.name = name
        self._curve = ndch(self.points)

    @property
    def curve(self) -> np.ndarray:
        return self._curve

    def quality_at(self, cost: float) -> float:
        return value_at(self._curve, cost)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<ZeroRouter {self.name} vertices={len(self._curve)}>"
