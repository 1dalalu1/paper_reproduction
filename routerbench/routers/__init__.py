"""路由器：把 (perf, cost) 宽表映射成选择结果与 cost-quality 曲线。"""

from .base import Router, PointRouter
from .oracle import OracleRouter
from .zero import ZeroRouter

__all__ = ["Router", "PointRouter", "OracleRouter", "ZeroRouter"]
