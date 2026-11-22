"""RouterBench 复现工程的核心库。

模块划分
--------
constants : 论文常量（模型清单、数据集清单、Table 1 原始数值、评测集规模）
config    : 路径与运行参数
data      : 长格式数据的读写、校验、聚合、按样本切分
surrogate : 合成替代数据生成器（无网络环境下打通并自测全流程）
metrics   : cost-quality 曲线代数（上凸包 / NDCH / AIQ / 共享定义域）
routers   : 路由器实现（Oracle / Zero / 单模型点）
plotting  : 出图
report    : 表 1 对比、AIQ 汇总、Markdown 序列化
"""

__version__ = "0.1.0"

from . import constants  # noqa: F401
