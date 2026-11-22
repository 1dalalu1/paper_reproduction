"""论文常量：模型 / 数据集清单、Table 1 原始数值、评测集规模。

唯一真值来源：RouterBench: A Benchmark for Multi-LLM Routing System
(Hu et al., 2024) 正文 §3–§4 与 Table 1（p.15）。

Table 1 每格写作 (Perf, Cost)：
  * Perf 为该模型在该数据集上的平均质量，取值 [0, 1]；
  * Cost 为该模型在该数据集上**全部样本的成本之和**（美元, sum），不是均值。
    画图与 AIQ 使用每样本均值 cost_sum / n_d，二者仅差常数，AIQ 是尺度不变的。
"""

from __future__ import annotations

from typing import Dict, Tuple

# --------------------------------------------------------------------------
# 清单
# --------------------------------------------------------------------------
MODELS: Tuple[str, ...] = (
    "WizardLM 13B",
    "Mistral 7B",
    "Mixtral 8x7B",
    "Code Llama 34B",
    "Yi 34B",
    "GPT-3.5",
    "Claude Instant V1",
    "Llama 70B",
    "Claude V1",
    "Claude V2",
    "GPT-4",
)

#: P0 阶段使用的 7 个数据集（RAG 子集不公开，留到后续阶段单独处理）
DATASETS: Tuple[str, ...] = (
    "mmlu",
    "mt_bench",
    "mbpp",
    "hellaswag",
    "winogrande",
    "gsm8k",
    "arc",
)

DATASET_LABELS: Dict[str, str] = {
    "mmlu": "MMLU",
    "mt_bench": "MT-Bench",
    "mbpp": "MBPP",
    "hellaswag": "HellaSwag",
    "winogrande": "Winogrande",
    "gsm8k": "GSM8K",
    "arc": "ARC-Challenge",
}

#: 5-shot + exact match 的 0/1 数据集
EXACT_MATCH_DATASETS: Tuple[str, ...] = (
    "mmlu",
    "hellaswag",
    "winogrande",
    "arc",
    "gsm8k",
    "mbpp",
)

#: GPT-4 裁判打分、归一化到 [0, 1] 的数据集
JUDGE_DATASETS: Tuple[str, ...] = ("mt_bench",)

# --------------------------------------------------------------------------
# Table 1（p.15），列顺序与 DATASETS 一致
# --------------------------------------------------------------------------
TABLE1: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "WizardLM 13B": (
        (0.568, 0.122),
        (0.796, 0.006),
        (0.364, 0.011),
        (0.636, 0.727),
        (0.512, 0.040),
        (0.510, 0.354),
        (0.660, 0.068),
    ),
    "Mistral 7B": (
        (0.562, 0.081),
        (0.779, 0.003),
        (0.349, 0.006),
        (0.541, 0.485),
        (0.562, 0.027),
        (0.409, 0.210),
        (0.642, 0.046),
    ),
    "Mixtral 8x7B": (
        (0.733, 0.245),
        (0.921, 0.012),
        (0.573, 0.023),
        (0.707, 1.455),
        (0.677, 0.081),
        (0.515, 0.594),
        (0.844, 0.137),
    ),
    "Code Llama 34B": (
        (0.569, 0.317),
        (0.796, 0.015),
        (0.465, 0.021),
        (0.525, 1.882),
        (0.617, 0.104),
        (0.462, 0.752),
        (0.644, 0.177),
    ),
    "Yi 34B": (
        (0.743, 0.326),
        (0.938, 0.018),
        (0.333, 0.031),
        (0.931, 1.938),
        (0.748, 0.107),
        (0.552, 0.867),
        (0.882, 0.182),
    ),
    "GPT-3.5": (
        (0.720, 0.408),
        (0.908, 0.026),
        (0.651, 0.044),
        (0.816, 2.426),
        (0.630, 0.134),
        (0.601, 1.170),
        (0.855, 0.228),
    ),
    "Claude Instant V1": (
        (0.384, 0.327),
        (0.863, 0.030),
        (0.550, 0.064),
        (0.801, 1.943),
        (0.512, 0.108),
        (0.626, 1.300),
        (0.821, 0.183),
    ),
    "Llama 70B": (
        (0.647, 0.367),
        (0.854, 0.022),
        (0.302, 0.039),
        (0.736, 2.183),
        (0.504, 0.121),
        (0.529, 0.870),
        (0.794, 0.205),
    ),
    "Claude V1": (
        (0.475, 3.269),
        (0.938, 0.361),
        (0.527, 0.607),
        (0.841, 19.43),
        (0.570, 1.077),
        (0.653, 11.09),
        (0.889, 1.829),
    ),
    "Claude V2": (
        (0.619, 3.270),
        (0.854, 0.277),
        (0.605, 0.770),
        (0.421, 19.50),
        (0.446, 1.081),
        (0.664, 13.49),
        (0.546, 1.833),
    ),
    "GPT-4": (
        (0.828, 4.086),
        (0.971, 0.721),
        (0.682, 1.235),
        (0.923, 24.29),
        (0.858, 1.346),
        (0.654, 19.08),
        (0.921, 2.286),
    ),
}

#: Table 1 最后一行 Oracle（每样本在"答对的模型"里选最便宜的）
ORACLE_ROW: Tuple[Tuple[float, float], ...] = (
    (0.957, 0.297),
    (0.996, 0.052),
    (0.899, 0.041),
    (0.994, 0.860),
    (1.000, 0.042),
    (0.748, 1.282),
    (0.977, 0.091),
)

#: 论文报告的样本总数（11 个通用模型 × 8 数据集 / 64 子任务）
PAPER_N_ROWS: int = 405_467

#: 各数据集的标准评测集规模（论文口径；surrogate 与日志校验都用它）
SAMPLE_SIZES: Dict[str, int] = {
    "mmlu": 14_042,
    "hellaswag": 10_042,
    "arc": 1_172,
    "winogrande": 1_267,
    "gsm8k": 1_319,
    "mbpp": 500,
    "mt_bench": 80,
}

#: 划分比例（论文 §4：每个任务 70/30）
TEST_SIZE: float = 0.30


def table1_perf() -> Dict[Tuple[str, str], float]:
    """(model, dataset) -> 论文报告的 Perf。"""
    return {
        (m, d): TABLE1[m][i][0] for m in MODELS for i, d in enumerate(DATASETS)
    }


def table1_cost() -> Dict[Tuple[str, str], float]:
    """(model, dataset) -> 论文报告的 Cost（数据集总成本, 美元）。"""
    return {
        (m, d): TABLE1[m][i][1] for m in MODELS for i, d in enumerate(DATASETS)
    }


def oracle_perf() -> Dict[str, float]:
    return {d: ORACLE_ROW[i][0] for i, d in enumerate(DATASETS)}


def oracle_cost() -> Dict[str, float]:
    return {d: ORACLE_ROW[i][1] for i, d in enumerate(DATASETS)}
