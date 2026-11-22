"""合成替代数据（surrogate）生成器。

为什么需要它
------------
P0 基线需要 40 万行预生成推理结果。本机当前**无法访问 HuggingFace / PyPI**
（HTTPS 请求全部超时或被重置），无法下载 withmartian/routerbench。
为了让聚合、度量、图表链路可以被真正执行和自测，这里生成一份
「结构等价、边际分布与 Oracle 对齐论文 Table 1」的合成数据。

它不是复现结果，只是代码链路的**测试夹具**：
  * 列名、行数规模、每个 (模型, 数据集) 的质量均值与成本总和按论文 Table 1 标定
    （12 个矩：11 个模型边际 + 该数据集的 Oracle 质量）；
  * 样本之间通过共享的潜在难度 d 相关，使「路由」这件事本身有意义；
  * 成本服从对数正态，按 (模型, 数据集) 重缩放，使总成本与 Table 1 完全一致。

一旦拿到真实数据（data/raw/ 下的 parquet/pkl），把 config 的 data.source
改成 "real" 即可，下游代码零改动。

生成模型（二值项目反应模型）
----------------------------
对数据集 d 的样本 i、模型 j：
    u_ij       = a_j + d_i + sigma_d * e_ij     # 潜在效用
    d_i        ~ N(0, 1)                        # 共享难度（越大越简单）
    e_ij       ~ N(0, 1)                        # 模型自身的偶然性
    perf_ij    = Phi(u_ij) 再逐列秩标定          # exact-match 数据集取 1[u>阈值]
于是  p_j(d) = Phi((a_j + d) / sigma_d)，且
    E_d[p_j]      = Phi(a_j / sqrt(1 + sigma_d^2)) = 目标质量
=>  a_j = sqrt(1 + sigma_d^2) * Phi^{-1}(target_j)          （闭式解）
    E[max_j perf] = E_d[1 - Π_j (1 - p_j(d))]                （对 sigma_d 单调）
=> sigma_d 由外层二分求解，使 Oracle 质量对齐论文。

sigma_d 的直觉：它衡量「模型各自犯错的独立性」。
    sigma_d → 0   ：所有模型同对同错，Oracle ≈ 最强单模型；
    sigma_d → ∞   ：各模型独立犯错，Oracle ≈ 1 - Π(1 - p_j)。
论文的 Oracle 值就落在两者之间，因此总能解出唯一 sigma_d。
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

from . import constants as C

_EPS = 1e-9


def _clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(p, _EPS, 1.0 - _EPS)


def _oracle_quality(sigma: float, targets: np.ndarray, grid: np.ndarray, w: np.ndarray) -> float:
    """E_d[1 - Π_j (1 - p_j(d))]，即独立抽样下的期望 Oracle 质量。"""
    a = np.sqrt(1.0 + sigma ** 2) * ndtri(_clip_prob(targets))
    p = ndtr((a[:, None] + grid[None, :]) / sigma)  # (n_models, n_grid)
    return float(np.sum(w * (1.0 - np.prod(1.0 - p, axis=0))))


def _solve_sigma(
    targets: np.ndarray,
    oracle_target: float,
    grid: np.ndarray,
    w: np.ndarray,
) -> float:
    """外层二分：找数据集级独立性 sigma_d，使 Oracle 质量对齐论文。"""
    lo, hi = 0.02, 40.0
    q_lo = _oracle_quality(lo, targets, grid, w)
    q_hi = _oracle_quality(hi, targets, grid, w)
    if oracle_target <= q_lo:
        return lo
    if oracle_target >= q_hi:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _oracle_quality(mid, targets, grid, w) < oracle_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def generate(
    seed: int = 20240501,
    sample_sizes: Dict[str, int] | None = None,
    cost_cv: float = 0.6,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """生成 surrogate 长格式表，返回 (df, 标定报告)。"""
    rng = np.random.default_rng(seed)
    sizes = dict(sample_sizes or C.SAMPLE_SIZES)

    # 难度积分网格（标准正态的分位数 + 权重）
    grid = np.linspace(-6.0, 6.0, 1601)
    w = np.exp(-0.5 * grid ** 2)
    w = w / w.sum()

    frames: list[pd.DataFrame] = []
    calib: Dict[str, object] = {}

    for ds in C.DATASETS:
        idx = C.DATASETS.index(ds)
        n = int(sizes[ds])
        targets = np.array([C.TABLE1[m][idx][0] for m in C.MODELS])
        total_cost = np.array([C.TABLE1[m][idx][1] for m in C.MODELS])
        oracle_target = C.ORACLE_ROW[idx][0]

        sigma = _solve_sigma(targets, oracle_target, grid, w)
        a = np.sqrt(1.0 + sigma ** 2) * ndtri(_clip_prob(targets))

        d = rng.normal(size=n)
        e = rng.normal(size=(n, len(C.MODELS)))
        u = a[None, :] + d[:, None] + sigma * e
        p = ndtr(u)

        if ds in C.EXACT_MATCH_DATASETS:
            # 秩标定：按列取分位阈值，使「做对数」精确等于 round(target * n)。
            # u 已含共享难度与模型随机项，阈值只是逐列的单调平移，不改变相关性结构。
            perf = np.zeros_like(u)
            for j in range(u.shape[1]):
                k = int(round(float(targets[j]) * n))
                srt = np.sort(u[:, j])[::-1]
                if k <= 0:
                    thr = srt[0] + 1.0
                elif k >= n:
                    thr = srt[-1] - 1.0
                else:
                    thr = 0.5 * (srt[k - 1] + srt[k])
                perf[:, j] = (u[:, j] > thr).astype(float)
        else:
            # 裁判集：保留全部相对差异，缩放到 [0,1] 的可用余量内，
            # 均值精确等于 target；不截断，避免高分区系统性偏低。
            dev = p - p.mean(axis=0, keepdims=True)
            spread = np.max(np.abs(dev), axis=0, keepdims=True)
            headroom = np.minimum(targets[None, :], 1.0 - targets[None, :])
            scale = np.where(
                spread > 1e-12, np.minimum(1.0, headroom / np.maximum(spread, 1e-12)), 1.0
            )
            perf = np.clip(targets[None, :] + dev * scale, 0.0, 1.0)

        sigma2 = math.log1p(cost_cv ** 2)
        costs = rng.lognormal(
            mean=np.log(total_cost / n) - sigma2 / 2.0,
            sigma=math.sqrt(sigma2),
            size=(n, len(C.MODELS)),
        )
        costs = costs * (total_cost / costs.sum(axis=0, keepdims=True))  # 精确对齐总成本

        sample_ids = np.array([f"{ds}-{i:06d}" for i in range(n)])
        frames.append(
            pd.DataFrame(
                {
                    "sample_id": np.repeat(sample_ids, len(C.MODELS)),
                    "dataset": ds,
                    "model": np.tile(np.array(C.MODELS), n),
                    "performance": perf.reshape(-1),
                    "cost": costs.reshape(-1),
                }
            )
        )
        calib[ds] = {
            "n_samples": n,
            "sigma_idiosyncratic": round(float(sigma), 4),
            "oracle_target": float(oracle_target),
            "oracle_realised": round(float(perf.max(axis=1).mean()), 5),
            "perf_calibration": (
                "rank_exact" if ds in C.EXACT_MATCH_DATASETS else "mean_exact"
            ),
            "perf_diff_max": round(
                float(np.max(np.abs(perf.mean(axis=0) - targets))), 5
            ),
            "cost_sum_exact": bool(
                np.allclose(costs.sum(axis=0), total_cost, rtol=1e-12)
            ),
        }

    df = pd.concat(frames, ignore_index=True)
    return df, calib
