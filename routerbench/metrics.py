"""cost-quality 平面上的曲线代数：上凸包、NDCH、AIQ、共享定义域。

对应论文 §3.2 的三类操作：

1. 线性插值 —— 两个路由器按概率 t 混合，期望点 `t*p1 + (1-t)*p2` 落在连线上，
   因此「可达集合」就是点集的上凸包；
2. 外推 —— 允许把曲线水平向右延长（多花钱但质量不变），或与 null router (0, 0)
   连线以统一定义域；
3. NDCH（Non-Dominated Cost-Quality Hull）—— 先取上凸包，再强制质量随成本
   单调不减：若 `c2 > c1` 但 `q2 < q1`，则把 `(c2, q2)` 换成 `(c2, q1)`。

主指标：`AIQ(R) = 1/(c_max - c_min) * ∫_{c_min}^{c_max} f_R(c) dc`
其中 f_R 为 NDCH 的分段线性插值，定义域 [c_min, c_max] 必须在所有被比较的
系统之间共享，否则数值不可比。

所有函数只用 numpy，输入为 (n, 2) 的 [cost, quality] 点集，输出按 cost 升序。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

PointsLike = Sequence[Sequence[float]]


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def as_points(points: PointsLike) -> np.ndarray:
    """把任意可迭代的 (cost, quality) 转成 float64 的 (n, 2) 数组。"""
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"points 必须是 (n, 2) 形状，收到 {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("points 不能为空")
    if not np.all(np.isfinite(arr)):
        raise ValueError("points 含 NaN/Inf")
    return arr


def sort_and_collapse(points: PointsLike) -> np.ndarray:
    """按 cost 升序；同 cost 只保留最大 quality（同价取最优）。"""
    arr = as_points(points)
    order = np.lexsort((-arr[:, 1], arr[:, 0]))
    arr = arr[order]
    out: list[list[float]] = []
    for c, q in arr:
        if out and np.isclose(c, out[-1][0], rtol=0.0, atol=1e-12):
            out[-1][1] = max(out[-1][1], float(q))
        else:
            out.append([float(c), float(q)])
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------
# 凸包与 NDCH
# --------------------------------------------------------------------------
def upper_hull(points: PointsLike) -> np.ndarray:
    """点集的上凸包顶点（cost 升序，斜率严格递减）。

    做法是 Andrew monotone chain 的「只保留右转」版本：遍历时若
    `slope(a, b) <= slope(b, p)`，说明 b 落在 a→p 的弦上或下方，弹出 b。
    """
    pts = sort_and_collapse(points)
    hull: list[np.ndarray] = []
    for p in pts:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # 斜率比较，交叉相乘避免除零（cost 严格递增 => 分母为正）
            if (b[1] - a[1]) * (p[0] - b[0]) <= (p[1] - b[1]) * (b[0] - a[0]):
                hull.pop()
            else:
                break
        hull.append(p)
    return np.asarray(hull, dtype=float)


def enforce_monotone(points: PointsLike) -> np.ndarray:
    """强制 quality 随 cost 单调不减（论文 NDCH 的第二步）。"""
    arr = as_points(points)
    out = arr.copy()
    out[:, 1] = np.maximum.accumulate(out[:, 1])
    return out


def ndch(points: PointsLike) -> np.ndarray:
    """上凸包 + 单调修复 = NDCH。"""
    return enforce_monotone(upper_hull(points))


# --------------------------------------------------------------------------
# 曲线取值与积分
# --------------------------------------------------------------------------
def value_at(curve: PointsLike, cost: float) -> float:
    """曲线在 cost 处的质量；区间内线性插值，区间外水平外推（外推操作）。"""
    pts = sort_and_collapse(curve)
    return float(np.interp(cost, pts[:, 0], pts[:, 1]))


def integrate(curve: PointsLike, c_min: float, c_max: float) -> float:
    """∫ f(c) dc，f 在 [c_min, c_max] 上分段线性。

    只在 c_min / c_max 与曲线断点处取节点，梯形法在这里是**精确**的。
    """
    if c_max < c_min:
        raise ValueError(f"c_max({c_max}) 不能小于 c_min({c_min})")
    if c_max == c_min:
        return 0.0
    pts = sort_and_collapse(curve)
    inner = pts[(pts[:, 0] > c_min) & (pts[:, 0] < c_max), 0]
    xs = np.concatenate(([c_min], inner, [c_max]))
    ys = np.interp(xs, pts[:, 0], pts[:, 1])
    return float(np.sum(0.5 * (ys[1:] + ys[:-1]) * np.diff(xs)))


def aiq(
    curve: PointsLike,
    c_min: float | None = None,
    c_max: float | None = None,
) -> float:
    """AIQ：NDCH 在共享定义域 [c_min, c_max] 上的归一化面积。

    未给定定义域时退化为曲线自身的 [c_first, c_last]。
    """
    pts = ndch(curve)
    lo = float(pts[0, 0]) if c_min is None else float(c_min)
    hi = float(pts[-1, 0]) if c_max is None else float(c_max)
    if hi <= lo:
        # 定义域退化成一点：AIQ 就是该点的质量
        return float(pts[-1, 1])
    return integrate(pts, lo, hi) / (hi - lo)


def aiq_official(curve: PointsLike, c_max: float | None = None) -> float:
    """官方实现（`evaluation/AIQ.py::calc_AIQ`）口径的 AIQ，用于交叉校验。

    与 `aiq` 的两点差异：
    1. 每条曲线只在**自己的首点**之后才有面积，[0, 首点] 区间按 0 计；
    2. 分母是全局 `c_max`（曲线右端水平外推到 c_max），而不是 `c_max - c_min`。

    因此官方口径的 AIQ 一般略小于本仓库的共享定义域口径（少了 [0, c_min] 那段零面积，
    却把相同的面积除以更大的分母）。
    """
    pts = ndch(curve)
    hi = float(pts[-1, 0]) if c_max is None else float(c_max)
    if hi <= 0:
        return float("nan")
    if hi <= float(pts[-1, 0]):
        xs, ys = pts[:, 0], pts[:, 1]
    else:
        xs = np.append(pts[:, 0], hi)
        ys = np.append(pts[:, 1], pts[-1, 1])
    return float(np.trapz(ys, xs)) / hi


def shared_domain(
    curves: Iterable[PointsLike], mode: str = "intersection"
) -> tuple[float, float]:
    """多个系统共同的定义域。

    intersection（默认）：取交集，最保守，保证每个系统在整个区间上都有定义；
    union：取并集，需配合水平外推使用。
    """
    spans = []
    for c in curves:
        pts = sort_and_collapse(c)
        spans.append((float(pts[0, 0]), float(pts[-1, 0])))
    if not spans:
        raise ValueError("curves 为空")
    if mode == "intersection":
        lo = max(s[0] for s in spans)
        hi = min(s[1] for s in spans)
        if hi <= lo:  # 交集为空时退化为并集，避免抛错中断整批实验
            mode = "union"
        else:
            return lo, hi
    if mode != "union":
        raise ValueError(f"未知的 domain mode: {mode}")
    return min(s[0] for s in spans), max(s[1] for s in spans)


def cost_at_quality(curve: PointsLike, quality: float) -> float | None:
    """达到给定质量所需的最小 cost；达不到（质量高于曲线峰值）返回 None。

    用于「相近质量下成本相差几倍」这类断言。
    """
    pts = sort_and_collapse(curve)
    if quality > pts[-1, 1] + 1e-12:
        return None
    if quality <= pts[0, 1]:
        return float(pts[0, 0])
    for (c0, q0), (c1, q1) in zip(pts[:-1], pts[1:]):
        if q1 >= quality >= q0 and q1 > q0:
            w = (quality - q0) / (q1 - q0)
            return float(c0 + w * (c1 - c0))
    return float(pts[-1, 0])


def cost_ratio_at_quality(curve_a: PointsLike, curve_b: PointsLike, quality: float):
    """在同一质量水平上比较两条曲线的成本，返回 (cost_b / cost_a, cost_a, cost_b)。"""
    ca = cost_at_quality(curve_a, quality)
    cb = cost_at_quality(curve_b, quality)
    if ca is None or cb is None or ca <= 0:
        return None
    return float(cb / ca), ca, cb


def augment_with_null(curve: PointsLike) -> np.ndarray:
    """与 null router (0, 0) 连线，用于把单点路由器放进统一坐标系（外推操作）。"""
    pts = sort_and_collapse(curve)
    if pts[0, 0] <= 0:
        return pts
    return np.vstack(([0.0, 0.0], pts))


def degenerate_aiq(cost: float, quality: float, c_min: float, c_max: float) -> float:
    """单点路由器（无法插值）在共享定义域上的 AIQ：经 null router 连线后积分。

    论文用它把「单个 LLM」和「真正的路由器」放到同一把尺子上比较。
    """
    return aiq(augment_with_null([[cost, quality]]), c_min, c_max)
