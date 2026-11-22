"""metrics.py 的单元测试：凸包、NDCH、AIQ、定义域、成本反查。"""

from __future__ import annotations

import numpy as np

from routerbench.metrics import (
    aiq,
    aiq_official,
    as_points,
    cost_at_quality,
    cost_ratio_at_quality,
    degenerate_aiq,
    enforce_monotone,
    integrate,
    ndch,
    shared_domain,
    sort_and_collapse,
    upper_hull,
    value_at,
)


def test_as_points_accepts_flat_pair():
    assert as_points([1.0, 0.5]).shape == (1, 2)


def test_sort_and_collapse_keeps_best_on_tie():
    out = sort_and_collapse([[1.0, 0.2], [1.0, 0.5], [2.0, 0.4]])
    assert out.shape == (2, 2)
    assert out[0].tolist() == [1.0, 0.5]


def test_sort_is_ascending():
    out = sort_and_collapse([[3, 0.3], [1, 0.1], [2, 0.2]])
    assert out[:, 0].tolist() == [1, 2, 3]


def test_upper_hull_drops_collinear_middle():
    assert len(upper_hull([[0, 0], [1, 0.5], [2, 1.0]])) == 2


def test_upper_hull_keeps_concave_bulge():
    assert len(upper_hull([[0, 0], [1, 1], [2, 0]])) == 3


def test_upper_hull_drops_dominated_inner_point():
    # (1, 0.2) 在 (0,0)-(2,0.8) 线下方
    hull = upper_hull([[0, 0], [1, 0.2], [2, 0.8]])
    assert hull.shape[0] == 2


def test_enforce_monotone_cumulative_max():
    out = enforce_monotone([[1, 0.9], [2, 0.5], [3, 0.7]])
    assert out[:, 1].tolist() == [0.9, 0.9, 0.9]


def test_ndch_replaces_lower_quality_with_previous():
    out = ndch([[1, 0.9], [2, 0.5]])
    assert out[-1].tolist() == [2.0, 0.9]


def test_ndch_is_non_decreasing_in_quality():
    rng = np.random.default_rng(0)
    pts = np.column_stack([np.sort(rng.random(30)), rng.random(30)])
    out = ndch(pts)
    assert np.all(np.diff(out[:, 1]) >= -1e-12)


def test_value_at_interpolates_inside_and_flat_outside():
    curve = [[0.0, 0.0], [2.0, 1.0]]
    assert abs(value_at(curve, 1.0) - 0.5) < 1e-12
    assert abs(value_at(curve, -1.0) - 0.0) < 1e-12
    assert abs(value_at(curve, 5.0) - 1.0) < 1e-12


def test_integrate_exact_for_piecewise_linear():
    assert abs(integrate([[0, 0], [1, 1], [2, 1]], 0, 2) - 1.5) < 1e-12


def test_aiq_linear_is_half():
    assert abs(aiq([[0, 0], [1, 1]], 0, 1) - 0.5) < 1e-12


def test_aiq_constant_curve():
    assert abs(aiq([[0, 0.3], [1, 0.3]]) - 0.3) < 1e-12


def test_aiq_uses_own_domain_by_default():
    assert abs(aiq([[0.5, 0.2], [1.5, 0.8]]) - 0.5) < 1e-12


def test_aiq_flat_extrapolation_beyond_curve():
    assert abs(aiq([[0, 0.5], [1, 0.5]], 0, 2) - 0.5) < 1e-12


def test_shared_domain_intersection():
    lo, hi = shared_domain([[[0, 0], [2, 1]], [[1, 0], [3, 1]]])
    assert abs(lo - 1.0) < 1e-12 and abs(hi - 2.0) < 1e-12


def test_shared_domain_falls_back_to_union_when_disjoint():
    lo, hi = shared_domain([[[0, 0], [1, 1]], [[5, 0], [6, 1]]])
    assert abs(lo - 0.0) < 1e-12 and abs(hi - 6.0) < 1e-12


def test_cost_at_quality_interpolates():
    assert abs(cost_at_quality([[0, 0], [2, 1]], 0.5) - 1.0) < 1e-12


def test_cost_at_quality_returns_none_above_max():
    assert cost_at_quality([[0, 0], [2, 1]], 1.5) is None


def test_cost_ratio_at_quality():
    ratio = cost_ratio_at_quality([[0, 0], [1, 1]], [[0, 0], [3, 1]], 0.5)
    assert ratio is not None and abs(ratio[0] - 3.0) < 1e-12


def test_degenerate_aiq_matches_null_router_line():
    # (0,0)->(1,0.8) 的连线在 [0.5,1] 上的均值 = (0.4+0.8)/2 = 0.6
    assert abs(degenerate_aiq(1.0, 0.8, 0.5, 1.0) - 0.6) < 1e-9


def test_aiq_never_exceeds_one():
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.random(50), rng.random(50)])
    assert 0.0 <= aiq(pts) <= 1.0

def test_aiq_official_uses_own_domain():
    # 曲线从 x=1 开始、到 x=2 结束，官方口径的分母是 max_x=2，[0,1] 段没有面积
    # 面积 = 梯形((1,0.5),(2,1.0)) = 0.75 → AIQ_official = 0.75/2 = 0.375
    assert abs(aiq_official([[1.0, 0.5], [2.0, 1.0]], 2.0) - 0.375) < 1e-12


def test_aiq_official_not_greater_than_shared_domain_aiq():
    pts = np.array([[0.5, 0.2], [1.0, 0.6], [3.0, 0.9]])
    assert aiq_official(pts, 3.0) <= aiq(pts, 0.5, 3.0) + 1e-12
