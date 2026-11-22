"""routers 的单元测试：Oracle 的并列规则、Zero 的凸包性质、单模型点。"""

from __future__ import annotations

import numpy as np

from routerbench.metrics import ndch
from routerbench.routers import OracleRouter, PointRouter, ZeroRouter


def _case():
    perf = np.array([[1, 0, 1], [0, 0, 0], [1, 1, 0]], dtype=float)
    cost = np.array([[0.5, 0.1, 0.2], [0.3, 0.2, 0.1], [0.4, 0.05, 0.3]], dtype=float)
    return perf, cost


def test_oracle_picks_cheapest_among_correct():
    perf, cost = _case()
    assert OracleRouter().select(perf, cost).tolist() == [2, 2, 1]


def test_oracle_all_wrong_picks_cheapest():
    perf = np.zeros((2, 3))
    cost = np.array([[0.5, 0.1, 0.2], [0.3, 0.2, 0.1]])
    assert OracleRouter().select(perf, cost).tolist() == [1, 2]


def test_oracle_point_is_mean_of_selected():
    perf, cost = _case()
    c, q = OracleRouter().point(perf, cost)
    assert abs(c - (0.2 + 0.1 + 0.05) / 3) < 1e-12
    assert abs(q - (1 + 0 + 1) / 3) < 1e-12


def test_oracle_selection_share_sums_to_one():
    perf, cost = _case()
    share = OracleRouter().selection_share(perf, cost)
    assert abs(share.sum() - 1.0) < 1e-12
    assert share.tolist() == [0.0, 1 / 3, 2 / 3]


def test_point_router_returns_its_own_model():
    perf, cost = _case()
    r = PointRouter(1, name="B")
    assert r.select(perf, cost).tolist() == [1, 1, 1]
    c, q = r.point(perf, cost)
    assert abs(c - cost[:, 1].mean()) < 1e-12
    assert abs(q - perf[:, 1].mean()) < 1e-12


def test_zero_curve_equals_ndch_of_points():
    pts = np.array([[0.0, 0.2], [1.0, 0.8], [2.0, 0.5], [3.0, 0.9]])
    z = ZeroRouter(pts)
    assert np.allclose(z.curve, ndch(pts))


def test_zero_curve_dominates_every_point():
    pts = np.array([[0.0, 0.2], [1.0, 0.8], [2.0, 0.5], [3.0, 0.9]])
    z = ZeroRouter(pts)
    for c, q in pts:
        assert z.quality_at(c) >= q - 1e-12


def test_zero_curve_is_non_decreasing():
    rng = np.random.default_rng(3)
    pts = np.column_stack([np.sort(rng.random(12)), rng.random(12)])
    curve = ZeroRouter(pts).curve
    assert np.all(np.diff(curve[:, 1]) >= -1e-12)
