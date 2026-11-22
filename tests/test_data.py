"""data / surrogate 的单元测试：校验、聚合、按样本切分、合成数据的标定精度。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from routerbench import constants as C
from routerbench.data import (
    aggregate,
    oracle_select,
    split_by_sample,
    to_wide,
    validate_longframe,
)
from routerbench.surrogate import generate

MODELS = ["A", "B"]
DATASETS = ["d1"]


def _tiny() -> pd.DataFrame:
    rows = []
    for i in range(10):
        for j, m in enumerate(MODELS):
            rows.append(
                {
                    "sample_id": f"d1-{i:02d}",
                    "dataset": "d1",
                    "model": m,
                    "performance": float((i + j) % 2),
                    "cost": 0.1 * (j + 1),
                }
            )
    return pd.DataFrame(rows)


def test_validate_accepts_well_formed_frame():
    rep = validate_longframe(_tiny(), models=MODELS, datasets=DATASETS)
    assert rep["errors"] == []
    assert rep["n_rows"] == 20


def test_validate_rejects_duplicate_rows():
    df = pd.concat([_tiny(), _tiny().head(1)], ignore_index=True)
    try:
        validate_longframe(df, models=MODELS, datasets=DATASETS)
    except ValueError as exc:
        assert "重复" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("重复行没有被拦下来")


def test_validate_rejects_out_of_range_performance():
    df = _tiny()
    df.loc[0, "performance"] = 1.5
    try:
        validate_longframe(df, models=MODELS, datasets=DATASETS)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("越界 performance 没有被拦下来")


def test_aggregate_sums_cost_and_means_perf():
    agg = aggregate(_tiny(), models=MODELS, datasets=DATASETS).set_index("model")
    assert abs(agg.loc["A", "cost_sum"] - 1.0) < 1e-12
    assert abs(agg.loc["B", "cost_sum"] - 2.0) < 1e-12
    assert abs(agg.loc["A", "perf_mean"] - 0.5) < 1e-12
    assert abs(agg.loc["A", "cost_mean"] - 0.1) < 1e-12


def test_to_wide_shape_and_column_order():
    perf, cost = to_wide(_tiny(), "d1", models=MODELS)
    assert perf.shape == (10, 2)
    assert list(perf.columns) == MODELS
    assert list(cost.columns) == MODELS


def test_split_by_sample_keeps_all_models_of_a_sample_together():
    df = _tiny()
    perf, cost = to_wide(df, "d1", models=MODELS)
    split = split_by_sample(perf, cost, test_size=0.3, seed=0)
    train_idx, test_idx = split["train"][2], split["test"][2]
    assert len(test_idx) == 3 and len(train_idx) == 7
    assert set(train_idx).isdisjoint(set(test_idx))
    assert sorted(np.concatenate([train_idx, test_idx])) == list(range(10))
    # 行索引即样本，切分后每行仍带两列 => 同一样本不会被拆开
    assert split["train"][0].shape == (7, 2)


def test_oracle_select_prefers_cheapest_correct():
    perf = np.array([[1, 0], [0, 0], [1, 1]], dtype=float)
    cost = np.array([[0.9, 0.1], [0.5, 0.4], [0.2, 0.3]], dtype=float)
    assert oracle_select(perf, cost).tolist() == [0, 1, 0]


def test_surrogate_is_deterministic_and_calibrated():
    sizes = {
        "mmlu": 150,
        "hellaswag": 120,
        "arc": 60,
        "winogrande": 60,
        "gsm8k": 60,
        "mbpp": 50,
        "mt_bench": 40,
    }
    df1, _ = generate(seed=7, sample_sizes=sizes)
    df2, _ = generate(seed=7, sample_sizes=sizes)
    assert df1.equals(df2)
    assert len(df1) == sum(sizes.values()) * len(C.MODELS)

    agg = aggregate(df1).set_index(["dataset", "model"])
    devs = []
    for ds in C.DATASETS:
        for m in C.MODELS:
            target_perf = C.TABLE1[m][C.DATASETS.index(ds)][0]
            target_cost = C.TABLE1[m][C.DATASETS.index(ds)][1]
            n = int(agg.loc[(ds, m), "n"])
            dev = abs(agg.loc[(ds, m), "perf_mean"] - target_perf)
            # 秩标定 / 均值标定是精确的，偏差只可能来自 round(target * n) 的取整
            assert dev <= 1.5 / n + 1e-9, (ds, m, dev, 1.5 / n)
            devs.append(dev)
            # 成本之和必须精确对齐（这是构造出来的不变量）
            assert abs(agg.loc[(ds, m), "cost_sum"] / target_cost - 1.0) < 1e-9
    assert max(devs) < 0.02
