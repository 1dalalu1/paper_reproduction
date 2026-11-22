"""运行参数与路径解析。

配置文件为 JSON（避免额外依赖），路径一律相对工程根目录书写，
载入时解析成绝对路径，脚本无论在哪个工作目录启动都能跑。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

#: 工程根目录 = 本文件的上上级（routerbench/config.py -> 工程根）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"


@dataclass
class Config:
    root: Path
    raw_dir: Path
    surrogate_path: Path
    derived_dir: Path
    tables_dir: Path
    figures_dir: Path
    source: str = "surrogate"          # "surrogate" | "real"
    seed: int = 20240501
    cost_cv: float = 0.6               # surrogate 成本离散度
    domain_mode: str = "intersection"  # AIQ 共享定义域取法
    aggregate_mode: str = "pooled"     # "pooled" | "macro"（详见 data.aggregate）
    real_path: Path | None = None      # source="real" 时的长表缓存
    dpi: int = 300
    figsize: tuple = (18.0, 7.6)
    raw: Dict[str, Any] = field(default_factory=dict)

    def ensure_dirs(self) -> None:
        for d in (
            self.raw_dir,
            self.surrogate_path.parent,
            self.derived_dir,
            self.tables_dir,
            self.figures_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # 便捷取数 ------------------------------------------------------------
    @property
    def table1_csv(self) -> Path:
        return self.tables_dir / "table1_reproduced.csv"


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    paths = raw.get("paths", {})
    data = raw.get("data", {})
    metrics = raw.get("metrics", {})
    figure = raw.get("figure", {})

    def _p(key: str, default: str) -> Path:
        p = Path(paths.get(key, default))
        return p if p.is_absolute() else PROJECT_ROOT / p

    cfg = Config(
        root=PROJECT_ROOT,
        raw_dir=_p("raw_dir", "data/raw"),
        surrogate_path=_p("surrogate", "data/surrogate/routerbench_surrogate.csv"),
        derived_dir=_p("derived_dir", "data/derived"),
        tables_dir=_p("tables_dir", "results/tables"),
        figures_dir=_p("figures_dir", "results/figures"),
        source=data.get("source", "surrogate"),
        seed=int(data.get("seed", 20240501)),
        cost_cv=float(data.get("cost_cv", 0.6)),
        aggregate_mode=data.get("aggregate", "pooled"),
        domain_mode=metrics.get("domain_mode", "intersection"),
        real_path=_p("real_long", "data/real/routerbench_long.pkl"),
        dpi=int(figure.get("dpi", 300)),
        figsize=tuple(figure.get("figsize", [18.0, 7.6])),
        raw=raw,
    )
    return cfg
