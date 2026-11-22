## 目录结构

```
RouterBench/
├── configs/
│   ├── default.json             # 合成数据（默认）
│   └── real.json                # 真实数据：source=real, aggregate=macro, 输出到 results_real/
├── routerbench/
│   ├── constants.py             # 11 模型、7 数据集、论文 Table 1 真值、样本量
│   ├── config.py                # Config dataclass + JSON 加载
│   ├── realdata.py              # HF 宽表 -> 长表（shot/split/数据集归并）
│   ├── data.py                  # 长格式契约：校验/聚合（pooled|macro）/按样本切分/Oracle
│   ├── metrics.py               # NDCH、AIQ（共享域 & 官方域两套）、插值积分
│   ├── surrogate.py             # 合成替代数据（双因子 IRT + 秩标定）
│   ├── routers/                 # Router 接口：base / oracle / zero
│   ├── plotting.py              # 2×4 cost-quality 网格、Oracle 选择热图
│   ├── report.py                # 矩阵转 markdown、与论文逐格对比
│   └── pipeline.py              # 四步流水线（纯函数，可被测试直接调用）
├── scripts/                     # 薄 CLI：download / convert / step1-3 / run_p0
├── tests/                       # 自带 runner（本机无 pytest）
├── data/raw/                    # 官方 pkl（不入库）
├── data/real/                   # 转换后的长表缓存
├── results/                     # 合成数据产物（冒烟测试）
└── results_real/                # 真实数据产物（主结果）+ REPORT.md
```

设计上区分 **库（routerbench）** 与 **入口（scripts）**：所有计算都能 `import routerbench` 直接调用，
脚本只做参数解析与打印。

---


