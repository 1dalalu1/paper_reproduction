# RouterBench P0 基线复现

对论文 **《ROUTERBENCH: A Benchmark for Multi-LLM Routing System》**（arXiv 2403.12031）的
评估框架做工程化复现，覆盖复现指南的 **Phase 0–4（P0 基线）**：

- **数据**：接官方 HuggingFace 数据集 `withmartian/routerbench`，宽表 → 统一长表的转换与校验；
- **聚合**：Table 1（11 模型 × 7 数据集的 Perf / Cost）按论文口径复算；
- **度量**：NDCH（非降凸包）、AIQ（归一化面积）、共享定义域，实现与校验；
- **可视化**：Figure 3 风格的逐数据集 cost-quality 平面、Oracle 选择分布。

论文的三块资产是 ① 40 万条预生成推理结果 ② 评估框架 ③ 路由器算法。本仓库复现 **②**，
并把 ① 完整接了进来；③（KNN / MLP / Cascading）是下一步 P1/P2。

---

## 0. 先看结论

| 指标                       | 真实数据（`results_real/`）     | 合成数据（`results/`）  |
| ------------------------ | ------------------------- | ----------------- |
| 行数 / 数据集                 | 379,280 行 / 7 个           | 312,642 行 / 7 个   |
| Table 1 Perf 落在 0.01 容差内 | **44 / 77**               | 77 / 77（构造性，不算证据） |
| Table 1 Cost 落在容差内       | 1 / 77（口径差 2.6–30×）       | 77 / 77（构造性）      |
| Zero router 优于最强单模型      | **7 / 7**                 | 7 / 7             |
| 与论文不可比的部分                | MT-Bench（平均差 0.315）、成本绝对值 | ——                |

**核心结论：论文"Zero router 优于任何单一模型"的中心论断在真实数据上 7/7 成立**；
但 Table 1 无法逐位复现——公开数据与论文所用快照不是同一批推理结果（详见
`results_real/REPORT.md` 第 4 节，有 5 条可证伪的证据）。

合成数据那条线保留下来，作为"没有网络时的冒烟测试"：它按论文 Table 1 标定，
所以 77/77 是构造出来的，**不能当作复现证据**。

---

## 1. 快速开始

```bash
# 依赖（本机已全部可用）
pip install -r requirements.txt

# ① 下载官方数据（约 270 MB，需要能访问 huggingface.co）
python scripts/download_data.py

# ② 宽表 -> 长表（论文口径：5-shot / 0-shot 混用，ARC 取 test 划分，丢弃 RAG 子集）
python scripts/step0b_convert_real.py --config configs/real.json

# ③ 一键跑完 P0（校验 -> Table 1 -> Figure 3），约 5 秒
python scripts/run_p0.py --config configs/real.json
```

没有网络时用合成数据跑通同一条链路：

```bash
python scripts/run_p0.py --force-surrogate      # 产物在 results/
```

也可以分步执行：`step0b_convert_real.py` → `step1_validate_data.py` →
`step2_reproduce_table1.py` → `step3_reproduce_figure3.py`（后者加 `--config configs/real.json`）。

> Windows 传统 cmd 下如需正确显示中文：`chcp 65001`（脚本已把 stdout 设为 UTF-8）。
> 下载脚本走 Python + HTTP 代理（默认探测 `127.0.0.1:7897/7890/10809`），
> 因为本机沙箱里 curl 的 Schannel 与 `Invoke-WebRequest` 都无法出网。

---

## 2. 数据：从 HuggingFace 到长表

数据集 `withmartian/routerbench` 提供两个**宽表**（一行一个 prompt，每模型三列：
质量分、`<model>|model_response`、`<model>|total_cost`）：

| 文件                      | 大小      | 形状          | 用途                                          |
| ----------------------- | ------- | ----------- | ------------------------------------------- |
| `routerbench_5shot.pkl` | 171 MB  | 36,511 × 37 | MMLU / HellaSwag / Winogrande / GSM8K / ARC |
| `routerbench_0shot.pkl` | 99.6 MB | 36,497 × 37 | MBPP / MT-Bench（+ RAG 子集）                   |

`routerbench_raw.pkl`（1.2 GB）是转换前的长格式原始输出，仅供追溯用。

`routerbench/realdata.py` 定义了转换口径，全部可在 `configs/real.json` 里改：

| 决策             | 取值                                            | 理由                                                                  |
| -------------- | --------------------------------------------- | ------------------------------------------------------------------- |
| `shot_map`     | 五个推理集用 5-shot，MBPP / MT-Bench 用 0-shot        | 论文第 4 节的推理设置                                                        |
| `split_filter` | `arc: test`                                   | ARC test 划分 1172 题 = 论文 Table 1 的样本数；公开文件里还混了 312 行 val             |
| 丢弃 RAG         | 是                                             | 论文的 800 条 Martian 客户查询不公开；公开数据里对应的是 1719 条中文/商业 prompt，与 Table 1 无关 |
| NaN 质量         | 先丢整行全 NaN 的记录（5-shot ARC 有 14 行），再把残余 NaN 记 0 | 与官方 `convert_data.py` 的 `fillna(0.0)` 一致                            |
| 模型列名           | `WizardLM/WizardLM-13B-V1.2` 等 11 个 → 本仓库短名   | 见 `realdata.MODEL_COLUMNS`                                          |

转换后：**34,480 个样本 × 11 模型 = 379,280 行**，7 个数据集，`errors=0`。
（MMLU 14,042 / HellaSwag 10,042 / GSM8K 7,450 / ARC 1,172 / Winogrande 1,267 / MBPP 427 / MT-Bench 80）

---

## 3. 目录结构

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

## 4. 数据契约

长格式，一行 = 一个样本在一个模型上的一次推理：

| 列                | 类型      | 含义                                                                          |
| ---------------- | ------- | --------------------------------------------------------------------------- |
| `sample_id`      | str     | 样本唯一标识（跨模型一致，切分 train/test 的唯一依据）                                           |
| `dataset`        | str     | `mmlu` / `mt_bench` / `mbpp` / `hellaswag` / `winogrande` / `gsm8k` / `arc` |
| `model`          | str     | 11 个模型之一（`constants.MODELS` 的写法）                                            |
| `performance`    | float   | 质量分 ∈ [0,1]；exact-match 集取 0/1                                              |
| `cost`           | float   | 该次推理的美元成本（per-sample）                                                       |
| `eval_name`      | str     | 可选，原始子任务名（MMLU 的 57 个学科），`aggregate(mode="macro")` 要用                       |
| `shot` / `split` | int/str | 可选，出处信息（5/0、val/dev/test）                                                   |

`data.validate_longframe` 检查：列齐全、无 NaN、`performance ∈ [0,1]`、`cost ≥ 0`、
`(sample_id, model)` 唯一、每个数据集内 11 模型 × 样本数 = 行数。

**聚合两种口径**：`pooled`（按样本加权）与 `macro`（先按 `eval_name` 求均值再等权平均，
官方 `evaluation/eval.py` 用的是这个——MMLU 57 个学科题数差异很大，两者差 1–2 个点）。
`configs/real.json` 用 `macro`。

---

## 5. 流水线与产物

| 步骤  | 做什么                      | 产物（`results_real/`）                                                                                                                         |
| --- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| 0b  | 宽表 → 长表                  | `data/real/routerbench_long.pkl`、`data/derived_real/real_data_card.json`                                                                    |
| 1   | 契约校验 + 完整性矩阵             | `tables/data_validation.md`                                                                                                                 |
| 2   | 聚合 Table 1 + 逐格对比        | `tables/table1_{reproduced.csv,md}`、`table1_vs_paper.{csv,md}`                                                                              |
| 3   | cost-quality 平面、NDCH、AIQ | `figures/figure3_*.png`、`tables/{table2_aiq,zero_router_vs_best_single,cost_ratio_matched_quality,model_points,oracle_selection_share}.csv` |

`data/derived_real/p0_summary.json` 汇总全部关键数字。

---

## 6. 度量实现

- **成本口径**：Table 1 的 Cost 列是**该数据集全部样本的成本之和**；图上与 AIQ 用**每样本均值**。
  两者只差常数 `n_d`，对 AIQ 这种归一化面积无影响。
- **NDCH**：按成本升序、同成本取最大质量 → 求上凸包 → 强制质量随成本单调不减。
- **AIQ**：`1/(c_max−c_min) · ∫ f(c) dc`，在断点取节点做梯形求和（分段线性下精确）。
  定义域必须显式共享，默认取所有曲线定义域的**交集**。
- **官方口径 `aiq_official`**：官方 `evaluation/AIQ.py` 的做法是
  `trapz / max_x`（[0, 首点] 段按零面积、曲线右端水平外推）。两种口径在真实数据的
  Zero router 上平均差 **0.0115**（官方总是略低），表里同时给出，便于和论文 Table 2 对照。
- **Oracle**：逐样本在"答对的模型"里挑最便宜；全错则挑最便宜。Perf/Cost 由数据涌现。
- **Zero router**：对 11 个单模型点取 NDCH 做期望混合，不做蒙特卡洛抽样。

---

## 7. 测试

```bash
python tests/run_tests.py        # 约 1 秒
python tests/run_tests.py -v
```

覆盖 NDCH 凸包 / AIQ 解析解 / 共享定义域 / 插值积分 / 路由器选择 / 数据契约与聚合 /
按样本切分 / 合成数据标定精度 / 出图不崩。本机无 pytest，故自带极简 runner
（发现 `tests/test_*.py` 中的 `test_*` 函数）；有 pytest 时 `pytest tests/` 同样可用。

---

## 8. 已知限制

1. **Table 1 不能逐位复现**：公开数据 ≠ 论文快照（MMLU 差约 0.03、MT-Bench 差 0.31、
   GSM8K 行数 7450 vs 1319、MBPP 427 vs 500），成本口径更是不同定价快照。
2. **RAG 子集缺失**：论文的 800 条 Martian 查询未公开，Table 1 也不含它。
3. **MT-Bench 只有 80 题且被拆成三组**（`mtbench` 41 / `mtbench-math` 20 / `mtbench-reference` 19），
   三组难度差异很大，论文把它排除在预测式训练-测试之外。
4. **embedding 相关实验（P1）暂不可用**：`sentence-transformers` 与模型权重需要额外下载。

---

## 9. 下一步（Phase 5+）

- **P1 预测式路由器**：KNN（k∈{5,10,40} × 3 种 embedding × cosine）与 MLP（1–2 层 ×100，ReLU，lr=1e-3），
  `P_ij − λ·cost_j` 取 argmax，λ 扫描成曲线；
- **P2 级联**：按成本升序调用，完美判官 + 判官误差 ε，预算 T 在调用下一个模型**之前**判断；
- **P3 三操作**：线性插值、外推（含 null router (0,0)）、统一坐标系后比 AIQ。


