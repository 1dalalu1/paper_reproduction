# Table 1 与论文逐格对比

数据来源：**real**

- 单元格数：77
- Perf 通过：44 / 77（57.1%）
- Cost 通过：1 / 77（1.3%）
- 最大 |Perf 偏差|：0.3644
- 最大相对 Cost 偏差：2955.66%

判定规则：Perf 允许 |diff| ≤ 0.01 或 ≤ 3 倍二项标准误；Cost 允许 |diff| ≤ max(0.01, 1%×论文值)。

## 偏差最大的 5 格

| dataset | model | perf_diff | perf_z |
|---|---|---|---|
| MT-Bench | Yi 34B | -0.364 | 13.514 |
| MT-Bench | Claude V1 | -0.364 | 13.491 |
| MT-Bench | Mixtral 8x7B | -0.359 | 11.920 |
| MT-Bench | Llama 70B | -0.349 | 8.844 |
| MT-Bench | WizardLM 13B | -0.338 | 7.505 |

## Oracle 行（由逐样本数据涌现，非标定量）

- Perf 通过 6 / 7，Cost 通过 0 / 7

| dataset | perf_repro | perf_paper | perf_diff | cost_repro | cost_paper |
|---|---|---|---|---|---|
| MMLU | 0.966 | 0.957 | 0.009 | 8.770 | 0.297 |
| MT-Bench | 0.919 | 0.996 | -0.077 | 0.279 | 0.052 |
| MBPP | 0.867 | 0.899 | -0.032 | 0.206 | 0.041 |
| HellaSwag | 0.984 | 0.994 | -0.010 | 3.232 | 0.860 |
| Winogrande | 0.999 | 1.000 | -0.001 | 0.136 | 0.042 |
| GSM8K | 0.749 | 0.748 | 0.001 | 4.319 | 1.282 |
| ARC-Challenge | 0.988 | 0.977 | 0.011 | 0.268 | 0.091 |
