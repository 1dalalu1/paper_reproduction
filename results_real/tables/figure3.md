# Figure 3 复现结果

数据来源：**real**

定义域取法：每个数据集上 [min 单模型平均成本, max 单模型平均成本]，所有系统在同一区间上计算 AIQ（论文 §3.2 要求显式共享定义域）。

## AIQ（NDCH 归一化面积）

|  | MMLU | MT-Bench | MBPP | HellaSwag | Winogrande | GSM8K | ARC-Challenge |
|---|---|---|---|---|---|---|---|
| WizardLM 13B | 0.538 | 0.457 | 0.370 | 0.598 | 0.528 | 0.506 | 0.651 |
| Mistral 7B | 0.531 | 0.445 | 0.344 | 0.537 | 0.566 | 0.412 | 0.671 |
| Mixtral 8x7B | 0.689 | 0.559 | 0.538 | 0.683 | 0.670 | 0.516 | 0.837 |
| Code Llama 34B | 0.523 | 0.459 | 0.516 | 0.502 | 0.572 | 0.452 | 0.649 |
| Yi 34B | 0.708 | 0.568 | 0.383 | 0.855 | 0.737 | 0.541 | 0.880 |
| GPT-3.5 | 0.657 | 0.623 | 0.645 | 0.764 | 0.620 | 0.592 | 0.802 |
| Claude Instant V1 | 0.420 | 0.557 | 0.592 | 0.749 | 0.505 | 0.612 | 0.804 |
| Llama 70B | 0.599 | 0.499 | 0.327 | 0.690 | 0.489 | 0.516 | 0.776 |
| Claude V1 | 0.307 | 0.428 | 0.463 | 0.501 | 0.344 | 0.467 | 0.557 |
| Claude V2 | 0.365 | 0.478 | 0.444 | 0.192 | 0.292 | 0.433 | 0.364 |
| GPT-4 | 0.414 | 0.380 | 0.345 | 0.460 | 0.438 | 0.333 | 0.488 |
| Zero router | 0.761 | 0.690 | 0.666 | 0.878 | 0.797 | 0.645 | 0.921 |
| Oracle | 0.948 | 0.853 | 0.848 | 0.979 | 0.997 | 0.731 | 0.981 |

## Zero router 是否优于最好的单模型（论文核心结论之一）

| dataset | zero_aiq | best_single_aiq | best_single | zero_beats_best_single |
|---|---|---|---|---|
| mmlu | 0.761 | 0.708 | Yi 34B | True |
| mt_bench | 0.690 | 0.623 | GPT-3.5 | True |
| mbpp | 0.666 | 0.645 | GPT-3.5 | True |
| hellaswag | 0.878 | 0.855 | Yi 34B | True |
| winogrande | 0.797 | 0.737 | Yi 34B | True |
| gsm8k | 0.645 | 0.612 | Claude Instant V1 | True |
| arc | 0.921 | 0.880 | Yi 34B | True |

## 相近质量下的成本倍数

| dataset | quality_best | n_models_near | cheapest | priciest | cost_ratio |
|---|---|---|---|---|---|
| mmlu | 0.812 | 1 | GPT-4 | GPT-4 | 1.000 |
| mt_bench | 0.757 | 1 | GPT-4 | GPT-4 | 1.000 |
| mbpp | 0.686 | 2 | GPT-3.5 | GPT-4 | 28.011 |
| hellaswag | 0.902 | 2 | Yi 34B | GPT-4 | 12.532 |
| winogrande | 0.859 | 1 | GPT-4 | GPT-4 | 1.000 |
| gsm8k | 0.663 | 3 | Claude V1 | GPT-4 | 1.720 |
| arc | 0.957 | 2 | Claude V1 | GPT-4 | 1.250 |

## 产物

- `E:\论文复现\RouterBench\results_real\figures\figure3_cost_quality.png`
- `E:\论文复现\RouterBench\results_real\figures\figure3_cost_quality_logx.png`
- `E:\论文复现\RouterBench\results_real\figures\figure3_oracle_selection.png`
- `E:\论文复现\RouterBench\results_real\tables\oracle_selection_share.csv`
- `E:\论文复现\RouterBench\results_real\tables\table2_aiq.csv`
- `E:\论文复现\RouterBench\results_real\tables\zero_router_vs_best_single.csv`
