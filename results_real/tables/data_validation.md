# 数据校验

## 每个数据集的完整性

|  | n_samples | n_models | n_rows |
|---|---|---|---|
| MMLU | 14042 | 11 | 154462 |
| MT-Bench | 80 | 11 | 880 |
| MBPP | 427 | 11 | 4697 |
| HellaSwag | 10042 | 11 | 110462 |
| Winogrande | 1267 | 11 | 13937 |
| GSM8K | 7450 | 11 | 81950 |
| ARC-Challenge | 1172 | 11 | 12892 |

## 校验结论

- 必需列齐全、无 NaN、performance ∈ [0,1]、cost ≥ 0：通过
- (sample_id, model) 无重复：通过
- 每个数据集内 样本数 × 11 = 行数：通过
