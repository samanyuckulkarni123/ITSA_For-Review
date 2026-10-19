# Manuscript tables

Use “average precision (AP)” consistently. Confidence intervals are paired percentile intervals from 2,000 superfamily-clustered bootstrap replicates.

# Table 1. Corrected benchmark design

| Domains | Families | Superfamilies | Pairs | Positive pairs | Positive prevalence |
|---:|---:|---:|---:|---:|---:|
| 222 | 28 | 24 | 24,520 | 828 | 3.38% |

# Table 2. Pair-level discrimination

| Method | ROC AUC | AP |
|---|---:|---:|
| Raw ordered ITSA | 0.8775 | 0.3745 |
| Shuffled ITSA | 0.7294 | 0.1019 |
| Composition cosine | 0.5762 | 0.0456 |
| Exhaustive Foldseek probability | 0.9854 | 0.7799 |
| TM-align | 0.9935 | 0.9001 |

# Table 3. Equal-superfamily query retrieval

| Method | AP (95% CI) | AUC (95% CI) | P@10 (95% CI) | R@10 (95% CI) |
|---|---:|---:|---:|---:|
| Raw ITSA | 0.5814 (0.5018–0.6616) | 0.9148 (0.8805–0.9417) | 0.4061 (0.3457–0.4656) | 0.5923 (0.5162–0.6683) |
| Optimized ITSA | 0.7016 (0.6239–0.7737) | 0.9488 (0.9259–0.9676) | 0.4812 (0.4167–0.5451) | 0.7028 (0.6248–0.7799) |
| Foldseek | 0.9134 (0.8407–0.9764) | 0.9893 (0.9777–0.9982) | 0.6458 (0.5633–0.7211) | 0.9341 (0.8739–0.9835) |
| TM-align | 0.9597 (0.9137–0.9936) | 0.9949 (0.9885–0.9997) | 0.6610 (0.5766–0.7342) | 0.9512 (0.9006–0.9917) |

# Table 4. Nested model selection

| Selected feature | Outer folds selected |
|---|---:|
| Normalized ITSA alignment | 24/24 |
| Raw alignment / geometric mean self-score | 24/24 |
| Secondary-structure Jensen-Shannon similarity | 24/24 |

The complete 24-fold selection record is in table_4b_outer_fold_selections.csv. The complete 28-family AP/AUC table is in table_5_family_level_results.csv and is suitable for the main text or supplement.
