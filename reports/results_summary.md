# Measured results

| Model | n | Accuracy | Numeric accuracy | Format validity | Abstention recall |
|---|---:|---:|---:|---:|---:|
| baseline | 120 | 0.125 | 0.000 | 0.500 | 0.375 |
| sft | 120 | 0.408 | 0.067 | 0.992 | 1.000 |
| dpo | 120 | 0.500 | 0.100 | 0.975 | 1.000 |
| dpo_beta05 | 120 | 0.500 | 0.100 | 0.942 | 1.000 |

Reserved replication:

| Model | n | Accuracy | Numeric accuracy | Format validity | Abstention recall |
|---|---:|---:|---:|---:|---:|
| baseline | 120 | 0.158 | 0.000 | 0.508 | 0.458 |
| sft | 120 | 0.242 | 0.000 | 0.933 | 0.500 |
| dpo | 120 | 0.350 | 0.000 | 0.925 | 0.500 |
| dpo_beta05 | 120 | 0.325 | 0.000 | 0.958 | 0.375 |

Reward test accuracy: 86.7%; final: 87.5%.

See research_report.md for limitations and paired intervals.
