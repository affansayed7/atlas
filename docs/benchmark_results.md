# BKT Benchmark Results — ASSISTments 2009-2010 Skill Builder

## Dataset
- 525,534 raw attempts → 459,208 after cleaning (dropped 12.6% skill-less rows)
- 4,163 students, 123 skills

## Methodology
- Per (student, skill) knowledge sequences
- Predict-then-observe protocol (each attempt predicted from prior history only — no leakage)
- Metric: AUC (ROC)

## Results
| Model | AUC |
|-------|-----|
| BKT (from scratch) | 0.7018 |
| Naive per-skill baseline | 0.6544 |
| **BKT improvement** | **+0.0474** |

## Interpretation
- BKT AUC of 0.70 is consistent with published results for ASSISTments 2009 (~0.70–0.73), validating the implementation.
- BKT beats the per-skill baseline, demonstrating it captures individual learning trajectories beyond average skill difficulty.

## Default parameters
- P(L0)=0.2, P(T)=0.3, P(slip)=0.1, P(guess)=0.2 (untuned — future work: fit per-skill)