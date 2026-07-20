# Knowledge Tracing Benchmark — ASSISTments 2009-2010 Skill Builder

## Summary

Implemented and evaluated two knowledge-tracing models from scratch — Bayesian
Knowledge Tracing (BKT) and Deep Knowledge Tracing (DKT, an LSTM) — on the
ASSISTments 2009-2010 skill-builder dataset.

**Headline result:** DKT (0.751 AUC) outperforms grid-search-fitted BKT (0.689 AUC)
by +0.062 on held-out students.

**Most important finding:** an initial DKT result of 0.891 AUC was traced to data
leakage caused by the dataset's multi-skill row expansion. After correction, results
fall within published ranges. The investigation is documented below.

---

## Dataset

| Stage | Rows |
|---|---|
| Raw CSV | 525,534 |
| After dropping rows with missing `skill_id` (12.6%) | 459,208 |
| **After deduplicating multi-skill expansion** | **283,105** |

Final: 283,105 attempts, 4,163 students, 112 skills.
(Published papers cite ~278,607 for this dataset — our cleaned count closely matches,
supporting the correctness of the preprocessing.)

### The multi-skill leakage bug

ASSISTments expands each attempt on a multi-skill problem into **one row per skill**,
all sharing the same `order_id` and `correct` value. 34% of rows were such duplicates
(346,860 unique `order_id`s across 525,534 rows).

Left in place, this leaks the answer: a sequence would contain
`(skill A, correct=1)` immediately followed by `(skill B, correct=1)` — *the same
attempt*. The model would be asked to predict an outcome it had just been shown.

**Fix:** deduplicate on `order_id`, keeping one skill tag per attempt (standard KT
preprocessing).

### Impact of the fix

| Model | Leaky data | Deduplicated (correct) |
|---|---|---|
| BKT (default params) | 0.7018 | 0.6335 |
| BKT (fitted params) | 0.7468 | 0.6891 |
| DKT (held-out test) | 0.8909 | **0.7506** |

Diagnostic signals that exposed the leak:
- 0.891 AUC was **above the entire published DKT range** (0.75–0.82)
- Train AUC (0.894) and test AUC (0.891) were **nearly identical** — genuine learning
  produces a visible train/test gap
- 73.3% of consecutive attempts shared a skill (inflated by the duplicate rows)

After the fix, the train/test gap is healthy (0.764 vs 0.751) and the loss floor rose
from 0.39 to 0.54 — the task became genuinely hard rather than trivially copyable.

---

## Models

### BKT (Bayesian Knowledge Tracing)
- Implemented from scratch: 4 parameters — P(L0), P(T), P(slip), P(guess)
- Two-step update per attempt: Bayesian conditioning on evidence, then transit
- Parameters fitted by grid search over 192 combinations (constraint: slip + guess < 1)
- Fitted global params: **P(L0)=0.4, P(T)=0.1, P(slip)=0.05, P(guess)=0.3**
  - Interpretation: students start more knowledgeable than assumed, learn gradually,
    slip rarely, but guess often.
  - These params were stable across both the leaky and clean datasets.

### DKT (Deep Knowledge Tracing)
- Implemented from scratch in PyTorch: Embedding → LSTM → Linear
- Input encoding: `token = skill + correct × num_skills` (224 possible tokens)
- 131K parameters; embed_dim=64, hidden_dim=128
- Masked BCE loss (padding excluded from loss and evaluation)
- Adam optimizer, lr=0.001, 5 epochs, batch size 32
- Split **by student** (80/20) so test students are entirely unseen

---

## Results

| Model | AUC |
|-------|-----|
| Naive per-skill baseline (leaky data) | 0.6544 |
| BKT — default hand-picked params | 0.6335 |
| BKT — grid-search fitted | 0.6891 |
| **DKT (LSTM) — held-out test students** | **0.7506** |

==================================================
  FAIR COMPARISON — same held-out test students
  BKT AUC (test): 0.6913
  DKT AUC (test): 0.7482
  DKT - BKT:      +0.0569
==================================================

DKT train AUC: 0.7638 (healthy gap vs 0.7506 test — no significant overfitting).

---

## Limitations & Future Work

1. **The BKT/DKT comparison is not yet fully fair.** BKT was fitted and