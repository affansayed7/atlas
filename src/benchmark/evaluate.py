"""BKT benchmark evaluation on ASSISTments — predict-then-observe protocol.

Protocol (critical for validity):
- For each attempt, BKT predicts P(correct) using ONLY prior attempts.
- We record (prediction, actual), THEN update BKT with the actual outcome.
- Predicting an outcome you've already seen would be data leakage — the
  cardinal ML sin. Predict BEFORE update.

Metric: AUC (ranking quality), compared against a naive per-skill baseline.
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

from src.analytics.bkt import BKT, BKTParams

DATA = Path(__file__).resolve().parents[2] / "data" / "skill_builder_data.csv"


def load_and_clean() -> pd.DataFrame:
    """Load ASSISTments, drop skill-less rows, DEDUPE multi-skill expansions, sort."""
    df = pd.read_csv(DATA, encoding="ISO-8859-15", low_memory=False)
    df = df[["user_id", "skill_id", "correct", "order_id"]].copy()
    df = df.dropna(subset=["skill_id"])
    df = df[df["correct"].isin([0, 1])]

    # CRITICAL: multi-skill problems are expanded into one row PER SKILL, sharing
    # the same order_id. Keeping them all leaks the answer (the model would see
    # "skill A -> correct" then be asked to predict "skill B" of the SAME attempt).
    # Standard KT preprocessing: keep one row per attempt.
    before = len(df)
    df = df.drop_duplicates(subset=["order_id"], keep="first")
    print(f"Deduped multi-skill rows: {before:,} -> {len(df):,} "
          f"({before - len(df):,} duplicate-attempt rows removed)")

    df = df.sort_values("order_id")
    print(f"After cleaning: {len(df):,} attempts, "
          f"{df['user_id'].nunique():,} students, {df['skill_id'].nunique():,} skills")
    return df

def evaluate_bkt(df: pd.DataFrame, params: BKTParams) -> tuple[list, list]:
    """Predict each attempt from prior history (per student+skill). Returns (preds, actuals)."""
    preds, actuals = [], []

    # Group by student AND skill: each (student, skill) is one knowledge sequence
    for (_uid, _sid), group in df.groupby(["user_id", "skill_id"], sort=False):
        bkt = BKT(params)
        for correct in group["correct"].astype(int):
            # PREDICT first, using current knowledge state (prior attempts only)
            p = bkt.params
            p_correct = bkt.p_known * (1 - p.p_slip) + (1 - bkt.p_known) * p.p_guess
            preds.append(p_correct)
            actuals.append(correct)
            # THEN update with the observed outcome
            bkt.update(bool(correct))

    return preds, actuals


def evaluate_baseline(df: pd.DataFrame) -> tuple[list, list]:
    """Naive baseline: predict each skill's global success rate for every attempt."""
    skill_rates = df.groupby("skill_id")["correct"].mean().to_dict()
    preds = [skill_rates[sid] for sid in df["skill_id"]]
    actuals = df["correct"].astype(int).tolist()
    return preds, actuals


def main():
    df = load_and_clean()

    print("\nRunning BKT evaluation...")
    bkt_preds, bkt_actuals = evaluate_bkt(df, BKTParams())
    bkt_auc = roc_auc_score(bkt_actuals, bkt_preds)

    print("Running baseline evaluation...")
    base_preds, base_actuals = evaluate_baseline(df)
    base_auc = roc_auc_score(base_actuals, base_preds)

    print("\n" + "=" * 40)
    print(f"  BKT AUC:      {bkt_auc:.4f}")
    print(f"  Baseline AUC: {base_auc:.4f}")
    print(f"  BKT beats baseline by: {bkt_auc - base_auc:+.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()