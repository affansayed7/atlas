"""First look at the ASSISTments dataset — shape, columns, missing values, duplicates.

DUPLICATE CHECK RATIONALE: the ASSISTments site warns that some published
copies of this dataset contain duplicated records, and a corrected version
exists. Published papers cite ~278K-347K records; if our row count is far
above that, we may have the uncorrected copy. Duplicates cause evaluation
leakage (the same attempt can appear in both history and prediction target),
which would inflate AUC.
"""

import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "skill_builder_data.csv"


def explore():
    df = pd.read_csv(DATA, encoding="ISO-8859-15", low_memory=False)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    print("Columns:")
    for c in df.columns:
        print(f"  {c}")

    key_cols = [c for c in ["user_id", "skill_id", "skill_name", "correct", "order_id"] if c in df.columns]
    print("\nMissing values in key columns:")
    for c in key_cols:
        print(f"  {c}: {df[c].isna().mean():.1%} missing")

    if "correct" in df.columns:
        print(f"\n'correct' value counts:\n{df['correct'].value_counts(dropna=False)}")
    if "user_id" in df.columns:
        print(f"\nUnique students: {df['user_id'].nunique():,}")
    if "skill_id" in df.columns:
        print(f"Unique skills: {df['skill_id'].nunique():,}")

    print("\nFirst 3 rows (key columns):")
    print(df[key_cols].head(3).to_string())

    # ---- DUPLICATE DIAGNOSTIC ----
    print("\n" + "=" * 50)
    print("DUPLICATE CHECK")
    print("=" * 50)
    print(f"Our row count: {len(df):,}")
    print(f"Published ASSISTments 2009 counts: ~278,607 (original) / ~346,860 (corrected+collapsed)")

    full_dupes = df.duplicated().sum()
    print(f"\nFully identical rows (all 30 columns): {full_dupes:,} ({full_dupes/len(df):.1%})")

    if "order_id" in df.columns:
        order_dupes = df.duplicated(subset=["order_id"]).sum()
        print(f"Duplicate order_id (should be unique per attempt!): {order_dupes:,} ({order_dupes/len(df):.1%})")
        print(f"Unique order_ids: {df['order_id'].nunique():,} out of {len(df):,} rows")

    if all(c in df.columns for c in ["user_id", "problem_id", "order_id"]):
        triple_dupes = df.duplicated(subset=["user_id", "problem_id", "order_id"]).sum()
        print(f"Duplicate (user, problem, order): {triple_dupes:,} ({triple_dupes/len(df):.1%})")


if __name__ == "__main__":
    explore()