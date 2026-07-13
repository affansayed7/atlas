"""First look at the ASSISTments dataset — shape, columns, missing values, sample."""

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


if __name__ == "__main__":
    explore()