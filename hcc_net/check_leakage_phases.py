# check_leakage_phases.py

import pandas as pd
from config import CFG

def main():
    # 1) Load the same CSV your training uses
    df = pd.read_csv(CFG.CSV_PATH)
    print(f"Loaded CSV with {len(df)} patients")
    print("Labels distribution:")
    print(df["label"].value_counts())
    print()

    phases = ["C1", "C2", "C3", "P"]

    # 2) Normalize "missing" to a single condition: NaN or empty string
    for ph in phases:
        col = f"path_{ph}"
        if col not in df.columns:
            print(f"[WARN] Column {col} not in CSV, skipping.")
            continue

        # Build a boolean "is missing" column
        missing = df[col].isna() | (df[col].astype(str).str.strip() == "")
        df[f"missing_{ph}"] = missing

    # 3) Count missing phases per class
    print("=== Missing phase counts per class ===")
    for ph in phases:
        miss_col = f"missing_{ph}"
        if miss_col not in df.columns:
            continue
        print(f"\nPhase {ph}:")
        # Count how many patients of each label are missing this phase
        counts = df.groupby("label")[miss_col].sum()
        totals = df.groupby("label")[miss_col].count()
        frac = counts / totals
        print("Missing counts:")
        print(counts)
        print("Missing fraction:")
        print(frac.round(3))

    # 4) Optional: show a few rows where any phase is missing
    any_missing = df[[f"missing_{ph}" for ph in phases if f"missing_{ph}" in df.columns]].any(axis=1)
    df_missing = df[any_missing]
    print("\n=== Example rows with at least one missing phase ===")
    print(df_missing[["patient_id", "label"] + [f"path_{ph}" for ph in phases if f"path_{ph}" in df.columns]].head(10))

if __name__ == "__main__":
    main()
