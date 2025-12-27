import pandas as pd
import glob
import numpy as np
import os

OUT_DIR = "D:/HCC/TumorDetection/patient_tumor_out"
# OUT_DIR = "D:/HCC/TumorDetection/patient_tumor_out_C1"


# Collect all per-fold eval files
csv_files = sorted(glob.glob(os.path.join(OUT_DIR, "eval_f*.csv")))
print("[info] found:", csv_files)

dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    df["fold"] = int(f.split("eval_f")[1].split(".")[0])  # add fold column
    dfs.append(df)

# Merge into one big OOF dataframe
oof = pd.concat(dfs, ignore_index=True)

oof_path = os.path.join(OUT_DIR, "oof_metrics.csv")
oof.to_csv(oof_path, index=False)

print(f"[info] saved merged OOF metrics to: {oof_path}")

# Compute mean ± std metrics
metrics = ["dice", "iou", "precision", "recall", "specificity", "hd95", "msd"]

print("\n=== FINAL OUT-OF-FOLD METRICS ===")
for m in metrics:
    vals = oof[m].dropna().values
    if len(vals) == 0:
        print(f"{m:12s}: N/A")
        continue
    print(f"{m:12s}: {vals.mean():.4f} ± {vals.std():.4f}")
