# oof_eval.py
import os
import numpy as np
import pandas as pd
import json
from config import CFG
from evaluate import evaluate


def run_oof(thresh: float = 0.97):
    cfg = CFG()
    out_dir = os.path.join(cfg.OUT_DIR, f"thresh_{str(thresh).replace('.', '_')}")
    os.makedirs(out_dir, exist_ok=True)
    n_folds = cfg.N_FOLDS

    os.makedirs(out_dir, exist_ok=True)

    csv_paths = []

    # 1) Run per-fold evaluation (true OOF chunks)
    for f in range(n_folds):
        csv_name = f"eval_f{f}.csv"
        print(f"\n[OOF] Evaluating fold {f} with thresh={thresh} -> {csv_name}")
        evaluate(
            fold=f,
            ensemble_folds=None,
            thresh=thresh,
            save_csv_name=csv_name,
            out_dir=out_dir,
        )
        csv_paths.append(os.path.join(out_dir, csv_name))

    # 2) Merge eval_f*.csv into one OOF CSV
    dfs = []
    for f, p in enumerate(csv_paths):
        if not os.path.exists(p):
            print(f"[warn] missing CSV for fold {f}: {p}, skipping.")
            continue
        df = pd.read_csv(p)
        df["fold"] = f
        dfs.append(df)

    if not dfs:
        print("[error] no per-fold CSVs found, aborting.")
        return

    oof = pd.concat(dfs, ignore_index=True)

    oof_name = f"eval_oof_thresh{str(thresh).replace('.', '_')}.csv"
    oof_path = os.path.join(out_dir, oof_name)
    oof.to_csv(oof_path, index=False)

    print(f"\n[OOF] Saved merged OOF metrics to: {oof_path}")
    print(f"[OOF] Total cases: {len(oof)}")

    # 3) Compute final mean ± std for each metric
    metrics = ["dice", "iou", "precision", "recall", "specificity", "hd95", "msd"]
    oof_summary = {}
    print("\n=== FINAL OUT-OF-FOLD METRICS ===")
    for m in metrics:
        if m not in oof.columns:
            print(f"{m:12s}: N/A (column not found)")
            oof_summary[m] = {"mean": None, "std": None}
            continue
        vals = oof[m].dropna().values.astype(float)
        if len(vals) == 0:
            print(f"{m:12s}: N/A (no values)")
            oof_summary[m] = {"mean": None, "std": None}
            continue
        mean = float(vals.mean())
        std  = float(vals.std())
        print(f"{m:12s}: {mean:.4f} ± {std:.4f}")
        oof_summary[m] = {"mean": mean, "std": std}
    
    # Save OOF summary metrics to JSON
    json_name = f"eval_oof_thresh{str(thresh).replace('.', '_')}.json"
    json_path = os.path.join(out_dir, json_name)
    with open(json_path, "w") as f:
        json.dump(oof_summary, f, indent=2)
    print(f"\n[OOF] Saved merged OOF metrics JSON to: {json_path}")


if __name__ == "__main__":
    # change thresh here if you want (e.g. 0.5, 0.9, 0.97, 0.99)
    run_oof(thresh=0.65)
