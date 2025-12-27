import os
import pandas as pd
from config import CFG

def _prefer_uncompressed(path):
    """Prefer .nii if both .nii and .nii.gz exist."""
    if not isinstance(path, str) or not path:
        return path
    p = path.strip()
    # if p.lower().endswith(".nii.gz"):
    #     unzipped = p[:-3]  # remove '.gz'
    #     if os.path.exists(unzipped):
    #         return unzipped
    return p

def build_patient_df(phase_csv=CFG.PHASE_CSV, out_csv=CFG.CSV_PATH):
    df = pd.read_csv(phase_csv)

    # Expected input columns per-row (phase-wise)
    req = ["patient_id", "phase", "cancer_type", "ct_path", "mask_path"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # keep only our classes
    df = df[df["cancer_type"].isin(CFG.CLASSES)].copy()

    # Prefer .nii if available
    df["ct_path"] = df["ct_path"].map(_prefer_uncompressed)
    df["mask_path"] = df["mask_path"].map(_prefer_uncompressed)

    # one label per patient (assumes consistency)
    labels = df.groupby("patient_id")["cancer_type"].agg(lambda x: x.iloc[0])

    # --- pivot CT paths per phase
    ct_wide = df.pivot_table(
        index="patient_id",
        columns="phase",
        values="ct_path",
        aggfunc="first"
    )
    ct_wide = ct_wide.reindex(columns=CFG.PHASES)

    # --- pivot MASK paths per phase
    mask_col = "mask_path"
    mask_wide = df.pivot_table(
        index="patient_id",
        columns="phase",
        values=mask_col,
        aggfunc="first"
    )
    mask_wide = mask_wide.reindex(columns=CFG.PHASES)

    # build output
    out = ct_wide.copy()
    out.columns = [f"path_{c}" for c in out.columns]

    mask_out = mask_wide.copy()
    mask_out.columns = [f"path_mask_{c}" for c in mask_out.columns]

    out = pd.concat([out, mask_out], axis=1)
    out["label"] = labels

    # Optional demographics
    for col in ["age", "gender"]:
        if col in df.columns:
            out[col] = df.groupby("patient_id")[col].agg(lambda x: x.iloc[0])

    out = out.reset_index()

    # Keep only desired columns
    keep_cols = [
        "patient_id",
        "path_C1", "path_C2", "path_C3",
        "path_mask_C1", "path_mask_C2", "path_mask_C3",
        "label",
    ]
    keep_cols += [c for c in ["age", "gender"] if c in out.columns]
    out = out[[c for c in keep_cols if c in out.columns]]

    out.to_csv(out_csv, index=False)
    print(f"Saved patient-level CSV: {out_csv}  (rows={len(out)})")

if __name__ == "__main__":
    build_patient_df()
