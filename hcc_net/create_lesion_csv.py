import pandas as pd
from config import CFG

# Input CSVs
phase_df = pd.read_csv(CFG.PHASE_CSV)      # original dataset CSV (with mask_path)
patient_df = pd.read_csv(CFG.CSV_PATH)     # your stable patient_rows.csv

# Priority order to pick one mask per patient
prio = ["P", "C3", "C2", "C1"]
phase_rank = {p:i for i,p in enumerate(prio)}

# Extract ONE lesion mask_path per patient
lesion_mask_choices = (
    phase_df.sort_values(by=["patient_id"], kind="stable")
            .assign(phase_order=phase_df["phase"].map(phase_rank).fillna(99))
            .sort_values(["patient_id","phase_order"])
            .groupby("patient_id")["mask_path"]
            .agg(lambda x: next((v for v in x if isinstance(v,str) and len(v)>0), ""))
            .reset_index()
)

# Merge into new patient-level CSV
out = patient_df.merge(lesion_mask_choices, on="patient_id", how="left")

# Save NEW csv (does not overwrite your original)
OUT_PATH = r"D:\HCC\patient_rows_with_lesion.csv"
out.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print("Rows:", len(out))
print(out.head())
