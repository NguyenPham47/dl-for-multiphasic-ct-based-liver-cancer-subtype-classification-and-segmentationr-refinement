import os
import pandas as pd
import numpy as np
import nibabel as nib
from config import CFG

CSV_PATH = r"D:\HCC\patient_rows_with_lesion.csv"
MASK_COL = "mask_path"
DATA_ROOT = r"D:\HCC"
df = pd.read_csv(CSV_PATH)

total_with_mask = 0
lesion_anywhere = 0
middle_has_lesion = 0

for _, row in df.iterrows():
    rel = row.get(MASK_COL, "")
    if not isinstance(rel, str) or len(rel) == 0:
        continue

    # resolve path similar to _resolve_path in your dataset.py
    if os.path.isabs(rel):
        mask_path = rel
    else:
        # dataset seems to live under IMG_ROOT/CECT/...
        mask_path = os.path.join(DATA_ROOT, "CECT", rel)

    if not os.path.exists(mask_path):
        print(f"[WARN] Missing mask: {mask_path}")
        continue

    total_with_mask += 1

    mask = nib.load(mask_path).get_fdata()
    mask = mask > 0

    if not mask.any():
        continue

    lesion_anywhere += 1

    H, W, D = mask.shape
    mid = D // 2

    if mask[:, :, mid].any():
        middle_has_lesion += 1

print("\n=== Middle-slice lesion stats ===")
print(f"Patients with a lesion mask file:         {total_with_mask}")
print(f"Patients with lesion voxels somewhere:    {lesion_anywhere}")
print(f"Patients with lesion on middle slice:     {middle_has_lesion}")

if lesion_anywhere > 0:
    frac = middle_has_lesion / lesion_anywhere
    print(f"\n=> Middle slice contains lesion in {frac*100:.2f}% of lesion cases.")
