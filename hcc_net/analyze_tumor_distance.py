import os
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# ---- CONFIG ----
CSV_PATH = r"D:\HCC\patient_rows_with_lesion.csv"
MASK_COL = "mask_path"
DATA_ROOT = r"D:\HCC"      # where "CECT" folder lives

df = pd.read_csv(CSV_PATH)

center_dists = []   # |center_of_tumor - mid_slice|
min_dists = []      # min |any_tumor_slice - mid_slice|
n_cases = 0

for _, row in df.iterrows():
    rel = row.get(MASK_COL, "")
    if not isinstance(rel, str) or not rel:
        continue

    # resolve path similar to your dataset
    if os.path.isabs(rel):
        mask_path = rel
    else:
        mask_path = os.path.join(DATA_ROOT, "CECT", rel)

    if not os.path.exists(mask_path):
        print(f"[WARN] Missing mask: {mask_path}")
        continue

    mask = nib.load(mask_path).get_fdata()
    mask = mask > 0

    if not mask.any():
        continue

    H, W, D = mask.shape
    mid = D // 2

    # all slices that contain tumor
    tumor_slices = np.where(mask.sum(axis=(0, 1)) > 0)[0]
    if len(tumor_slices) == 0:
        continue

    n_cases += 1

    # center of tumor in z
    center = tumor_slices.mean()
    center_dists.append(abs(center - mid))

    # closest tumor slice to the middle
    min_d = np.min(np.abs(tumor_slices - mid))
    min_dists.append(min_d)

center_dists = np.array(center_dists)
min_dists = np.array(min_dists)

print(f"\nUsed {n_cases} cases with valid lesion masks.")

def summarize(name, d):
    print(f"\n=== {name} distances (|slice - mid|) ===")
    print(f"Mean:   {d.mean():.2f} slices")
    print(f"Median: {np.median(d):.2f} slices")
    print(f"Std:    {d.std():.2f} slices")
    for k in [1, 2, 3, 5]:
        frac = (d <= k).mean() * 100
        print(f"<= {k} slices: {frac:.2f}% of cases")

summarize("Tumor CENTER", center_dists)
summarize("CLOSEST tumor slice", min_dists)

# ---- Histogram plot (for CENTER distance) ----
plt.figure(figsize=(6,4))
plt.hist(center_dists, bins=20)
plt.title("Distance between tumor CENTER and mid-slice")
plt.xlabel("|center_z - mid_z| (slices)")
plt.ylabel("Number of patients")
plt.tight_layout()
plt.show()
