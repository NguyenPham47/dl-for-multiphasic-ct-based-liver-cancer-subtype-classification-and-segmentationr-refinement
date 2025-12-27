import nibabel as nib
import numpy as np
import os
import sys

# ---------------------------
# CONFIGURE HERE
# ---------------------------
SLICES_PER_PHASE = 13   # or 13, or any odd number 2k+1
NIFTI_PATH = r"D:\HCC\CECT\ct_files\P0233_ct_C1.nii.gz"  # path to a sample NIfTI file
# ---------------------------


def load_nifti(path):
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    try:
        vol = nib.load(path).get_fdata()
    except Exception as e:
        print(f"[ERROR] Failed to load NIfTI: {e}")
        sys.exit(1)

    vol = np.asarray(vol)
    vol = np.squeeze(vol)

    # force final shape to (H, W, D)
    while vol.ndim > 3:
        vol = vol[..., 0]
    if vol.ndim == 2:
        vol = vol[..., None]

    if vol.ndim != 3:
        print(f"[ERROR] Unexpected shape: {vol.shape}")
        sys.exit(1)

    return vol   # (H, W, D)


def sanity_check(vol, slices_per_phase):
    H, W, D = vol.shape
    k = slices_per_phase // 2
    mid = D // 2

    idxs = np.arange(mid - k, mid + k + 1)
    idxs = np.clip(idxs, 0, D - 1)

    print("\n===============================")
    print("SANITY CHECK FOR 2.5D SLICES")
    print("===============================")
    print(f"NIfTI shape      : {vol.shape}  (H, W, D)")
    print(f"Depth (D)        : {D}")
    print(f"SLICES_PER_PHASE : {slices_per_phase}  (2k+1)")
    print(f"k                : {k}")
    print(f"Middle slice     : {mid}")
    print(f"Selected indices : {idxs.tolist()}")
    print(f"Total slices     : {len(idxs)}")
    print("===============================")

    # OPTIONAL: show values at those slices
    # Uncomment if you want:
    # for i in idxs:
    #     sl = vol[..., i]
    #     print(f"Slice {i}: min={sl.min():.2f}, max={sl.max():.2f}, mean={sl.mean():.2f}")


if __name__ == "__main__":
    vol = load_nifti(NIFTI_PATH)
    sanity_check(vol, SLICES_PER_PHASE)
