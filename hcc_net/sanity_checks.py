# sanity_checks.py

import os
import numpy as np
import pandas as pd
import nibabel as nib

from config import CFG
from dataset import _align_ct_and_mask, _to_2d, _resolve_path


def _load_nifti(path_or_rel, is_mask=False):
    if not isinstance(path_or_rel, str) or not path_or_rel:
        return None

    if is_mask:
        fp = path_or_rel if os.path.isabs(path_or_rel) else os.path.join(CFG.IMG_ROOT, "CECT", path_or_rel)
    else:
        fp = _resolve_path(path_or_rel)

    if not os.path.exists(fp):
        return None

    return np.squeeze(nib.load(fp).get_fdata())


def compute_expected_slice(mask):
    """
    Implements the NEW slice-selection logic:
      - compute tumor area per slice
      - find slices >= 50% of peak
      - expected slice = median of valid slices
    """
    mask = np.asarray(mask).squeeze()

    if mask.ndim != 3:
        return None

    H, W, D = mask.shape
    mask_bin = (mask > 0).astype(np.uint8)
    slice_sums = mask_bin.reshape(-1, D).sum(axis=0)

    if slice_sums.sum() == 0:
        return None  # no tumor → fallback to mid-slice

    peak = slice_sums.max()
    thresh = 0.5 * peak

    valid = np.where(slice_sums >= thresh)[0]
    if len(valid) == 0:
        return None

    return int(valid[len(valid)//2])


def sanity_check():
    df = pd.read_csv(CFG.CSV_PATH)

    ret_ratios = []
    slice_matches = 0
    slice_checked = 0

    for _, row in df.iterrows():

        lesion_rel = row.get("lesion_mask_path", "")
        if not lesion_rel or not isinstance(lesion_rel, str):
            continue

        # find any CT phase that is NIfTI
        ct_rel = None
        for ph in CFG.PHASES:
            rel = row.get(f"path_{ph}", "")
            if isinstance(rel, str) and rel.lower().endswith((".nii",".nii.gz")):
                ct_rel = rel
                break

        if not ct_rel:
            continue

        ct = _load_nifti(ct_rel, is_mask=False)
        mask = _load_nifti(lesion_rel, is_mask=True)
        if ct is None or mask is None:
            continue

        # ----- ALIGN -----
        try:
            ct_aligned, mask_aligned = _align_ct_and_mask(ct, mask)
        except:
            continue

        # ----- RETENTION CHECK -----
        mask_bin = (mask > 0).astype(np.uint8)
        before = mask_bin.sum()
        if before > 0:
            mask_aligned_bin = (mask_aligned > 0).astype(np.uint8)
            after = mask_aligned_bin.sum()
            ratio = after / float(before)
            ret_ratios.append(ratio)

        # ----- SLICE SELECTION CHECK -----
        expected_slice = compute_expected_slice(mask_aligned)

        # compute actual slice from _to_2d
        ct_used = _to_2d(ct, lesion_mask=mask)  # re-run whole pipeline

        # Find which aligned slice matches ct_used
        found_slice = None
        for z in range(ct_aligned.shape[2]):
            if np.allclose(ct_used, ct_aligned[..., z]):
                found_slice = z
                break

        if expected_slice is None:
            expected_slice = ct_aligned.shape[2] // 2  # mid-slice fallback

        if found_slice == expected_slice:
            slice_matches += 1

        slice_checked += 1

    # ----- PRINT RESULTS -----
    print("\n=== Tumor voxel retention after alignment ===")
    if ret_ratios:
        print(f"Mean ratio: {np.mean(ret_ratios):.4f}")
        print(f"Min ratio:  {np.min(ret_ratios):.4f}")
        print(f"Max ratio:  {np.max(ret_ratios):.4f}")
    else:
        print("No retention cases found.")

    print("\n=== Slice selection sanity (50% peak median) ===")
    print(f"Correct: {slice_matches}")
    print(f"Checked: {slice_checked}")


if __name__ == "__main__":
    sanity_check()
