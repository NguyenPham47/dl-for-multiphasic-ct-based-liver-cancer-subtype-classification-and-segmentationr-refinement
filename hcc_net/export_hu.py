# first_patient_export.py
import os
import cv2
import numpy as np
import pandas as pd

from config import CFG
from dataset import _read_gray, _hu_window, _resize_to_match, _to_2d, _tight_bbox, _crop

def _maybe_crop_with_liver_2d(img2d: np.ndarray, mask_path: str) -> np.ndarray:
    if not (CFG.CROP_TO_LIVER and mask_path and isinstance(mask_path, str)):
        return img2d
    m = _read_gray(mask_path)
    m = (m > 0).astype(np.uint8)
    bbox = _tight_bbox(m, pad=8)
    return img2d if bbox is None else _crop(img2d, bbox)

def window_phase_image(img, center=None, width=None, size=None, mask_path: str = ""):
    center = CFG.WINDOW_CENTER if center is None else center
    width  = CFG.WINDOW_WIDTH  if width  is None else width
    size   = CFG.IMG_SIZE      if size   is None else size

    img2d = _to_2d(img).astype(np.float32)
    # img2d = _maybe_crop_with_liver_2d(img2d, mask_path)
    img2d = _resize_to_match(img2d, target_shape=(size, size))
    out = _hu_window(img2d, center=center, width=width)
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def save_first_patient_hu(csv_path: str = CFG.CSV_PATH,
                          out_dir: str = "hu_windowed_first",
                          center: int = None,
                          width: int = None,
                          size: int = None,
                          use_mask_if_available: bool = True,
                          png_quality: int = 95):
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError("CSV is empty.")

    row = df.iloc[0]
    pid = str(row.get("patient_id", "row0"))
    os.makedirs(out_dir, exist_ok=True)
    pid_dir = os.path.join(out_dir, pid)
    os.makedirs(pid_dir, exist_ok=True)

    mask_path = row.get("liver_mask_path", "")
    if not use_mask_if_available:
        mask_path = ""

    saved = []
    for ph in CFG.PHASES:
        col = f"path_{ph}"
        p = row.get(col, "")
        if not isinstance(p, str) or not p:
            continue

        raw = _read_gray(p)
        wnd = window_phase_image(raw, center=center, width=width, size=size, mask_path=mask_path)
        im8 = np.round(wnd * 255.0).astype(np.uint8)

        out_path = os.path.join(pid_dir, f"{ph}.png")
        png_compression = max(0, min(9, int((100 - png_quality) / 10)))
        cv2.imwrite(out_path, im8, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
        saved.append(out_path)

    print(f"Saved {len(saved)} images to: {pid_dir}")
    return saved

if __name__ == "__main__":
    save_first_patient_hu(
        csv_path=CFG.CSV_PATH,
        out_dir=os.path.join(os.path.dirname(CFG.CSV_PATH), "hu_first"),
        center=CFG.WINDOW_CENTER,
        width=CFG.WINDOW_WIDTH,
        size=CFG.IMG_SIZE,
        use_mask_if_available=True,
        png_quality=95,
    )
