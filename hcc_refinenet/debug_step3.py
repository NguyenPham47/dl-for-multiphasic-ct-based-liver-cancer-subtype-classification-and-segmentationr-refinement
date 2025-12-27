# debug_step3.py
import argparse, os, numpy as np, pandas as pd, nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, EnsureTyped, MapTransform
)
from config import CFG
import os


def _resolve_like_shape_details(p: str):
    if p is None:
        return None
    p = str(p)
    # If already absolute, keep it; else join with IMG_ROOT (no extra 'CECT/' prefixing).
    return os.path.normpath(p) if os.path.isabs(p) else os.path.normpath(os.path.join(CFG.IMG_ROOT, p))

# --------- Step 1 (image only) ---------
class FixImageShaped(MapTransform):
    """
    Squeeze singleton dims; if trailing axis size==3 (RGB/vector artifact), collapse by mean.
    Ensures output is (Z, Y, X) float32.
    """
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            a = d.get(k, None)
            if a is None:
                continue
            a = np.asarray(a)
            a = np.squeeze(a)  # drop all singleton dims
            if a.ndim > 3:
                if a.ndim == 4 and a.shape[-1] == 3:
                    a = a.mean(axis=-1)
                else:
                    raise ValueError(f"[FixImageShaped] Unexpected image shape after squeeze: {a.shape}")
            if a.ndim != 3:
                raise ValueError(f"[FixImageShaped] Image must be 3-D, got {a.shape}")
            d[k] = a.astype(np.float32, copy=False)
        return d

# --------- Step 2 (HU -> [0,1]) ---------
class HUWindowd(MapTransform):
    def __init__(self, keys, hu_min=-100, hu_max=400):
        super().__init__(keys)
        self.hu_min = float(hu_min)
        self.hu_max = float(hu_max)

    def _normalize(self, a: np.ndarray) -> np.ndarray:
        vmin = float(np.percentile(a, 0.5))
        vmax = float(np.percentile(a, 99.5))
        looks_normalized = (vmin >= -0.2) and (vmax <= 1.5)
        looks_8bit = (vmin >= -10.0) and (vmax <= 300.0) and (vmax - vmin >= 50.0)
        looks_hu = (vmin < -200.0) or (vmax > 1000.0)

        a = a.astype(np.float32, copy=False)
        if looks_hu or not (looks_normalized or looks_8bit):
            a = np.clip(a, self.hu_min, self.hu_max)
            a = (a - self.hu_min) / (self.hu_max - self.hu_min + 1e-8)
            return a
        if looks_normalized:
            return np.clip(a, 0.0, 1.0)
        # looks_8bit
        a = (a - vmin) / max(1.0, (vmax - vmin))
        return np.clip(a, 0.0, 1.0)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            a = d.get(k, None)
            if a is not None:
                d[k] = self._normalize(a)
        return d

def build_partial_pipeline(cfg):
    return Compose([
        LoadImaged(keys=["image"]),                         # adds image_meta_dict
        FixImageShaped(keys=["image"]),                     # (Z,Y,X), numpy
        HUWindowd(keys=["image"], hu_min=cfg.HU_MIN, hu_max=cfg.HU_MAX),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),  # -> (1,Z,Y,X)
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=cfg.SPACING, mode="bilinear", padding_mode="border"),
        EnsureTyped(keys=["image"], dtype=np.float32, track_meta=True),
    ])

def pick_cases(df: pd.DataFrame, n_abnormal=2, n_normal=2):
    abnormal, normal = [], []
    for _, r in df.iterrows():
        p = _resolve_like_shape_details(r.get("path_C1"))
        if p is None or not os.path.exists(p):
            continue
        try:
            shp = nib.load(p).shape
        except Exception:
            continue
        # abnormal: extra dims or trailing 3
        if (len(shp) > 3) or (len(shp) == 4 and shp[-1] == 3):
            if len(abnormal) < n_abnormal:
                abnormal.append((r["patient_id"], p, shp))
        elif len(shp) == 3:
            if len(normal) < n_normal:
                normal.append((r["patient_id"], p, shp))
        if len(abnormal) >= n_abnormal and len(normal) >= n_normal:
            break
    return abnormal, normal

def run_debug(cfg: CFG, pids=None, n_abnormal=2, n_normal=2):
    df = pd.read_csv(cfg.CSV_PATH)
    if pids:
        rows = df[df["patient_id"].isin(pids)]
        if rows.empty:
            raise ValueError(f"No rows match patient_ids={pids}")
        cases = [(r["patient_id"], _resolve_like_shape_details(r["path_C1"]), None) for _, r in rows.iterrows()]
        if not cases:
            print("No test cases found. Check CSV paths and CFG.IMG_ROOT.")
            return
    else:
        abn, nor = pick_cases(df, n_abnormal, n_normal)
        cases = abn + nor

    xform = build_partial_pipeline(cfg)

    print(f"Target spacing: {cfg.SPACING}")
    print("-" * 70)
    for pid, path, shp in cases:
        d = {"image": path}
        out = xform(d)
        img = out["image"]  # MetaTensor
        arr_shape = tuple(img.shape)  # MONAI 3D is (C, H, W, D)
        affine = np.array(img.meta.get("affine", np.eye(4)))

        # derive voxel spacing from affine
        spacing_vec = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))  # (sx, sy, sz)
        ok_spacing = np.allclose(spacing_vec, np.array(CFG.SPACING), atol=1e-3, rtol=1e-3)

        print(f"PID: {pid}")
        if shp is not None:
            print(f"  raw shape: {shp}")
        print(f"  final shape: {arr_shape}  (expect (1, H, W, D) with D ~ Z)")
        print(f"  spacing from affine: {tuple(np.round(spacing_vec,3))}  -> {'OK' if ok_spacing else 'MISMATCH'}")
        print(f"  affine[0:2,0:2]:\n{affine[:2,:2]}")
        print('-' * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pids", nargs="*", help="Optional list of patient_ids to test, e.g., P0250 P0251")
    parser.add_argument("--n_abnormal", type=int, default=2)
    parser.add_argument("--n_normal", type=int, default=2)
    args = parser.parse_args()

    cfg = CFG()
    run_debug(cfg, pids=args.pids, n_abnormal=args.n_abnormal, n_normal=args.n_normal)
