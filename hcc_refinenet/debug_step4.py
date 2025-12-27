import argparse, os, numpy as np, pandas as pd, nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    EnsureTyped, SqueezeDimd, ResampleToMatchd, MapTransform
)
from config import CFG

# ---------- path resolve identical to shape_details.py ----------
def _resolve_like_shape_details(p: str):
    if p is None:
        return None
    p = str(p)
    return os.path.normpath(p) if os.path.isabs(p) else os.path.normpath(os.path.join(CFG.IMG_ROOT, p))

# ---------- Step 1: image shape fix ----------
class FixImageShaped(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            a = d.get(k, None)
            if a is None:
                continue
            a = np.asarray(a)
            a = np.squeeze(a)
            if a.ndim > 3:
                if a.ndim == 4 and a.shape[-1] == 3:
                    a = a.mean(axis=-1)
                else:
                    raise ValueError(f"[FixImageShaped] Unexpected image shape after squeeze: {a.shape}")
            if a.ndim != 3:
                raise ValueError(f"[FixImageShaped] Image must be 3-D, got {a.shape}")
            d[k] = a.astype(np.float32)
        return d
    
class FixMaskShaped(MapTransform):
    """
    Squeeze singleton dims; ensure masks are binary uint8.
    Does NOT try to average multi-channel masks; raises if >3D after squeeze.
    """
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            if k not in d or d[k] is None:
                continue
            a = np.asarray(d[k])
            a = np.squeeze(a)
            if a.ndim != 3:
                raise ValueError(f"[FixMaskShaped] Mask '{k}' not 3-D after squeeze: {a.shape}")
            d[k] = (a > 0).astype(np.uint8)
        return d

class UnionMasksd(MapTransform):
    """
    Union any present masks in `mask_keys` into 'mask_union' (H,W,D).
    If none present, creates zeros matching image spatial shape.
    """
    def __init__(self, img_key="image", mask_keys=("mask_C1","mask_C2","mask_C3")):
        super().__init__(keys=[img_key] + list(mask_keys))
        self.img_key = img_key
        self.mask_keys = list(mask_keys)

    def __call__(self, data):
        d = dict(data)
        # image is (1,H,W,D) now
        if self.img_key not in d:
            raise ValueError("image key missing before UnionMasksd")
        H, W, D = d[self.img_key].shape[-3:]
        acc = None
        present = 0
        for k in self.mask_keys:
            m = d.get(k, None)
            if m is None:
                continue
            m = np.asarray(m)
            if m.shape != (H, W, D):
                raise ValueError(f"[UnionMasksd] Mask '{k}' not aligned to image: {m.shape} vs {(H,W,D)}")
            acc = m if acc is None else np.clip(acc + m, 0, 1)
            present += 1
        if acc is None:
            acc = np.zeros((H, W, D), dtype=np.uint8)
            d["_mask_info"] = "no_masks"
        else:
            d["_mask_info"] = f"{present} masks unioned"
        d["mask_union"] = acc.astype(np.uint8)
        return d

# ---------- Step 2: HU -> [0,1] ----------
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

        a = a.astype(np.float32)
        if looks_hu or not (looks_normalized or looks_8bit):
            a = np.clip(a, self.hu_min, self.hu_max)
            a = (a - self.hu_min) / (self.hu_max - self.hu_min + 1e-8)
            return a
        if looks_normalized:
            return np.clip(a, 0.0, 1.0)
        a = (a - vmin) / max(1.0, (vmax - vmin))
        return np.clip(a, 0.0, 1.0)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            a = d.get(k, None)
            if a is not None:
                d[k] = self._normalize(a)
        return d

# ---------- Step 4: load + union masks ----------
class LoadMasksAndUniond(MapTransform):
    def __init__(self, img_key="image", mask_keys=("path_mask_C1","path_mask_C2","path_mask_C3")):
        super().__init__(keys=[img_key] + list(mask_keys))
        self.img_key = img_key
        self.mask_keys = list(mask_keys)

    def _load_mask(self, p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return None
        if not os.path.exists(p):
            return None
        a = np.asarray(nib.load(p).get_fdata())
        a = np.squeeze(a)
        if a.ndim != 3:
            raise ValueError(f"Mask at {p} is not 3-D after squeeze (shape={a.shape}).")
        a = (a > 0).astype(np.uint8)
        return a

    def __call__(self, data):
        d = dict(data)
        img = d[self.img_key]
        img_sz = tuple(img.shape[-3:]) if hasattr(img, "shape") else None

        masks = []
        for k in self.mask_keys:
            p = d.get(k, None)
            m = self._load_mask(p)
            if m is not None:
                masks.append(m)

        if len(masks) == 0:
            if img_sz is None:
                raise ValueError("Image size unknown; cannot create empty mask.")
            union = np.zeros(img_sz, dtype=np.uint8)
            d["_mask_info"] = "no_masks"
        else:
            base = masks[0].shape
            for mi in masks[1:]:
                if mi.shape != base:
                    raise ValueError(f"Mask shapes differ: {base} vs {mi.shape}. Fix upstream.")
            union = np.clip(np.sum(masks, axis=0), 0, 1).astype(np.uint8)
            d["_mask_info"] = f"{len(masks)} mask(s) unioned"
        d["mask_union"] = union  # (Z,Y,X)
        return d

MASK_KEYS = ["mask_C1", "mask_C2", "mask_C3"]

def build_pipeline(cfg: CFG):
    return Compose([
        # Load with meta
        LoadImaged(keys=["image"] + MASK_KEYS, allow_missing_keys=True, image_only=False),

        # Fix shapes / intensities
        FixImageShaped(keys=["image"]),                                # -> (Z,Y,X)
        FixMaskShaped(keys=MASK_KEYS),                                 # -> (Z,Y,X), binary
        HUWindowd(keys=["image"], hu_min=cfg.HU_MIN, hu_max=cfg.HU_MAX),

        # Channel-first for BOTH image and masks
        EnsureChannelFirstd(keys=["image"] + MASK_KEYS, channel_dim="no_channel"),

        # Make MetaTensor BEFORE spatial ops
        EnsureTyped(
            keys=["image"] + MASK_KEYS,
            dtype=[np.float32] + [np.uint8]*len(MASK_KEYS),
            track_meta=True,
            allow_missing_keys=True,
        ),

        # Orientation to RAS
        Orientationd(keys=["image"] + MASK_KEYS, axcodes="RAS", allow_missing_keys=True),

        # Space the IMAGE to target spacing
        Spacingd(keys=["image"], pixdim=cfg.SPACING, mode="bilinear", padding_mode="border"),

        # ⚠️ Instead of Spacingd on masks, resample them to MATCH the image grid
        ResampleToMatchd(
            keys=MASK_KEYS, key_dst="image", mode="nearest", padding_mode="border",
            allow_missing_keys=True
        ),

        # Squeeze back mask channel dim to (H,W,D)
        SqueezeDimd(keys=MASK_KEYS, dim=0, allow_missing_keys=True),

        # Union AFTER alignment
        UnionMasksd(img_key="image", mask_keys=MASK_KEYS),
    ])

def spacing_from_affine(affine: np.ndarray):
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))

def pick_cases_with_masks(df: pd.DataFrame, n=4):
    out = []
    for _, r in df.iterrows():
        p_img = _resolve_like_shape_details(r.get("path_C1"))
        if p_img is None or not os.path.exists(p_img):
            continue
        has_any_mask = False
        mask_paths = []
        for mk in ["path_mask_C1","path_mask_C2","path_mask_C3"]:
            p = _resolve_like_shape_details(r.get(mk))
            mask_paths.append(p if (p and os.path.exists(p)) else None)
            has_any_mask = has_any_mask or (mask_paths[-1] is not None)
        if has_any_mask:
            out.append((r["patient_id"], p_img, tuple(mask_paths)))
        if len(out) >= n:
            break
    return out

def run_debug(cfg: CFG, pids=None, n_cases=4):
    df = pd.read_csv(cfg.CSV_PATH)
    if pids:
        rows = df[df["patient_id"].isin(pids)]
        if rows.empty:
            raise ValueError(f"No rows match patient_ids={pids}")
        cases = []
        for _, r in rows.iterrows():
            p_img = _resolve_like_shape_details(r.get("path_C1"))
            if p_img is None or not os.path.exists(p_img):
                continue
            mask_paths = []
            for mk in ["path_mask_C1","path_mask_C2","path_mask_C3"]:
                p = _resolve_like_shape_details(r.get(mk))
                mask_paths.append(p if (p and os.path.exists(p)) else None)
            cases.append((r["patient_id"], p_img, tuple(mask_paths)))
        if not cases:
            print("No valid image+mask cases found for the provided PIDs.")
            return
    else:
        cases = pick_cases_with_masks(df, n=n_cases)
        if not cases:
            print("Couldn’t find cases with masks. Try --pids for known mask patients.")
            return

    xform = build_pipeline(cfg)

    print(f"Target spacing: {cfg.SPACING}")
    print("-" * 70)
    for pid, p_img, mask_paths in cases:
        d = {"image": p_img}

        # only add mask keys that actually exist on disk
        mk_map = {
            "mask_C1": mask_paths[0],
            "mask_C2": mask_paths[1],
            "mask_C3": mask_paths[2],
        }
        for k, p in mk_map.items():
            if p is not None and os.path.exists(p):
                d[k] = p
        out = xform(d)
        img = out["image"]            # (1,H,W,D)
        msk = out["mask_union"]       # (H,W,D) after spacing
        affine = np.array(img.meta.get("affine", np.eye(4)))
        spacing_vec = spacing_from_affine(affine)

        # checks
        img_shape = tuple(img.shape)
        msk_shape = tuple(msk.shape)
        aligned = img_shape[-3:] == msk_shape
        voxels = int(msk.sum())
        uniq = np.unique(msk).tolist()
        bbox = None
        if voxels > 0:
            pos = np.argwhere(msk > 0)
            zmin,zmax = pos[:,2].min(), pos[:,2].max()
            ymin,ymax = pos[:,0].min(), pos[:,0].max()
            xmin,xmax = pos[:,1].min(), pos[:,1].max()
            bbox = (int(ymin), int(ymax), int(xmin), int(xmax), int(zmin), int(zmax))

        print(f"PID: {pid}")
        print(f"  image path: {p_img}")
        print(f"  masks: {[os.path.basename(p) if p else None for p in mask_paths]}")
        print(f"  final image shape: {img_shape}  (expect (1,H,W,D))")
        print(f"  final mask shape : {msk_shape}  (expect (H,W,D))  aligned={aligned}")
        print(f"  spacing from affine: {tuple(np.round(spacing_vec,3))}  -> "
              f"{'OK' if np.allclose(spacing_vec, np.array(CFG.SPACING), atol=1e-3, rtol=1e-3) else 'MISMATCH'}")
        print(f"  mask unique values: {uniq}  | voxels: {voxels}")
        print(f"  mask bbox (ymin,ymax,xmin,xmax,zmin,zmax): {bbox}")
        print("-" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pids", nargs="*", help="Optional list of patient_ids to test.")
    parser.add_argument("--n", type=int, default=4, help="Auto-pick N cases with masks if PIDs not provided.")
    args = parser.parse_args()
    run_debug(CFG(), pids=args.pids, n_cases=args.n)
