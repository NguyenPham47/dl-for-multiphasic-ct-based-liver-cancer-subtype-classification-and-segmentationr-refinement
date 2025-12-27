import os
import numpy as np
import torch
import pandas as pd

from monai.data import CacheDataset, DataLoader
from monai.data.utils import list_data_collate
from monai.utils import set_determinism

from config import CFG
from dataset import df_to_items, _resolve_path
from transforms import make_train_transforms, make_val_transforms
from utils import seed_all

# --- helpers ---

def spacing_from_affine(aff):
    """
    Accepts:
      - (4,4) affine
      - (C,4,4) per-channel affines (take the first)
      - torch/numpy/list/tuple variants
    Returns (sx, sy, sz).
    """
    # unwrap lists/tuples of affines (e.g., batched metas)
    if isinstance(aff, (list, tuple)):
        aff = aff[0]
    # torch -> numpy
    if hasattr(aff, "detach"):
        aff = aff.detach().cpu().numpy()
    aff = np.asarray(aff)
    # per-channel affine tensor -> first channel
    if aff.ndim == 3 and aff.shape[-2:] == (4, 4):
        aff = aff[0]
    assert aff.shape == (4, 4), f"unexpected affine shape {aff.shape}"
    return np.sqrt((aff[:3, :3] ** 2).sum(axis=0))  # (sx,sy,sz)


def assert_between(name, arr, lo, hi):
    m = float(torch.min(arr).item() if torch.is_tensor(arr) else np.min(arr))
    M = float(torch.max(arr).item() if torch.is_tensor(arr) else np.max(arr))
    if not (m >= lo - 1e-5 and M <= hi + 1e-5):
        raise AssertionError(f"{name} values out of range [{lo},{hi}]: min={m}, max={M}")

def assert_binary(name, arr):
    uniq = torch.unique(arr) if torch.is_tensor(arr) else np.unique(arr)
    bad = [u for u in uniq.tolist() if u not in (0,1)]
    if bad:
        raise AssertionError(f"{name} not binary; unique values include {bad}")

def assert_equal_tuple(a, b, msg):
    if tuple(a) != tuple(b):
        raise AssertionError(f"{msg}: got {a} vs {b}")

def assert_close_vec(a, b, atol=1e-3, rtol=1e-3, msg="vectors not close"):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"{msg}: {a} vs {b}")

def print_ok(label):
    print(f"[OK] {label}")

# --- tests ---

def test_val_pipeline(cfg: CFG, n=4):
    seed_all(cfg.SEED); set_determinism(cfg.SEED)
    df = pd.read_csv(cfg.CSV_PATH).head(max(8, n))
    items = df_to_items(df)

    # val transforms: full-volumes
    val_tf = make_val_transforms(cfg)
    ds = CacheDataset(items, transform=val_tf, cache_rate=0.0, num_workers=0)
    dl = DataLoader(ds, batch_size=1, collate_fn=list_data_collate, num_workers=0, pin_memory=False)

    c = 0
    for batch in dl:
        img = batch["img"]           # (B,6,Z,Y,X)
        lab = batch["lab"]           # (B,Z,Y,X) uint8
        meta = img.meta if hasattr(img, "meta") else img
        if isinstance(meta, (list, tuple)):
            meta = meta[0]

        # shapes
        assert len(img.shape) == 5, "img must be (B,6,Z,Y,X)"
        assert img.shape[1] == 6, "img must have 6 channels"
        assert len(lab.shape) == 4, "lab must be (B,Z,Y,X)"
        assert_equal_tuple(img.shape[-3:], lab.shape[-3:], "img/lab spatial mismatch")

        # dtypes / values
        assert img.dtype == torch.float32, "img dtype must be float32"
        assert lab.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64), "lab must be integer type"
        assert_between("img", img, 0.0, 1.0)
        assert_binary("lab", lab)

        # spacing from affine
        aff = meta["affine"]
        sp = spacing_from_affine(aff)
        assert_close_vec(sp, cfg.SPACING, msg="val spacing mismatch")
        c += 1
        if c >= n:
            break
    print_ok("val pipeline basic")

def test_train_pipeline(cfg: CFG, n=4):
    seed_all(cfg.SEED); set_determinism(cfg.SEED)
    df = pd.read_csv(cfg.CSV_PATH).head(max(8, n))
    items = df_to_items(df)

    train_tf = make_train_transforms(cfg)
    ds = CacheDataset(items, transform=train_tf, cache_rate=0.0, num_workers=0)
    dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, collate_fn=list_data_collate, num_workers=0, pin_memory=False)

    batch = next(iter(dl))
    img = batch["img"]            # (B,6,rz,ry,rx)
    lab = batch["lab"]            # (B,rz,ry,rx)
    meta = img.meta if hasattr(img, "meta") else img
    if isinstance(meta, (list, tuple)):
        meta = meta[0]

    # ROI size exact
    assert_equal_tuple(img.shape[-3:], cfg.ROI_SIZE, "train ROI size wrong for img")
    assert_equal_tuple(lab.shape[-3:], cfg.ROI_SIZE, "train ROI size wrong for lab")
    assert img.shape[1] == 6, "img must have 6 channels"

    # dtypes / values
    assert img.dtype == torch.float32, "img dtype must be float32"
    assert_between("img", img, 0.0, 1.0)
    assert_binary("lab", lab)

    print_ok("train pipeline basic")

def test_determinism_val(cfg: CFG, n=2):
    # run twice and verify same shapes + spacing
    seed_all(cfg.SEED); set_determinism(cfg.SEED)
    df = pd.read_csv(cfg.CSV_PATH).head(max(4, n))
    items = df_to_items(df)

    tf = make_val_transforms(cfg)

    outs = []
    for _ in range(2):
        ds = CacheDataset(items, transform=tf, cache_rate=0.0, num_workers=0)
        dl = DataLoader(ds, batch_size=1, collate_fn=list_data_collate, num_workers=0, pin_memory=False)
        shapes = []
        spacings = []
        for batch in dl:
            img = batch["img"]
            shapes.append(tuple(img.shape))
            spacings.append(tuple(spacing_from_affine(img.meta["affine"])))
        outs.append((shapes, spacings))

    assert outs[0] == outs[1], "val pipeline not deterministic under fixed seed (shapes/spacings differ)"
    print_ok("val determinism (basic)")

if __name__ == "__main__":
    cfg = CFG()
    test_val_pipeline(cfg, n=4)
    test_train_pipeline(cfg, n=4)
    test_determinism_val(cfg, n=2)
    print_ok("ALL TRANSFORM TESTS PASSED")
