# predict.py (ensemble version)
import os
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
import pandas as pd
import scipy.ndimage as ndi

from monai.inferers import sliding_window_inference
from monai.utils import InterpolateMode
from monai.transforms import (
    Compose, LoadImaged, CopyItemsd, DeleteItemsd, EnsureChannelFirstd,
    EnsureTyped, Orientationd, Spacingd, ResampleToMatchd, SqueezeDimd,
    ConcatItemsd, SelectItemsd
)

from config import CFG
from dataset import _resolve_path
from model import build_model

# bring in helpers you already have in transforms.py
from transforms import (
    FixImageShaped, FixMaskShaped, EnsurePhaseKeysd, UnionMasksd,
    MakeROIdFromLabd, Ensure3DROId, WindowTo01d, GaussianSmoothd
)

# ----------------- utilities -----------------
def save_overlay_pngs(img_np, mask_np, out_dir, pid, gain=0.25):
    try:
        from imageio import imwrite
    except ImportError:
        from skimage.io import imsave as imwrite

    out = Path(out_dir) / f"preview_{pid}"
    out.mkdir(parents=True, exist_ok=True)
    Z = img_np.shape[0]
    for z in range(Z):
        base = img_np[z]
        m = mask_np[z]
        bright = np.clip(base + gain * m, 0.0, 1.0)
        im = (bright * 255).astype(np.uint8)
        imwrite(out / f"z{z:03d}.png", im)

def window_to_01(x, lo, hi):
    den = max(float(hi - lo), 1e-6)
    return np.clip((x - lo) / den, 0.0, 1.0)

# ------------- ROI-aware predict transforms (no image masking, no cropping) -------------
def make_predict_transforms_roi(cfg: CFG):
    mask_keys = ["mask_C1", "mask_C2", "mask_C3"]
    def per_phase(win_prefix, src_key):
        return [
            CopyItemsd(keys=[src_key], names=[f"{win_prefix}_w1", f"{win_prefix}_w2", f"{win_prefix}_w3"], times=3),
            WindowTo01d(keys=[f"{win_prefix}_w1"], lo=cfg.HU_MIN, hi=cfg.HU_MAX),
            WindowTo01d(keys=[f"{win_prefix}_w2"], lo=-160.0, hi=240.0),
            WindowTo01d(keys=[f"{win_prefix}_w3"], lo=0.0,   hi=200.0),
            CopyItemsd(keys=[f"{win_prefix}_w1", f"{win_prefix}_w2", f"{win_prefix}_w3"],
                       names=[f"{win_prefix}_w1s", f"{win_prefix}_w2s", f"{win_prefix}_w3s"]),
            GaussianSmoothd(keys=[f"{win_prefix}_w1s", f"{win_prefix}_w2s", f"{win_prefix}_w3s"], sigma=0.75),
        ]

    return Compose([
        LoadImaged(
            keys=["path_C1", "path_C2", "path_C3", "path_mask_C1", "path_mask_C2", "path_mask_C3"],
            allow_missing_keys=True, image_only=False
        ),
        CopyItemsd(keys=["path_C1"], names=["img_c1"]),
        CopyItemsd(keys=["path_C2"], names=["img_c2"], allow_missing_keys=True),
        CopyItemsd(keys=["path_C3"], names=["img_c3"], allow_missing_keys=True),
        CopyItemsd(keys=["path_mask_C1", "path_mask_C2", "path_mask_C3"], names=mask_keys, allow_missing_keys=True),
        DeleteItemsd(keys=["path_C1", "path_C2", "path_C3", "path_mask_C1", "path_mask_C2", "path_mask_C3"]),

        FixImageShaped(keys=["img_c1", "img_c2", "img_c3"]),
        FixMaskShaped(keys=mask_keys),
        EnsureChannelFirstd(keys=["img_c1","img_c2","img_c3"] + mask_keys,
                            channel_dim="no_channel", allow_missing_keys=True),
        EnsureTyped(keys=["img_c1","img_c2","img_c3"] + mask_keys,
                    dtype=[np.float32, np.float32, np.float32] + [np.uint8]*len(mask_keys),
                    track_meta=True, allow_missing_keys=True),

        Orientationd(keys=["img_c1","img_c2","img_c3"] + mask_keys, axcodes="RAS", allow_missing_keys=True),
        Spacingd(keys=["img_c1","img_c2","img_c3"], pixdim=cfg.SPACING,
                 mode=InterpolateMode.BILINEAR, padding_mode="border", allow_missing_keys=True),
        ResampleToMatchd(keys=["img_c2","img_c3"], key_dst="img_c1",
                         mode=InterpolateMode.BILINEAR, padding_mode="border", allow_missing_keys=True),
        ResampleToMatchd(keys=mask_keys, key_dst="img_c1",
                         mode=InterpolateMode.NEAREST, padding_mode="border", allow_missing_keys=True),

        SqueezeDimd(keys=mask_keys, dim=0, allow_missing_keys=True),
        EnsurePhaseKeysd(keys=["img_c2", "img_c3"], ref_key="img_c1"),

        UnionMasksd(img_key="img_c1", mask_keys=mask_keys),
        MakeROIdFromLabd(lab_key="lab", roi_key="roi", dilate_k=(5,5,5)),
        Ensure3DROId(keys=("roi",)),

        *per_phase("c1", "img_c1"),
        *per_phase("c2", "img_c2"),
        *per_phase("c3", "img_c3"),
        ConcatItemsd(
            keys=[
                "c1_w1","c1_w1s","c1_w2","c1_w2s","c1_w3","c1_w3s",
                "c2_w1","c2_w1s","c2_w2","c2_w2s","c2_w3","c2_w3s",
                "c3_w1","c3_w1s","c3_w2","c3_w2s","c3_w3","c3_w3s",
            ],
            name="img", dim=0
        ),

        SelectItemsd(keys=["img", "roi", "lab", "img_c1"]),
        EnsureTyped(keys=["img", "roi", "lab", "img_c1"],
                    dtype=[np.float32, np.uint8, np.uint8, np.float32],
                    track_meta=True),
    ])

# def make_predict_transforms_roi(cfg: CFG):
    """
    Phase-aware predict pipeline.
    Uses cfg.PHASES (e.g. ["C1"], ["C1","C2"], ["C1","C2","C3"]) to decide
    which phases contribute to the input channels.
    """

    mask_keys = ["mask_C1", "mask_C2", "mask_C3"]

    def per_phase(phase_name: str, src_key: str):
        """
        Build transforms for one phase:
        - 3 HU windows
        - raw + smoothed for each window
        """
        p = phase_name.lower()  # "C1" -> "c1"
        return [
            # duplicate source image 3 times for 3 HU windows
            CopyItemsd(
                keys=[src_key],
                names=[f"{p}_w1", f"{p}_w2", f"{p}_w3"],
                times=3
            ),
            # window #1: wide HU range
            WindowTo01d(keys=[f"{p}_w1"], lo=cfg.HU_MIN, hi=cfg.HU_MAX),
            # window #2: typical liver window
            WindowTo01d(keys=[f"{p}_w2"], lo=-160.0, hi=240.0),
            # window #3: soft tissue-ish
            WindowTo01d(keys=[f"{p}_w3"], lo=0.0, hi=200.0),
            # copy for smoothed versions
            CopyItemsd(
                keys=[f"{p}_w1", f"{p}_w2", f"{p}_w3"],
                names=[f"{p}_w1s", f"{p}_w2s", f"{p}_w3s"],
            ),
            GaussianSmoothd(
                keys=[f"{p}_w1s", f"{p}_w2s", f"{p}_w3s"],
                sigma=0.75,
            ),
        ]

    # mapping PHASE name -> image key produced earlier in the pipeline
    phase_to_imgkey = {
        "C1": "img_c1",
        "C2": "img_c2",
        "C3": "img_c3",
    }

    # build per-phase blocks + concat key list according to cfg.PHASES
    per_phase_blocks = []
    concat_keys = []
    for phase in cfg.PHASES:
        src = phase_to_imgkey[phase]  # e.g. "img_c1"
        p = phase.lower()             # "C1" -> "c1"

        per_phase_blocks += per_phase(phase, src)
        concat_keys += [
            f"{p}_w1",  f"{p}_w1s",
            f"{p}_w2",  f"{p}_w2s",
            f"{p}_w3",  f"{p}_w3s",
        ]

    return Compose([
        # ------------- load + geometry, similar to train/val -------------
        LoadImaged(
            keys=["path_C1", "path_C2", "path_C3",
                  "path_mask_C1", "path_mask_C2", "path_mask_C3"],
            allow_missing_keys=True,
            image_only=False,
        ),

        # copy from paths to working image/mask keys
        CopyItemsd(keys=["path_C1"], names=["img_c1"]),
        CopyItemsd(keys=["path_C2"], names=["img_c2"], allow_missing_keys=True),
        CopyItemsd(keys=["path_C3"], names=["img_c3"], allow_missing_keys=True),
        CopyItemsd(
            keys=["path_mask_C1", "path_mask_C2", "path_mask_C3"],
            names=mask_keys,
            allow_missing_keys=True,
        ),
        DeleteItemsd(
            keys=["path_C1", "path_C2", "path_C3",
                  "path_mask_C1", "path_mask_C2", "path_mask_C3"],
        ),

        FixImageShaped(keys=["img_c1", "img_c2", "img_c3"]),
        FixMaskShaped(keys=mask_keys),

        EnsureChannelFirstd(
            keys=["img_c1", "img_c2", "img_c3"] + mask_keys,
            channel_dim="no_channel",
            allow_missing_keys=True,
        ),
        EnsureTyped(
            keys=["img_c1", "img_c2", "img_c3"] + mask_keys,
            dtype=[np.float32, np.float32, np.float32] + [np.uint8] * len(mask_keys),
            track_meta=True,
            allow_missing_keys=True,
        ),

        Orientationd(
            keys=["img_c1", "img_c2", "img_c3"] + mask_keys,
            axcodes="RAS",
            allow_missing_keys=True,
        ),

        # resample all phases to cfg.SPACING and align to C1
        Spacingd(
            keys=["img_c1", "img_c2", "img_c3"],
            pixdim=cfg.SPACING,
            mode=InterpolateMode.BILINEAR,
            padding_mode="border",
            allow_missing_keys=True,
        ),
        ResampleToMatchd(
            keys=["img_c2", "img_c3"],
            key_dst="img_c1",
            mode=InterpolateMode.BILINEAR,
            padding_mode="border",
            allow_missing_keys=True,
        ),
        ResampleToMatchd(
            keys=mask_keys,
            key_dst="img_c1",
            mode=InterpolateMode.NEAREST,
            padding_mode="border",
            allow_missing_keys=True,
        ),

        SqueezeDimd(keys=mask_keys, dim=0, allow_missing_keys=True),

        # ensure missing phases exist as zeros (so resampling/meta is consistent)
        EnsurePhaseKeysd(keys=["img_c2", "img_c3"], ref_key="img_c1"),

        # union masks -> lab, then ROI, then ensure 3D ROI
        UnionMasksd(img_key="img_c1", mask_keys=mask_keys),
        MakeROIdFromLabd(lab_key="lab", roi_key="roi", dilate_k=(5, 5, 5)),
        Ensure3DROId(keys=("roi",)),

        # geometry / ROI alignment
        EnsureSameShaped(keys=["img_c1", "lab", "roi"]),
        ApplyROIMaskd(img_keys=("img_c1",), lab_key="lab", roi_key="roi"),
        Ensure3DROId(keys=("roi",)),
        EnsureChannelFirstd(keys=["lab", "roi"], channel_dim="no_channel"),

        # we keep original HU C1 for saving background
        CopyItemsd(keys=["img_c1"], names=["img_c1_raw"]),

        # ----------------- phase-dependent windows -----------------
        *per_phase_blocks,

        ConcatItemsd(
            keys=concat_keys,
            name="img",
            dim=0,
        ),

        # we want: "img" (C,Z,Y,X), "roi" (Z,Y,X), "lab" optional, and "img_c1_raw"
        SelectItemsd(keys=["img", "roi", "lab", "img_c1_raw"]),
        EnsureTyped(
            keys=["img", "roi", "lab", "img_c1_raw"],
            dtype=[np.float32, np.uint8, np.uint8, np.float32],
            track_meta=True,
        ),
    ])

# ----------------- ensemble helpers -----------------
def _load_models(folds, in_channels):
    """Load available checkpoints for the given folds; skip missing ones."""
    cfg = CFG()
    models = []
    infos = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for f in folds:
        ckpt = os.path.join(cfg.fold_dir(f), "best.pt")
        if not os.path.exists(ckpt):
            print(f"[warn] missing checkpoint for fold {f}: {ckpt} (skipping)")
            continue
        m = build_model(in_channels=in_channels)
        state = torch.load(ckpt, map_location="cpu")
        m.load_state_dict(state["state_dict"])
        m.eval().to(device)
        models.append(m)
        infos.append((f, ckpt, state.get("epoch"), state.get("dice")))
    assert len(models) > 0, "No checkpoints found."
    print("[info] loaded folds:", ", ".join(
        f"f{fi}(ep={ep},dice={d:.4f})" if d is not None else f"f{fi}(ep={ep})"
        for (fi, _, ep, d) in infos
    ))
    return models, device, infos

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _prob_to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def _logit_to_prob(z):
    return _sigmoid(z)

def _infer_one(models, device, x, roi_size, overlap):
    """
    x: torch tensor [1, C, Z, Y, X] on device
    Returns: prob np.ndarray (Z,Y,X) averaged across models (logit-avg).
    """
    logits_acc = None
    with torch.no_grad():
        use_amp = torch.cuda.is_available()
        amp_ctx = torch.autocast(device_type="cuda") if use_amp else torch.cpu.amp.autocast(enabled=False)
        with amp_ctx:
            for m in models:
                preds = sliding_window_inference(
                    x, roi_size=roi_size, sw_batch_size=1,
                    predictor=m, overlap=overlap
                )  # [1,1,Z,Y,X]
                prob = torch.sigmoid(preds)[0, 0].float().cpu().numpy()
                logit = _prob_to_logit(prob)
                logits_acc = logit if logits_acc is None else (logits_acc + logit)
    logits_mean = logits_acc / float(len(models))
    return _logit_to_prob(logits_mean)

# ----------------- main inference -----------------
def run_infer_ens(folds=(0,1,2,3,4), out_subdir="exports_ens", thresh=0.97):
    cfg = CFG()
    df = pd.read_csv(cfg.CSV_PATH)
    vt = make_predict_transforms_roi(cfg)

    models, device, infos = _load_models(folds, in_channels=cfg.IN_CHANNELS)
    export_dir = os.path.join(cfg.OUT_DIR, out_subdir)
    os.makedirs(export_dir, exist_ok=True)

    print("[info] exporting to:", export_dir)

    with torch.no_grad():
        for _, r in df.iterrows():
            pid = r["patient_id"]
            data = {
                "path_C1": _resolve_path(r.get("path_C1", None)),
                "path_C2": _resolve_path(r.get("path_C2", None)),
                "path_C3": _resolve_path(r.get("path_C3", None)),
                "path_mask_C1": _resolve_path(r.get("path_mask_C1", None)),
                "path_mask_C2": _resolve_path(r.get("path_mask_C2", None)),
                "path_mask_C3": _resolve_path(r.get("path_mask_C3", None)),
            }
            data = vt(data)

            img = data["img"]             # (18,Z,Y,X) in [0,1]
            roi = np.asarray(data["roi"]) # (Z,Y,X) uint8
            assert img.shape[0] == cfg.IN_CHANNELS, f"bad channels: {img.shape}"
            if roi.sum() == 0:
                print(f"[warn] {pid}: ROI empty, skipping.")
                continue

            # forward (full-volume sliding window) for each model, then logit-average
            x = img.unsqueeze(0).to(device)  # [1,C,Z,Y,X]
            prob = _infer_one(models, device, x, cfg.ROI_SIZE, cfg.SW_OVERLAP)  # (Z,Y,X) float

            # gate strictly by ROI
            prob *= roi.astype(np.float32)
            assert float(prob[roi == 0].sum()) == 0.0

            # --- CLEAN tumor mask ---
            T = float(thresh)  # stricter threshold; tune if needed
            seg = (prob > T).astype(np.uint8)

            # keep largest connected component
            lbl, n = ndi.label(seg)
            if n > 0:
                sizes = ndi.sum(seg, lbl, index=np.arange(1, n+1))
                seg = (lbl == (1 + sizes.argmax())).astype(np.uint8)

            # morphology cleanup
            seg = ndi.binary_opening(seg, structure=np.ones((3,3,3))).astype(np.uint8)
            seg = ndi.binary_fill_holes(seg, structure=np.ones((3,3,3))).astype(np.uint8)

            # affine for saving
            # affine = getattr(data["img"], "affine", None)
            # if affine is None:
            #     affine = data["img"].meta.get("affine", np.eye(4))

            # # save prob / seg (transpose (Z,Y,X) -> (X,Y,Z))
            # prob_xyz = np.transpose(prob.astype(np.float32), (2, 1, 0))
            # seg_xyz  = np.transpose(seg.astype(np.uint8),    (2, 1, 0))
            # nib.save(nib.Nifti1Image(prob_xyz, affine), os.path.join(export_dir, f"{pid}_prob.nii.gz"))
            # nib.save(nib.Nifti1Image(seg_xyz,  affine), os.path.join(export_dir, f"{pid}_seg.nii.gz"))

            # # save raw HU C1 (resampled) as background
            # c1hu = data["img_c1"][0].cpu().numpy()           # (Z,Y,X), HU-like
            # c1hu_xyz = np.transpose(c1hu.astype(np.float32), (2, 1, 0))
            # nib.save(nib.Nifti1Image(c1hu_xyz, affine), os.path.join(export_dir, f"{pid}_c1_resampled.nii.gz"))

            # # save boosted background: brighten ONLY tumor voxels by +25%
            # c1v = window_to_01(c1hu, lo=-160.0, hi=240.0)    # [0,1]
            # boosted = np.clip(c1v + 0.25 * seg, 0.0, 1.0)
            # boosted_xyz = np.transpose(boosted.astype(np.float32), (2, 1, 0))
            # nib.save(nib.Nifti1Image(boosted_xyz, affine), os.path.join(export_dir, f"{pid}_c1_boosted.nii.gz"))

            # affine for saving – use img_c1 as the reference geometry
            ref = data["img_c1"]
            affine = getattr(ref, "affine", None)
            if affine is None:
                affine = ref.meta.get("affine", np.eye(4))

            # prob, seg: (Z, Y, X)
            prob_zyx = prob.astype(np.float32)
            seg_zyx  = seg.astype(np.uint8)
            nib.save(nib.Nifti1Image(prob_zyx, affine), os.path.join(export_dir, f"{pid}_prob.nii.gz"))
            nib.save(nib.Nifti1Image(seg_zyx,  affine), os.path.join(export_dir, f"{pid}_seg.nii.gz"))

            # save raw HU C1 (resampled) as background
            c1hu_zyx = data["img_c1"][0].cpu().numpy().astype(np.float32)  # (Z,Y,X)
            nib.save(nib.Nifti1Image(c1hu_zyx, affine), os.path.join(export_dir, f"{pid}_c1_resampled.nii.gz"))

            # boosted
            c1v = window_to_01(c1hu_zyx, lo=-160.0, hi=240.0)  # [0,1]
            boosted = np.clip(c1v + 0.25 * seg_zyx, 0.0, 1.0).astype(np.float32)
            nib.save(nib.Nifti1Image(boosted, affine), os.path.join(export_dir, f"{pid}_c1_boosted.nii.gz"))


            # png previews
            save_overlay_pngs(c1v, seg, export_dir, pid, gain=0.25)

            print(f"{pid} | img {tuple(img.shape)} | ROI vox {int(roi.sum())} | "
                  f"prob min/max/mean {prob.min():.3f}/{prob.max():.3f}/{prob.mean():.3f} | "
                  f"seg vox {int(seg.sum())}")

if __name__ == "__main__":
    # Use all five folds by default; you can pass a subset, e.g., (0,2,4)
    run_infer_ens(folds=(0,1,2,3,4), out_subdir="exports_ens", thresh=0.95)
