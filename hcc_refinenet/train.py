import os, numpy as np, torch, pandas as pd
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from config import CFG
from dataset import make_folds, make_loaders
from transforms import make_train_transforms, make_val_transforms
from model import build_model
from eval_metrics import DiceMeter
from losses import DiceFocalCombo
from monai.losses import DiceLoss


def roi_sanity(batch):
    img = batch["img"]                     # [B,C,Z,Y,X]
    roi = batch["roi"]                     # [B,Z,Y,X] or [B,1,Z,Y,X]
    lab = batch["lab"]                     # [B,Z,Y,X] or [B,1,Z,Y,X]

    if roi.ndim == 4: roi = roi.unsqueeze(1)          # -> [B,1,Z,Y,X]
    if lab.ndim == 4: lab = lab.unsqueeze(1)          # -> [B,1,Z,Y,X]

    # sums
    total_img_sum = img.abs().sum().item()
    roi_img_sum   = (img * roi).abs().sum().item()
    outside_img_sum = (img * (1 - roi)).abs().sum().item()

    total_lab_sum = lab.sum().item()
    roi_lab_sum   = (lab * roi).sum().item()
    outside_lab_sum = (lab * (1 - roi)).sum().item()

    print(
        f"[ROI sanity] img_sum: total={total_img_sum:.3f} | "
        f"roi={roi_img_sum:.3f} | outside={outside_img_sum:.3f}"
    )
    print(
        f"[ROI sanity] lab_sum: total={total_lab_sum:.0f} | "
        f"roi={roi_lab_sum:.0f} | outside={outside_lab_sum:.0f}"
    )

def train_one_fold(fold):
    cfg = CFG()
    os.makedirs(cfg.fold_dir(fold), exist_ok=True)

    df = pd.read_csv(cfg.CSV_PATH)

    folds = make_folds(df, n_folds=cfg.N_FOLDS, seed=cfg.SEED)
    tr_idx, va_idx = folds[fold]
    train_df = df.iloc[tr_idx].reset_index(drop=True)
    val_df   = df.iloc[va_idx].reset_index(drop=True)

    train_tf = make_train_transforms(cfg)
    val_tf   = make_val_transforms(cfg)

    train_loader, val_loader = make_loaders(cfg, train_df, val_df, train_tf, val_tf)

    # ---------- SANITY PROBE: paste this block ----------
    from itertools import islice
    pos_vox = 0
    tot_vox = 0
    for b in islice(iter(train_loader), 4):  # check a few batches
        roi_sanity(b)
        lab = b["lab"]  # shape [B,1,Z,Y,X] or [B,Z,Y,X]
        if lab.ndim == 4:
            lab = lab.unsqueeze(1)  # -> [B,1,Z,Y,X]
        pos_vox += (lab > 0.5).sum().item()
        tot_vox += lab.numel()
    print(f"[Sanity] label positive voxels = {pos_vox} / {tot_vox} "
          f"({100*pos_vox/max(1,tot_vox):.4f}%)")
    # -----------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(in_channels=cfg.IN_CHANNELS).to(device)

    if cfg.USE_FOCAL:
        criterion = DiceFocalCombo(cfg.DICE_WEIGHT, cfg.FOCAL_WEIGHT, cfg.FOCAL_ALPHA, cfg.FOCAL_GAMMA)
    else:
        criterion = DiceLoss(sigmoid=True, include_background=False, squared_pred=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    best_dice = -1.0
    ckpt_path = os.path.join(cfg.fold_dir(fold), "best.pt")

    print("train batches:", len(train_loader))
    first = next(iter(train_loader))  # should return quickly
    print("first batch OK:", {k: v.shape if hasattr(v, 'shape') else type(v) for k,v in first.items()})


    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Fold{fold} Epoch{epoch}"):
            x = batch["img"].to(device)        # [B, 6, Z, Y, X]
            y = batch["lab"].to(device).unsqueeze(1) if batch["lab"].ndim==4 else batch["lab"].to(device)  # [B,1,...]
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())

        # ---- validation
        model.eval()
        meter = DiceMeter()
        with torch.no_grad():
            for batch in val_loader:
                x = batch["img"].to(device)
                y = batch["lab"].to(device)
                if y.ndim == 4:
                    y = y.unsqueeze(1)  # -> [B,1,Z,Y,X]

                preds = sliding_window_inference(
                    x, roi_size=cfg.ROI_SIZE, sw_batch_size=1,
                    predictor=model, overlap=cfg.SW_OVERLAP
                )
                probs = torch.sigmoid(preds)  # [B,1,Z,Y,X]

                # >>> ROI masking block (insert this)
                roi = batch.get("roi", None)
                if roi is not None:
                    roi = roi.to(device)
                    if roi.ndim == 4:
                        roi = roi.unsqueeze(1)  # -> [B,1,Z,Y,X]
                    roi = roi.float()
                    probs = probs * roi
                    y = y * roi
                # <<< end ROI masking

                seg = (probs > 0.5).float()
                meter.update(seg, y)

        mean_loss = float(np.mean(losses))
        dice = meter.compute()
        print(f"[Fold {fold}] Epoch {epoch:03d} | loss {mean_loss:.4f} | dice {dice:.4f}")

        if dice > best_dice:
            best_dice = dice
            torch.save({"state_dict": model.state_dict(), "dice": best_dice, "epoch": epoch}, ckpt_path)
            print(f"  -> saved new best to {ckpt_path} (dice={best_dice:.4f})")

    print(f"[Fold {fold}] best dice = {best_dice:.4f}")

if __name__ == "__main__":
    train_one_fold(CFG().FOLD_IDX)
