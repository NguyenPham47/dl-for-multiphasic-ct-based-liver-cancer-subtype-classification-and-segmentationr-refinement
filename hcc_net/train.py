import os, pandas as pd, numpy as np, torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from config import CFG
from dataset import HCCDataset, NAME2IDX
from model import build_model
from losses import get_criterion
from utils import seed_everything, AverageMeter, compute_metrics

def compute_class_weights(df):
    counts = df["label"].value_counts().reindex(CFG.CLASSES).fillna(0).values.astype(float)
    inv = counts.max() / np.clip(counts, 1, None)
    return (inv / inv.mean())

def get_fold(df, fold_idx=0, n_folds=5):
    df = df[df["label"].isin(CFG.CLASSES)].copy()
    df = df.dropna(subset=[f"path_{p}" for p in CFG.PHASES], how='all')  # remove totally empty rows
    gkf = GroupKFold(n_splits=n_folds)
    groups = df["patient_id"].values
    for i, (tr, va) in enumerate(gkf.split(df, df["label"], groups)):
        if i == fold_idx:
            return df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)
    raise ValueError("Fold index out of range")

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    meter = AverageMeter()
    for b in tqdm(loader, desc="train", leave=False):
        x = b["image"].to(device); y = b["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=CFG.MIXED_PRECISION):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        meter.update(loss.item(), x.size(0))
    return meter.avg

@torch.no_grad()
def valid_one_epoch(model, loader, criterion, device):
    model.eval()
    meter = AverageMeter(); probs=[]; gts=[]
    for b in tqdm(loader, desc="valid", leave=False):
        x = b["image"].to(device); y = b["label"].to(device)
        logits = model(x)
        loss = criterion(logits, y)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p); gts += y.cpu().tolist()
        meter.update(loss.item(), x.size(0))
    probs = np.concatenate(probs, axis=0)
    acc, f1m, auc = compute_metrics(np.array(gts), probs)
    return meter.avg, acc, f1m, auc

def main():
    print("Training on model:", CFG.MODEL_NAME, ", on fold:", CFG.FOLD_IDX, ", on phase(s):", CFG.PHASES, ", with", CFG.SLICES_PER_PHASE, "slices per phase.")
    seed_everything(CFG.SEED)
    os.makedirs(CFG.CKPT_DIR, exist_ok=True)

    df = pd.read_csv(CFG.CSV_PATH)
    tr_df, va_df = get_fold(df, CFG.FOLD_IDX, CFG.N_FOLDS)
    class_w = compute_class_weights(tr_df)

    train_loader = DataLoader(HCCDataset(tr_df, training=True), batch_size=CFG.BATCH_SIZE,
                              shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(HCCDataset(va_df, training=False), batch_size=CFG.BATCH_SIZE*2,
                            shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    criterion = get_criterion(class_weights=class_w).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.COSINE_TMAX)
    scaler = GradScaler(enabled=CFG.MIXED_PRECISION)

    if CFG.FREEZE_BACKBONE_EPOCHS > 0:
        for p in model.backbone.parameters(): p.requires_grad = False

    best_f1, patience = -1, 0
    for epoch in range(1, CFG.EPOCHS+1):
        if epoch == CFG.FREEZE_BACKBONE_EPOCHS + 1:
            for p in model.backbone.parameters(): p.requires_grad = True

        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        va_loss, acc, f1m, auc = valid_one_epoch(model, val_loader, criterion, device)
        scheduler.step(epoch + va_loss)
        print(f"Epoch {epoch:02d} | tr={tr_loss:.4f} va={va_loss:.4f} acc={acc:.3f} f1m={f1m:.3f} auc={auc:.3f}")

        if f1m > best_f1:
            best_f1, patience = f1m, 0
            torch.save({"model": model.state_dict(), "classes": CFG.CLASSES, "best_f1": best_f1}, CFG.BEST_CKPT)
        else:
            patience += 1
            if patience >= CFG.EARLY_STOP_PATIENCE:
                print(f"Early stop. Best macro-F1={best_f1:.3f}")
                break

if __name__ == "__main__":
    main()
