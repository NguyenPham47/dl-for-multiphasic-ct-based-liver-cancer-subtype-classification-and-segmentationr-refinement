import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import CFG
from dataset import HCCDataset
from model import build_model


@torch.no_grad()
def infer_external_ensemble(csv_path, ckpt_dir, out_csv):
    df = pd.read_csv(csv_path)
    ds = HCCDataset(df, training=False)
    loader = DataLoader(
        ds,
        batch_size=CFG.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_probs = []
    for fold_idx in range(CFG.N_FOLDS):
        ckpt_path = os.path.join(ckpt_dir, f"best_fold{fold_idx}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[WARN] missing ckpt: {ckpt_path}, skipping fold")
            continue

        print(f"[INFO] loading: {ckpt_path}")
        model = build_model().to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"], strict=True)
        model.eval()

        probs_all = []
        for b in tqdm(loader, desc=f"Fold {fold_idx} inference", leave=False):
            x = b["image"].to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_all.append(probs)

        probs_all = np.concatenate(probs_all, axis=0)
        fold_probs.append(probs_all)

    if not fold_probs:
        raise RuntimeError("No checkpoints loaded; fold_probs is empty.")

    probs_avg = np.mean(np.stack(fold_probs, axis=0), axis=0)  # (N, num_classes)
    pred_idx = probs_avg.argmax(axis=1)
    pred_label = [CFG.CLASSES[i] for i in pred_idx]

    out = df.copy()
    for i, cls in enumerate(CFG.CLASSES):
        out[f"prob_{cls}"] = probs_avg[:, i]
    out["pred"] = pred_label

    out.to_csv(out_csv, index=False)
    print(f"[INFO] saved ensemble predictions to: {out_csv}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="external_patient_rows.csv")
    ap.add_argument("--ckpt_dir", type=str, required=True,
                    help="folder containing best_fold0.pt ... best_fold4.pt")
    ap.add_argument("--out_csv", type=str, required=True,
                    help="output CSV with ensemble predictions")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer_external_ensemble(args.csv, args.ckpt_dir, args.out_csv)
