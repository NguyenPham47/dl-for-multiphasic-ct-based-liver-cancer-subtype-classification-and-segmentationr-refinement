import pandas as pd, numpy as np, torch
from torch.utils.data import DataLoader
from config import CFG
from dataset import HCCDataset
from model import build_model

@torch.no_grad()
def infer(csv_path=CFG.CSV_PATH, ckpt_path=CFG.BEST_CKPT, out_csv="pred_multiphase.csv"):
    df = pd.read_csv(csv_path)
    ds = HCCDataset(df, training=False)
    loader = DataLoader(ds, batch_size=CFG.BATCH_SIZE*2, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    probs_all, preds_all = [], []
    for b in loader:
        x = b["image"].to(device)
        p = torch.softmax(model(x), dim=1).cpu().numpy()
        probs_all.append(p); preds_all += p.argmax(1).tolist()
    probs_all = np.concatenate(probs_all, axis=0)

    out = df.copy()
    for i, cls in enumerate(CFG.CLASSES):
        out[f"prob_{cls}"] = probs_all[:, i]
    out["pred"] = [CFG.CLASSES[i] for i in preds_all]
    out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    infer()
