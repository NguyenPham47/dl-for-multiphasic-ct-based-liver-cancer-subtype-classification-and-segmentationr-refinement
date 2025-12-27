import pandas as pd
from torch.utils.data import DataLoader
import torch

from config import CFG
from dataset import HCCDataset


CSV = r"D:\Downloads\ExternalData\external_patient_rows.csv"


def main():
    print("=== EXTERNAL DATASET SHAPE DEBUG ===")
    print(f"CSV: {CSV}")
    df = pd.read_csv(CSV)
    print(f"Total patients: {len(df)}")

    # IMPORTANT: set the slices-per-phase you want to test
    # 1 for 2D, 5 for 2.5D, 13 for 2.5D
    print(f"SLICES_PER_PHASE = {CFG.SLICES_PER_PHASE}")

    # Build dataset
    ds = HCCDataset(df, training=False)

    # Use num_workers=0 to avoid silent hangs
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    for i, batch in enumerate(loader):
        x = batch["image"]       # CHW
        lbl = batch["label"]
        pid = batch["pid"]

        print("\n-----------------------------------")
        print(f"Sample index       : {i}")
        print(f"Patient ID         : {pid[0]}")
        print(f"Label              : {lbl.item()}")
        print(f"Image tensor shape : {tuple(x.shape)}")  # (C, H, W)

        # On first sample, break? No—print first 5.
        if i >= 4:
            break

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
