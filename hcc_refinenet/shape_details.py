import pandas as pd
import nibabel as nib
from config import CFG
import os

# df = pd.read_csv(CFG.CSV_PATH)
df = pd.read_csv(r"D:\HCC\TumorDetection\patient_tumor_shape_details.csv")

for i, r in df.iterrows():
    try:
        # img_path_C1 = os.path.join(CFG.IMG_ROOT, r["path_C1"])
        # img_path_C2 = os.path.join(CFG.IMG_ROOT, r["path_C2"])
        # img_path_C3 = os.path.join(CFG.IMG_ROOT, r["path_C3"])
        # img_path_P = os.path.join(CFG.IMG_ROOT, r["path_P"])
        # img_path = os.path.join(CFG.IMG_ROOT, r["path_C1"])

        msk_path = os.path.join(CFG.IMG_ROOT, r["path_mask_C1"])
        # img_C1 = nib.load(img_path_C1)
        # img_C2 = nib.load(img_path_C2)
        # img_C3 = nib.load(img_path_C3)
        # img_P = nib.load(img_path_P)

        msk = nib.load(msk_path)
        # print(f"{r['patient_id']}: img {img.shape}, mask {msk.shape}")
        # print(f"{r['patient_id']}: img C1 {img_C1.shape}, img C2 {img_C2.shape}, img C3 {img_C3.shape}, img P {img_P.shape}")
        print(f"{r['patient_id']}: mask {msk.shape}")
    except Exception as e:
        print(f"{r['patient_id']}: error -> {e}")
