import pandas as pd
from config import CFG

def build_patient_df(phase_csv=CFG.PHASE_CSV, out_csv=CFG.CSV_PATH):
    df = pd.read_csv(phase_csv)

    # Standardize column names expected from your description
    # patient_id, age, gender, phase, cancer_type, ct_path, mask_path, liver_mask_path
    req = ["patient_id","phase","cancer_type","ct_path","mask_path","liver_mask_path"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # filter to our 3 classes
    df = df[df["cancer_type"].isin(CFG.CLASSES)].copy()

    # pick one label per patient (should all agree)
    labels = df.groupby("patient_id")["cancer_type"].agg(lambda x: x.iloc[0])

    # pivot CT paths per phase
    ct_wide = df.pivot_table(index="patient_id", columns="phase", values="ct_path", aggfunc="first")
    ct_wide = ct_wide.reindex(columns=CFG.PHASES)

    # choose a liver mask path per patient: priority P -> C3 -> C2 -> C1
    prio = ["P","C3","C2","C1"]
    mask_choices = (
        df.sort_values(by=["patient_id"], kind="stable")
          .assign(phase_order=df["phase"].map({p:i for i,p in enumerate(prio)}).fillna(99))
          .sort_values(["patient_id","phase_order"])
          .groupby("patient_id")["liver_mask_path"].agg(lambda x: next((v for v in x if isinstance(v,str) and len(v)>0), ""))
    )

    lesion_mask_choices = (
        df.sort_values(by=["patient_id"], kind="stable")
          .assign(phase_order=df["phase"].map({p:i for i,p in enumerate(prio)}).fillna(99))
          .sort_values(["patient_id","phase_order"])
          .groupby("patient_id")["mask_path"].agg(lambda x: next((v for v in x if isinstance(v,str) and len(v)>0), ""))
    )

    out = ct_wide.copy()
    out.columns = [f"path_{c}" for c in out.columns]  # path_C1, path_C2, ...
    out["label"] = labels
    out["liver_mask_path"] = mask_choices
    out["lesion_mask_path"] = lesion_mask_choices
    out = out.reset_index()

    out.to_csv(out_csv, index=False)
    print(f"Saved patient-level CSV: {out_csv}  (rows={len(out)})")

if __name__ == "__main__":
    build_patient_df()
