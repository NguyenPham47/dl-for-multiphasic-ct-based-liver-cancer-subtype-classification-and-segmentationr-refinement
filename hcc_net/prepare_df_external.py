import os
import pandas as pd
from typing import Dict, List
from config import CFG   # to reuse PHASES: ["C1","C2","C3","P"]


# External dataset folder roots you described
EXTERNAL_ROOTS: List[str] = [
    r"D:\Downloads\ExternalData\ct_scans_1_4_wawtace_09_05_24",
    r"D:\Downloads\ExternalData\ct_scans_2_4_wawtace_09_05_24",
    r"D:\Downloads\ExternalData\ct_scans_3_4_wawtace_09_05_24",
    r"D:\Downloads\ExternalData\ct_scans_4_4_wawtace_09_05_24",
]


# ✔️ CORRECT MAPPING according to dataset + your research paper:
# External index → Your internal phase name
PHASE_IDX2NAME = {
    0: "P",     # plain
    1: "C1",    # arterial
    2: "C2",    # venous
    3: "C3",    # delayed
}


def _parse_phase_from_filename(fname: str, expected_pid: str):
    """
    Expect files like:  '35_0_scan.nii.gz', '35_2_scan.nii.gz'
    Return phase_idx (int) or None.
    """
    if fname.endswith(".nii.gz"):
        base = fname[:-7]
    elif fname.endswith(".nii"):
        base = fname[:-4]
    else:
        return None

    parts = base.split("_")
    if len(parts) < 2:
        return None

    pid_part, phase_part = parts[0], parts[1]

    # optional consistency check
    if pid_part != str(expected_pid):
        return None

    try:
        return int(phase_part)
    except:
        return None


def build_external_patient_df(
    out_csv: str = r"D:\Downloads\ExternalData\external_patient_rows.csv"
):
    """
    Produces a patient-wise CSV with columns:
        patient_id, path_C1, path_C2, path_C3, path_P

    Missing phases → empty strings.
    """
    patients: Dict[str, Dict[str, str]] = {}

    for root in EXTERNAL_ROOTS:
        if not os.path.isdir(root):
            print(f"[WARN] Directory missing: {root}")
            continue

        for pid_dir in os.listdir(root):
            full_pid_dir = os.path.join(root, pid_dir)
            if not os.path.isdir(full_pid_dir):
                continue

            pid = str(pid_dir)
            rec = patients.setdefault(pid, {"patient_id": pid})

            for fname in os.listdir(full_pid_dir):
                if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
                    continue

                phase_idx = _parse_phase_from_filename(fname, pid)
                if phase_idx is None:
                    continue

                phase_name = PHASE_IDX2NAME.get(phase_idx)
                if phase_name is None:
                    continue

                abs_path = os.path.abspath(os.path.join(full_pid_dir, fname))
                rec[f"path_{phase_name}"] = abs_path

    # Convert dict → DataFrame
    df = pd.DataFrame(list(patients.values()))

    # Ensure all required phase columns exist
    for ph in CFG.PHASES:   # ["C1","C2","C3","P"]
        col = f"path_{ph}"
        if col not in df.columns:
            df[col] = ""

    # Arrange column order
    df = df[
        ["patient_id"] +
        [f"path_{ph}" for ph in CFG.PHASES]  # C1, C2, C3, P
    ]
    df["label"] = "HCC"
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved external patient CSV: {out_csv}")
    print(f"[INFO] Total patients: {len(df)}")


if __name__ == "__main__":
    build_external_patient_df()
