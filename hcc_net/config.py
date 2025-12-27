from dataclasses import dataclass
import os

@dataclass
class CFG:
    IMG_ROOT: str = r"D:\HCC"

    SEED: int = 2025

    CLASSES = ["HCC", "ICC", "CHCC"]
    NUM_CLASSES: int = 3

    # Phases expected in your CSV
    # PHASES = ["C1", "C2", "C3", "P"]     # matches your 'phase' column
    PHASES = ["C1", "C2", "C3", "P"]
    N_PHASES: int = len(PHASES)

    SLICES_PER_PHASE: int = 13  # number of slices to sample per phase, 5 for 2.5D, 13 for 3D

    # Mask usage
    USE_LIVER_MASK: bool = False          # if a liver mask path is available
    ADD_MASK_AS_CHANNEL: bool = False    # True => append mask as an extra channel
    CROP_TO_LIVER: bool = False           # True => crop tightly to liver bbox before resize

    IMG_SIZE: int = 384
    BATCH_SIZE: int = 12                 # slightly smaller because we have 4 channels
    EPOCHS: int = 30
    LR: float = 2e-4
    WEIGHT_DECAY: float = 1e-4
    LABEL_SMOOTHING: float = 0.05
    USE_FOCAL: bool = True
    FOCAL_GAMMA: float = 2.0
    MIXED_PRECISION: bool = True
    EARLY_STOP_PATIENCE: int = 6
    MODEL_NAME: str = "efficientnetv2_rw_s"
    FREEZE_BACKBONE_EPOCHS: int = 1
    COSINE_TMAX: int = 10
    NUM_WORKERS: int = 4

    # Splits
    N_FOLDS: int = 5
    FOLD_IDX: int = 3 # 0 1 2 3 4

    # Paths
    PHASE_CSV: str = r"D:\Downloads\patient_data.csv"  # your original CSV
    CSV_PATH: str = r"D:\HCC\patient_rows.csv"         # 1 row / patient (auto-built)
    CKPT_DIR: str = r"D:\HCC\checkpoints"
    # BEST_CKPT: str = os.path.join(r"D:\HCC\checkpoints", "best.pt")
    # BEST_CKPT: str = os.path.join(r"D:\HCC\efficientnet_b0", "best.pt")
    # BEST_CKPT: str = os.path.join(rf"D:\HCC\{MODEL_NAME}", f"best_fold{FOLD_IDX}.pt") # evaluate
    BEST_CKPT: str = os.path.join(rf"D:\HCC\Naive_Model\{SLICES_PER_PHASE}\{MODEL_NAME}", f"best_fold{FOLD_IDX}.pt") # evaluate

    # CT windowing (HU) — liver-friendly
    WINDOW_CENTER: int = 150 # 150
    WINDOW_WIDTH: int = 300  # 300 # 400

    # Augment knobs
    HFLIP_P: float = 0.5
    VFLIP_P: float = 0.0
    ROT_P: float = 0.35
    ROT_DEG: int = 10
    SHIFT_SCALE_ROT_P: float = 0.35
    GRID_DISTORT_P: float = 0.15
