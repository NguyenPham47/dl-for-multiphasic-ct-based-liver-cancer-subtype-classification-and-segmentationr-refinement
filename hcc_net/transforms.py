import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import CFG

def _norm_channels():
    base = CFG.N_PHASES * CFG.SLICES_PER_PHASE
    if CFG.ADD_MASK_AS_CHANNEL and CFG.USE_LIVER_MASK:
        base += 1
    return (0.5,) * base, (0.25,) * base

def get_train_tfms():
    mean, std = _norm_channels()
    return A.Compose([
        A.LongestMaxSize(max_size=CFG.IMG_SIZE),
        A.PadIfNeeded(CFG.IMG_SIZE, CFG.IMG_SIZE, border_mode=0),
        A.HorizontalFlip(p=CFG.HFLIP_P),
        A.Rotate(limit=CFG.ROT_DEG, p=CFG.ROT_P, border_mode=0),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=0,
                           p=CFG.SHIFT_SCALE_ROT_P, border_mode=0),
        A.GridDistortion(num_steps=5, distort_limit=0.20, p=CFG.GRID_DISTORT_P),
        A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
        ToTensorV2(transpose_mask=False),
    ])

def get_valid_tfms():
    mean, std = _norm_channels()
    return A.Compose([
        A.LongestMaxSize(max_size=CFG.IMG_SIZE),
        A.PadIfNeeded(CFG.IMG_SIZE, CFG.IMG_SIZE, border_mode=0),
        A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
        ToTensorV2(transpose_mask=False),
    ])
