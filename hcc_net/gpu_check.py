import torch

from config import CFG

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU only")

print("Number of available GPUs:", CFG.FOLD_IDX)