import pandas as pd
import matplotlib.pyplot as plt
from dataset import HCCDataset
from config import CFG

# Load the same CSV used for training
df = pd.read_csv(CFG.CSV_PATH)

# Initialize dataset (no augmentation)
ds = HCCDataset(df, training=False)

# Get one sample
sample = ds[0]  # ← change index to visualize different patients
arr = sample["image"].permute(1,2,0).cpu().numpy()  # CHW -> HWC

print("Image shape:", arr.shape)

# Plot each phase channel separately
fig, axs = plt.subplots(1, arr.shape[-1], figsize=(16,4))
for i in range(arr.shape[-1]):
    axs[i].imshow(arr[:,:,i], cmap="gray")
    axs[i].set_title(f"Phase {i}")
    axs[i].axis("off")

plt.tight_layout()
plt.show()
