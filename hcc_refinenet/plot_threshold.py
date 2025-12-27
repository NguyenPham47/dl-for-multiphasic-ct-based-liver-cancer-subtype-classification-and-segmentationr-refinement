import json
import os
import re
import matplotlib.pyplot as plt

root = r"D:\HCC\TumorDetection\patient_tumor_out"

overlap_metrics = ["dice", "iou", "precision", "recall", "specificity"]
distance_metrics = ["hd95", "msd"]
all_metrics = overlap_metrics + distance_metrics

# Load data storage
data = {m: {"thresh": [], "mean": [], "std": []} for m in all_metrics}

# regex to parse folder name
pattern = re.compile(r"thresh_(\d+)_(\d+)")

# scan folders
for folder in os.listdir(root):
    match = pattern.match(folder)
    if not match:
        continue

    t = float(match.group(1) + "." + match.group(2))
    json_path = os.path.join(root, folder, f"eval_oof_thresh{match.group(1)}_{match.group(2)}.json")
    if not os.path.isfile(json_path):
        print("Missing:", json_path)
        continue

    with open(json_path, "r") as f:
        j = json.load(f)

    for m in all_metrics:
        data[m]["thresh"].append(t)
        data[m]["mean"].append(j[m]["mean"])
        data[m]["std"].append(j[m]["std"])

# sort everything by threshold
for m in all_metrics:
    idx = sorted(range(len(data[m]["thresh"])), key=lambda i: data[m]["thresh"][i])
    data[m]["thresh"] = [data[m]["thresh"][i] for i in idx]
    data[m]["mean"] = [data[m]["mean"][i] for i in idx]
    data[m]["std"] = [data[m]["std"][i] for i in idx]


# ---------- FIGURE 1: OVERLAP METRICS ----------
fig1, axes1 = plt.subplots(3, 2, figsize=(10, 12))
axes1 = axes1.flatten()

for i, m in enumerate(overlap_metrics):
    ax = axes1[i]
    ax.errorbar(data[m]["thresh"], data[m]["mean"], yerr=data[m]["std"],
                marker="o", capsize=4, linewidth=1.5)
    ax.set_title(m.upper(), fontsize=12)
    ax.set_xlabel("Threshold")
    ax.set_ylabel(m)
    ax.grid(True)

# Hide unused subplot (axes1[5])
axes1[-1].axis("off")

fig1.tight_layout()
fig1.savefig("overlap_metrics_subplots.pdf", bbox_inches="tight")  # MDPI-ready vector
# fig1.savefig("overlap_metrics_subplots.png", dpi=600, bbox_inches="tight")  # optional raster


# ---------- FIGURE 2: DISTANCE METRICS ----------
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))

for i, m in enumerate(distance_metrics):
    ax = axes2[i]
    ax.errorbar(data[m]["thresh"], data[m]["mean"], yerr=data[m]["std"],
                marker="o", capsize=4, linewidth=1.5)
    ax.set_title(m.upper(), fontsize=12)
    ax.set_xlabel("Threshold")
    ax.set_ylabel(m)
    ax.grid(True)

fig2.tight_layout()
fig2.savefig("distance_metrics_subplots.pdf", bbox_inches="tight")
# fig2.savefig("distance_metrics_subplots.png", dpi=600, bbox_inches="tight")

plt.close("all")
