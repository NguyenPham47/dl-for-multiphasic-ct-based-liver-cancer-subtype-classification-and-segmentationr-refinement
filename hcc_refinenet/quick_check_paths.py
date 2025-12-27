import pandas as pd, os
from config import CFG
from dataset import df_to_items  # uses _resolve_path under the hood

df = pd.read_csv(CFG().CSV_PATH)
items = df_to_items(df)  # applies _resolve_path to all path_* columns

cols = ["path_C1","path_C2","path_C3","path_mask_C1","path_mask_C2","path_mask_C3"]
counts = {c: {"exists":0, "missing":0, "none":0} for c in cols}

for it in items:
    for c in cols:
        p = it.get(c)
        if p is None:
            counts[c]["none"] += 1
        elif os.path.exists(p):
            counts[c]["exists"] += 1
        else:
            counts[c]["missing"] += 1

print(counts)

# also peek at a few resolved examples:
for k in range(min(3, len(items))):
    print({c: items[k].get(c) for c in cols})
