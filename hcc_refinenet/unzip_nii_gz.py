import os
import gzip
import shutil

root = r"D:\HCC\CECT"  # <-- change this to your dataset root

# Walk through every subdirectory
for dirpath, _, filenames in os.walk(root):
    for f in filenames:
        if f.lower().endswith(".nii.gz"):
            src = os.path.join(dirpath, f)
            dst = os.path.join(dirpath, f[:-3])  # remove ".gz" -> .nii

            if os.path.exists(dst):
                print(f"[SKIP] already exists: {dst}")
                continue

            try:
                print(f"Extracting: {src} -> {dst}")
                with gzip.open(src, 'rb') as f_in:
                    with open(dst, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            except Exception as e:
                print(f"[ERROR] {src}: {e}")

print("✅ Done. All .nii.gz files extracted.")
