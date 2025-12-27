import pandas as pd
import re
from config import CFG

def main():
    df = pd.read_csv(CFG.CSV_PATH)

    phases = ["C1", "C2", "C3", "P"]

    print("=== Checking for potential leakage in file paths ===")
    suspicious_words = ["HCC", "ICC", "CHCC"]

    for ph in phases:
        col = f"path_{ph}"
        if col not in df.columns:
            continue

        print(f"\n--- Phase {ph} ---")
        paths = df[col].astype(str)

        # Count occurrences of label words inside the path
        for word in suspicious_words:
            count = paths.str.upper().str.contains(word.upper()).sum()
            if count > 0:
                print(f"WARNING: Found {count} paths containing '{word}'")

        # Count unique folders by class
        df[f"folder_{ph}"] = paths.apply(lambda s: re.split(r"[\\/]", s)[0])

        print("Unique folders per class:")
        print(df.groupby("label")[f"folder_{ph}"].apply(lambda x: x.unique()))

if __name__ == "__main__":
    main()
