import pandas as pd
import json
import os

OUTPUT_DIR = r"D:\HCC\Naive_Model\external_validations"

def analyze(pred_csv):
    df = pd.read_csv(pred_csv)

    results = {}
    results["file"] = pred_csv
    results["total_patients"] = len(df)

    if "label" in df.columns:
        results["ground_truth_counts"] = df["label"].value_counts().to_dict()

    sens = (df["pred"] == "HCC").mean()
    results["HCC_sensitivity"] = sens

    results["predicted_counts"] = df["pred"].value_counts().to_dict()

    if "prob_HCC" in df.columns:
        results["prob_stats"] = df["prob_HCC"].describe().to_dict()

        high = (df["prob_HCC"] >= 0.9).mean()
        mid  = ((df["prob_HCC"] < 0.9) & (df["prob_HCC"] >= 0.6)).mean()
        low  = (df["prob_HCC"] < 0.6).mean()

        results["confidence_breakdown"] = {
            "high": high,
            "medium": mid,
            "low": low
        }

    # ---------------------------------------------------
    # SAVE JSON TO YOUR DESIRED DIRECTORY
    # ---------------------------------------------------
    base = os.path.splitext(os.path.basename(pred_csv))[0]
    json_path = os.path.join(OUTPUT_DIR, base + "_metrics.json")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved metrics to: {json_path}\n")


if __name__ == "__main__":
    preds = [
        # "external_predictions_2D.csv",
        # r"D:\HCC\Naive_Model\external_predictions_5slices.csv",
        r"D:\HCC\Naive_Model\external_predictions_13slices.csv",
    ]

    for p in preds:
        analyze(p)
