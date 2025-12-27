# eval_metrics_cv.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import CFG
from train import get_fold
from eval_metrics import (
    evaluate,
    _bootstrap_ci,
    _compute_per_class_stats,
    _per_class_roc_pr,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def _plot_confusion_matrix(cm, classes, normalize, out_path, title):
    """
    Plot and save a confusion matrix (raw or normalized).
    """
    if normalize:
        cm = cm.astype(np.float32)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = cm / np.clip(row_sums, 1e-12, None)
    
    fig = plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_roc_curves(roc_data, out_path, title="ROC Curves"):
    fig = plt.figure(figsize=(6, 5))
    for cls, d in roc_data.items():
        fpr = np.array(d["fpr"])
        tpr = np.array(d["tpr"])
        roc_auc = d["auc"]
        if len(fpr) > 1 and len(tpr) > 1:
            plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_pr_curves(pr_data, out_path, title="Precision–Recall Curves"):
    fig = plt.figure(figsize=(6, 5))
    for cls, d in pr_data.items():
        prec = np.array(d["precision"])
        rec = np.array(d["recall"])
        pr_auc = d["auc"]
        if len(prec) > 1 and len(rec) > 1:
            plt.plot(rec, prec, label=f"{cls} (AUPRC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    df_full = pd.read_csv(CFG.CSV_PATH)

    all_acc = []
    all_f1 = []
    all_auc = []

    # Directory where fold checkpoints live (unchanged)
    base_ckpt_dir = os.path.dirname(CFG.BEST_CKPT)
    base_ckpt_name = "best_fold"  # assumes best_fold{fold_idx}.pt

    # Use model name in output path
    modelname = CFG.MODEL_NAME  # e.g., "efficientnet_b1", "convnext_small", etc.
    slices_per_phase = CFG.SLICES_PER_PHASE
    base_outdir = rf"D:\HCC\Naive_Model\{slices_per_phase}\classification_metrics\{modelname}"
    os.makedirs(base_outdir, exist_ok=True)

    fold_pred_paths = []

    # ----------------- Per-fold evaluation -----------------
    for fold_idx in range(CFG.N_FOLDS):
        print(f"\n=== Fold {fold_idx} ===")

        # split this fold
        _, va_df = get_fold(df_full, fold_idx, CFG.N_FOLDS)

        # save a temporary val CSV for this fold
        val_csv = os.path.join(base_outdir, f"val_fold{fold_idx}.csv")
        va_df.to_csv(val_csv, index=False)

        # checkpoint for this fold
        ckpt_path = os.path.join(base_ckpt_dir, f"{base_ckpt_name}{fold_idx}.pt")

        # outdir for this fold
        outdir = os.path.join(base_outdir, f"fold{fold_idx}")
        os.makedirs(outdir, exist_ok=True)

        # run existing evaluation (this writes predictions.csv inside outdir)
        metrics = evaluate(
            ckpt_path=ckpt_path,
            csv_path=val_csv,
            outdir=outdir,
            batch_size=None,
            num_workers=None,
            device=None,
        )

        acc = metrics["accuracy"]
        f1m = metrics["f1_macro"]
        auc = metrics["auc_macro_ovr"]

        all_acc.append(acc)
        all_f1.append(f1m)
        all_auc.append(auc)

        # path to this fold's predictions
        fold_pred_csv = os.path.join(outdir, "predictions.csv")
        if os.path.exists(fold_pred_csv):
            fold_pred_paths.append(fold_pred_csv)
        else:
            print(f"[WARN] predictions.csv not found for fold {fold_idx}: {fold_pred_csv}")

        print(f"Fold {fold_idx} | acc={acc:.4f} f1m={f1m:.4f} auc={auc:.4f}")

    # ----------------- Simple per-fold mean ± std -----------------
    def mean_std(x):
        x = np.array(x, dtype=float)
        return float(x.mean()), float(x.std())

    acc_mean, acc_std = mean_std(all_acc)
    f1_mean, f1_std = mean_std(all_f1)
    auc_mean, auc_std = mean_std(all_auc)

    print("\n=== Cross-validation summary over "
          f"{CFG.N_FOLDS} folds (patient-wise, per-fold avg) ===")
    print(f"Accuracy:   {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"Macro-F1:   {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Macro-AUC:  {auc_mean:.4f} ± {auc_std:.4f}")

    # ----------------- OOF aggregation -----------------
    if not fold_pred_paths:
        print("\n[ERROR] No predictions.csv files found; cannot compute OOF metrics.")
        return

    oof_list = [pd.read_csv(p) for p in fold_pred_paths]
    oof_df = pd.concat(oof_list, axis=0, ignore_index=True)

    print(f"\nOOF preds: {len(oof_df)} rows, "
          f"{oof_df['patient_id'].nunique()} unique patients.")

    # build y_true, y_prob
    y_true = oof_df["gt_idx"].astype(int).to_numpy()
    prob_cols = [f"prob_{c}" for c in CFG.CLASSES]
    y_prob = oof_df[prob_cols].to_numpy(dtype=float)
    y_pred = y_prob.argmax(axis=1)

    labels = list(range(len(CFG.CLASSES)))

    # ---- Core OOF metrics ----
    acc_oof = accuracy_score(y_true, y_pred)
    f1_macro_oof = f1_score(y_true, y_pred, average="macro")
    try:
        auc_macro_oof = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except Exception:
        auc_macro_oof = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=CFG.CLASSES,
        output_dict=True,
        zero_division=0,
    )

    per_class_stats, balanced_accuracy = _compute_per_class_stats(cm, cls_report)
    cis = _bootstrap_ci(y_true, y_prob, n_boot=1000, seed=CFG.SEED)
    roc_data, pr_data = _per_class_roc_pr(y_true, y_prob, CFG.CLASSES)

    print("\n=== OOF (out-of-fold) metrics over ALL patients ===")
    print(f"Accuracy (OOF):          {acc_oof:.4f}")
    print(f"Macro-F1 (OOF):          {f1_macro_oof:.4f}")
    print(f"Macro-AUC OvR (OOF):     {auc_macro_oof:.4f}")
    print(f"Balanced Acc (OOF):      {balanced_accuracy:.4f}")

    print("\nPer-class stats (OOF):")
    for cls_name, stats in per_class_stats.items():
        print(
            f"  {cls_name}: "
            f"prec={stats['precision']:.3f}, "
            f"sens/rec={stats['sensitivity']:.3f}, "
            f"spec={stats['specificity']:.3f}, "
            f"F1={stats['f1']:.3f}, "
            f"support={stats['support']}"
        )

    print("\n95% bootstrap CIs (OOF):")
    print(f"  Accuracy:   {cis['acc'][0]:.4f} – {cis['acc'][1]:.4f}")
    print(f"  Macro-F1:   {cis['f1_macro'][0]:.4f} – {cis['f1_macro'][1]:.4f}")
    print(f"  Macro-AUC:  {cis['auc_macro_ovr'][0]:.4f} – {cis['auc_macro_ovr'][1]:.4f}")

    # ----------------- Save OOF CSV + metrics JSON -----------------
    oof_pred_path = os.path.join(base_outdir, "oof_predictions.csv")
    oof_df.to_csv(oof_pred_path, index=False)

    oof_metrics = {
        "accuracy_oof": float(acc_oof),
        "f1_macro_oof": float(f1_macro_oof),
        "auc_macro_ovr_oof": float(auc_macro_oof),
        "balanced_accuracy_oof": float(balanced_accuracy),
        "confusion_matrix_oof": cm.tolist(),
        "classification_report_oof": cls_report,
        "ci_95_oof": {
            "accuracy": {
                "low": float(cis["acc"][0]),
                "high": float(cis["acc"][1]),
            },
            "f1_macro": {
                "low": float(cis["f1_macro"][0]),
                "high": float(cis["f1_macro"][1]),
            },
            "auc_macro_ovr": {
                "low": float(cis["auc_macro_ovr"][0]),
                "high": float(cis["auc_macro_ovr"][1]),
            },
        },
        "classes": CFG.CLASSES,
        "per_class_stats_oof": per_class_stats,
    }

    oof_metrics_path = os.path.join(base_outdir, "oof_metrics.json")
    with open(oof_metrics_path, "w", encoding="utf-8") as f:
        json.dump(oof_metrics, f, indent=2)

    # ----------------- Plot & save CM / ROC / PR (OOF) -----------------
    cm_path = os.path.join(base_outdir, "oof_confusion_matrix.png")
    cm_norm_path = os.path.join(base_outdir, "oof_confusion_matrix_normalized.png")
    _plot_confusion_matrix(
        cm, CFG.CLASSES, normalize=False,
        out_path=cm_path,
        title="Confusion Matrix (Counts)",
    )
    _plot_confusion_matrix(
        cm, CFG.CLASSES, normalize=True,
        out_path=cm_norm_path,
        title="Confusion Matrix (Row Normalized)",
    )

    roc_path = os.path.join(base_outdir, "oof_roc_curves.png")
    pr_path = os.path.join(base_outdir, "oof_pr_curves.png")
    _plot_roc_curves(roc_data, out_path=roc_path)
    _plot_pr_curves(pr_data, out_path=pr_path)

    print(f"\nSaved OOF predictions to:        {oof_pred_path}")
    print(f"Saved OOF metrics json to:       {oof_metrics_path}")
    print(f"Saved OOF confusion matrix to:   {cm_path}")
    print(f"Saved OOF norm. CM to:           {cm_norm_path}")
    print(f"Saved OOF ROC curves to:         {roc_path}")
    print(f"Saved OOF PR curves to:          {pr_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
