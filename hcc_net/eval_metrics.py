import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train import get_fold

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc, precision_recall_curve
)

# ---- project imports (must be in the same folder) ----
from config import CFG
from dataset import HCCDataset
from model import build_model


def _safe_auc_ovr(y_true, y_prob):
    """Macro AUC (OvR) with try/except guard."""
    try:
        return float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
    except Exception:
        return float('nan')


def _per_class_roc_pr(y_true, y_prob, class_names):
    """
    Compute ROC and PR curves per class.
    Returns dicts that can be used both for plotting and JSON dump if needed.
    """
    num_classes = len(class_names)
    y_true_bin = np.eye(num_classes)[y_true]  # (N, C)
    roc_data = {}
    pr_data = {}

    for i, name in enumerate(class_names):
        # ROC
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr) if len(fpr) > 1 else float('nan')
        except Exception:
            fpr, tpr, roc_auc = np.array([0, 1]), np.array([0, 1]), float('nan')
        roc_data[name] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(roc_auc)
        }

        # PR
        try:
            prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            pr_auc = auc(rec, prec) if len(rec) > 1 else float('nan')
        except Exception:
            prec, rec, pr_auc = np.array([1, 1]), np.array([0, 1]), float('nan')
        pr_data[name] = {
            'precision': prec.tolist(),
            'recall': rec.tolist(),
            'auc': float(pr_auc)
        }

    return roc_data, pr_data


def _bootstrap_ci(y_true, y_prob, n_boot=1000, seed=123):
    """
    95% CI for accuracy, macro-F1 and macro-AUC (OvR) via bootstrap
    (resample N with replacement).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, f1s, aucs = [], [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n, dtype=int)
        yt = y_true[idx]
        yp = y_prob[idx]
        yhat = yp.argmax(1)
        accs.append(accuracy_score(yt, yhat))
        f1s.append(f1_score(yt, yhat, average='macro'))
        try:
            aucs.append(roc_auc_score(yt, yp, multi_class='ovr'))
        except Exception:
            pass

    def ci(a):
        if not a:
            return (float('nan'), float('nan'))
        lo, hi = np.percentile(a, [2.5, 97.5])
        return float(lo), float(hi)

    return {
        'acc': ci(accs),
        'f1_macro': ci(f1s),
        'auc_macro_ovr': ci(aucs)
    }


def _compute_per_class_stats(cm, cls_report):
    """
    From confusion matrix + classification_report, compute:
    - per-class precision, recall, F1, support
    - per-class sensitivity (= recall)
    - per-class specificity
    - balanced accuracy (mean sensitivity)
    """
    num_classes = len(CFG.CLASSES)

    # Confusion-matrix-based stats
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1).astype(float) - tp
    fp = cm.sum(axis=0).astype(float) - tp
    tn = cm.sum().astype(float) - (tp + fp + fn)

    eps = 1e-12
    sensitivity = tp / (tp + fn + eps)        # recall
    specificity = tn / (tn + fp + eps)

    per_class_stats = {}
    for i, name in enumerate(CFG.CLASSES):
        d = cls_report.get(name, {})
        per_class_stats[name] = {
            "precision": float(d.get('precision', 0.0)),
            "recall": float(d.get('recall', 0.0)),       # same as sensitivity from cls_report
            "f1": float(d.get('f1-score', 0.0)),
            "support": int(d.get('support', 0)),
            "sensitivity": float(sensitivity[i]),
            "specificity": float(specificity[i]),
        }

    balanced_accuracy = float(np.mean(sensitivity))

    return per_class_stats, balanced_accuracy


@torch.no_grad()
def evaluate(ckpt_path, csv_path, outdir, batch_size=None, num_workers=None, device=None):
    """
    Evaluate a trained model checkpoint on the CSV defined by csv_path
    and save metrics/plots to outdir.

    Typical workflow (what you asked for):
      1) Set CFG.BEST_CKPT in config.py to the model you want.
      2) Run:
         python eval_metrics.py --outdir D:\\HCC\\metrics\\some_model_name

    You can optionally override --ckpt from the command line as well.
    """
    os.makedirs(outdir, exist_ok=True)
    batch_size = batch_size or CFG.BATCH_SIZE * 2
    num_workers = num_workers or CFG.NUM_WORKERS
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------- Data -----------------
    df = pd.read_csv(csv_path)
    ds = HCCDataset(df, training=False)  # use ALL rows in the CSV
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # ----------------- Model -----------------
    model = build_model().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()

    # ----------------- Predict -----------------
    probs_all, gts, pids = [], [], []
    for b in loader:
        x = b['image'].to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs_all.append(probs)
        gts += b['label'].cpu().tolist()
        pids += list(b['pid'])

    probs_all = np.concatenate(probs_all, axis=0)
    y_true = np.array(gts, dtype=int)
    y_pred = probs_all.argmax(1)

    labels = list(range(len(CFG.CLASSES)))

    # ----------------- Core metrics -----------------
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    # One F1 value per class, robust to missing classes using labels
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels)
    auc_macro = _safe_auc_ovr(y_true, probs_all)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=CFG.CLASSES,
        output_dict=True,
        zero_division=0
    )

    roc_data, pr_data = _per_class_roc_pr(y_true, probs_all, CFG.CLASSES)
    cis = _bootstrap_ci(y_true, probs_all, n_boot=1000, seed=CFG.SEED)
    per_class_stats, balanced_accuracy = _compute_per_class_stats(cm, cls_report)

    # ----------------- Save predictions table -----------------
    # Includes per-sample per-class probabilities.
    pred_df = pd.DataFrame({
        'patient_id': pids,
        'gt_idx': y_true,
        'gt_label': [CFG.CLASSES[i] for i in y_true],
        'pred_idx': y_pred,
        'pred_label': [CFG.CLASSES[i] for i in y_pred],
    })
    for i, name in enumerate(CFG.CLASSES):
        pred_df[f'prob_{name}'] = probs_all[:, i]
    pred_csv = os.path.join(outdir, 'predictions.csv')
    pred_df.to_csv(pred_csv, index=False)

    # ----------------- Save metrics JSON -----------------
    metrics = {
        'accuracy': float(acc),
        'f1_macro': float(f1_macro),
        'f1_per_class': {
            CFG.CLASSES[i]: float(v) for i, v in enumerate(f1_per_class)
        },
        'auc_macro_ovr': float(auc_macro),
        'balanced_accuracy': float(balanced_accuracy),
        'confusion_matrix': cm.tolist(),
        'classification_report': cls_report,
        'ci_95': {
            'accuracy': {
                'low': cis['acc'][0],
                'high': cis['acc'][1]
            },
            'f1_macro': {
                'low': cis['f1_macro'][0],
                'high': cis['f1_macro'][1]
            },
            'auc_macro_ovr': {
                'low': cis['auc_macro_ovr'][0],
                'high': cis['auc_macro_ovr'][1]
            },
        },
        'classes': CFG.CLASSES,
        'per_class_stats': per_class_stats,
    }

    metrics_path = os.path.join(outdir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # ----------------- Flat CSV for LaTeX tables -----------------
    rows = []
    rows.append({'metric': 'Accuracy', 'value': acc})
    rows.append({'metric': 'Macro-F1', 'value': f1_macro})
    rows.append({'metric': 'Macro-AUC (OvR)', 'value': auc_macro})
    rows.append({'metric': 'Balanced Accuracy', 'value': balanced_accuracy})

    # per-class stats
    for c in CFG.CLASSES:
        pc = per_class_stats[c]
        rows.append({'metric': f'Precision ({c})', 'value': pc['precision']})
        rows.append({'metric': f'Recall/Sensitivity ({c})', 'value': pc['sensitivity']})
        rows.append({'metric': f'Specificity ({c})', 'value': pc['specificity']})
        rows.append({'metric': f'F1 ({c})', 'value': pc['f1']})
        rows.append({'metric': f'Support ({c})', 'value': pc['support']})

    table_csv = os.path.join(outdir, 'metrics_table.csv')
    pd.DataFrame(rows).to_csv(table_csv, index=False)

    # ----------------- Confusion matrix plot -----------------
    fig = plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')  # blue-ish CM
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(range(len(CFG.CLASSES)), CFG.CLASSES, rotation=45, ha='right')
    plt.yticks(range(len(CFG.CLASSES)), CFG.CLASSES)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.tight_layout()
    cm_path = os.path.join(outdir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)

    # ----------------- ROC curves -----------------
    fig = plt.figure(figsize=(6, 5))
    for cls, d in roc_data.items():
        fpr = np.array(d['fpr'])
        tpr = np.array(d['tpr'])
        roc_auc = d['auc']
        if len(fpr) > 1 and len(tpr) > 1:
            plt.plot(fpr, tpr, label=f'{cls} (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (OvR)')
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(outdir, 'roc_curves.png')
    plt.savefig(roc_path, dpi=200)
    plt.close(fig)

    # ----------------- PR curves -----------------
    fig = plt.figure(figsize=(6, 5))
    for cls, d in pr_data.items():
        prec = np.array(d['precision'])
        rec = np.array(d['recall'])
        pr_auc = d['auc']
        if len(prec) > 1 and len(rec) > 1:
            plt.plot(rec, prec, label=f'{cls} (AUC={pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.tight_layout()
    pr_path = os.path.join(outdir, 'pr_curves.png')
    plt.savefig(pr_path, dpi=200)
    plt.close(fig)

    # ----------------- Stdout summary -----------------
    print('--- Evaluation summary ---')
    print(f'Accuracy:          {acc:.4f}')
    print(f'Macro-F1:          {f1_macro:.4f}')
    print(f'Macro-AUC OvR:     {auc_macro:.4f}')
    print(f'Balanced Acc:      {balanced_accuracy:.4f}')
    print('Per-class F1:', {CFG.CLASSES[i]: float(v) for i, v in enumerate(f1_per_class)})
    print(f'Saved predictions to:   {pred_csv}')
    print(f'Saved metrics json to:  {metrics_path}')
    print(f'Saved table to:         {table_csv}')
    print(f'Saved confusion matrix: {cm_path}')
    print(f'Saved ROC curves:       {roc_path}')
    print(f'Saved PR curves:        {pr_path}')

    return metrics


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--ckpt',
        type=str,
        default=CFG.BEST_CKPT,  # <--- this is what lets you just edit config.py
        help='Path to .pt checkpoint saved by train.py'
    )
    ap.add_argument(
        '--csv',
        type=str,
        default=CFG.CSV_PATH,
        help='CSV with patient rows used for evaluation'
    )
    ap.add_argument(
        '--outdir',
        type=str,
        default='paper_metrics',
        help='Directory to save outputs'
    )
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--num_workers', type=int, default=None)
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args.ckpt, args.csv, args.outdir, args.batch_size, args.num_workers)
