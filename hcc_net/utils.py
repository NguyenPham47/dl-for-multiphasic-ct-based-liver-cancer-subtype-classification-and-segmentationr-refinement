import os, random, numpy as np, torch
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(y_true, y_prob):
    y_pred = y_prob.argmax(1)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except Exception:
        auc = float("nan")
    return acc, f1m, auc

class AverageMeter:
    def __init__(self): self.sum=0.0; self.count=0
    def update(self, v, n): self.sum += v*n; self.count += n
    @property
    def avg(self): return self.sum / max(1,self.count)
