import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG

class FocalLossMulti(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        # Register class weights as a buffer so .to(device) moves them with the module
        if weight is not None:
            w = torch.as_tensor(weight, dtype=torch.float32)
            self.register_buffer("weight", w)   # now lives on same device as the loss
        else:
            self.weight = None

    def forward(self, logits, targets):
        # logits: (B, C) on device; targets: (B,) on same device
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        # one-hot with label smoothing (on the same device as logits)
        with torch.no_grad():
            true = torch.zeros_like(logits)
            true.fill_(self.label_smoothing / (num_classes - 1))
            true.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)

        pt = (true * probs).sum(dim=1)             # p_t
        focal = (1 - pt).pow(self.gamma)           # (B,)

        # smoothed NLL
        nll = -(true * log_probs).sum(dim=1)       # (B,)

        if self.weight is not None:
            # self.weight is a buffer -> already on the same device
            cls_w = self.weight.gather(0, targets)  # (B,)
            nll = nll * cls_w
            focal = focal * cls_w

        loss = focal * nll
        return loss.mean()

def get_criterion(class_weights=None):
    # Pass raw list/np array; FocalLossMulti will register it as a buffer
    if CFG.USE_FOCAL:
        return FocalLossMulti(gamma=CFG.FOCAL_GAMMA,
                              weight=class_weights,
                              label_smoothing=CFG.LABEL_SMOOTHING)
    else:
        # CrossEntropyLoss accepts a weight Tensor; convert and let .to(device) move it.
        w = torch.as_tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        return nn.CrossEntropyLoss(weight=w, label_smoothing=CFG.LABEL_SMOOTHING)
