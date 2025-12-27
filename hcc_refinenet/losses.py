import torch
import torch.nn.functional as F
from monai.losses import DiceLoss

class BCEWithLogitsFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean', eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits).clamp(self.eps, 1 - self.eps)
        ce = F.binary_cross_entropy(p, targets.float(), reduction='none')
        pt = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

class DiceFocalCombo(torch.nn.Module):
    def __init__(self, dice_weight=1.0, focal_weight=1.0, focal_alpha=0.5, focal_gamma=2.0):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True, include_background=False, squared_pred=True)
        self.focal = BCEWithLogitsFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dw, self.fw = dice_weight, focal_weight

    def forward(self, logits, targets):
        return self.dw * self.dice(logits, targets) + self.fw * self.focal(logits, targets)
