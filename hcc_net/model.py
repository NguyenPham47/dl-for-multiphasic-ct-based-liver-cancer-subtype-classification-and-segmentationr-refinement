import timm, torch.nn as nn
from config import CFG

def in_chans_total():
    # one 2.5D stack per phase
    base = CFG.N_PHASES * CFG.SLICES_PER_PHASE
    if CFG.USE_LIVER_MASK and CFG.ADD_MASK_AS_CHANNEL:
        base += 1
    return base

class HCCNet(nn.Module):
    def __init__(self, model_name=CFG.MODEL_NAME, in_chans=in_chans_total(), num_classes=CFG.NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, in_chans=in_chans, num_classes=0)
        nf = self.backbone.num_features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(nf, 256),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        f = self.backbone.forward_features(x)
        return self.head(f)

def build_model():
    return HCCNet()
