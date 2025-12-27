from monai.metrics import DiceMetric

class DiceMeter:
    def __init__(self):
        self.metric = DiceMetric(include_background=False, reduction="mean")

    def update(self, preds, gts):
        self.metric(y_pred=preds, y=gts)

    def compute(self):
        val = self.metric.aggregate().item()
        self.metric.reset()
        return float(val)
