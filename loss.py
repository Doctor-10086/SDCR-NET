import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationLoss(nn.Module):

    def __init__(self, num_classes: int, multilabel: bool):
        super().__init__()
        self.multilabel = multilabel
        self.num_classes = num_classes
        self.cls_loss = nn.BCEWithLogitsLoss()

    def _to_bce_targets(self, targets: torch.Tensor, ref_logits: torch.Tensor):
        if targets.ndim == 2 and targets.shape[1] == self.num_classes:
            return targets.float()
        if self.multilabel:
            return targets.float()
        return F.one_hot(targets.long(), num_classes=self.num_classes).float().to(ref_logits.device)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        bce_targets = self._to_bce_targets(targets, logits)
        loss = self.cls_loss(logits, bce_targets)
        return loss
