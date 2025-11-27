import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss, TverskyLoss


class TotalEnergyClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, inputs, targets_class):
        total_energy_logits = torch.sum(inputs, dim=(2, 3))
        class_loss = self.cross_entropy(total_energy_logits, targets_class)
        
        return class_loss

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=0., lambda_seg = 1.0, lambda_bce=1.0, alpha=0.5, beta=0.5):
        super(DiceBCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = TverskyLoss(mode="multilabel", smooth=smooth, from_logits=True, alpha=alpha, beta=beta)
        self.lambda_bce = lambda_bce
        self.lambda_seg = lambda_seg
    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.lambda_bce * bce + self.lambda_seg * dice

class FullHybridLoss(nn.Module):
    def __init__(self, smooth=0., lambda_seg=1., lambda_bce=1., lambda_class=0.1, alpha=0.5, beta=0.5):
        super().__init__()
        self.dice_bce_loss = DiceBCELoss(smooth=smooth, lambda_seg=lambda_seg, lambda_bce=lambda_bce, alpha=alpha, beta=beta)
        self.class_loss = TotalEnergyClassificationLoss()
        self.lambda_class = lambda_class

    def forward(self, inputs, targets_mask, targets_class):
        segmentation_loss = self.dice_bce_loss(inputs, targets_mask)
        classification_loss = self.class_loss(inputs, targets_class)
        
        total_loss = segmentation_loss + self.lambda_class * classification_loss
        
        return total_loss