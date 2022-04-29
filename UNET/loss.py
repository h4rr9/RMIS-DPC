import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class DICE_Loss(nn.Module):

    """
    Simple Implementation of DICE Loss
    
    """

    def __init__(self):
        super(DICE_Loss, self).__init__()
        self.dice_dims = (1,2,3)
    
    def forward(self, pred, target):

        # calculation of DICE loss
        target = target.squeeze(1)
        numerator = (pred * target).sum((1,2))
        denominator = (pred ** 2 + target ** 2).sum((1,2))

        # epsilon to avoid 0 div error in the denominator
        dice_score =  2 * numerator / (denominator + 1e-15)

        return (1 - dice_score).mean()



class IoU_Loss:
    """
    Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard
    
    From Ternaus paper 
    Cite properly at some point 
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            #eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = F.sigmoid(outputs)

            # I and U calculation
            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            # jaccard loss
            loss -= self.jaccard_weight * torch.log((intersection + 1e-15) / (union - intersection + 1e-15))
        return loss
