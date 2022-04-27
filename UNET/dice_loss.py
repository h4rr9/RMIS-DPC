import torch
from torch import nn
from torch.nn import functional as F
import utils
import numpy as np

def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
    if(torch.cuda.is_available()):
        one_hot=one_hot.cuda()
    target = one_hot.scatter_(1, labels.data, 1)
    return target

class DiceLoss(nn.Module):
    """ 
    DICE Loss Implementation but, much stolen from older code
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        
    def forward(self, inputs, targets, smooth=1):

        # pre-processing pipeline I stole from old code, not sure if this works
        if self.ignore_index not in range(targets.min(), targets.max()):
            if (targets == self.ignore_index).sum() > 0:
                targets[targets == self.ignore_index] = targets.min()
        targets = make_one_hot(targets.unsqueeze(dim=1), classes=input.size()[1])
        input = F.softmax(input, dim=1)
        input_flat = input.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # make 
        intersection = (input_flat * targets_flat).sum()
        dice = 1 - ((2. * intersection + self.smooth) / (input_flat.sum() + targets_flat.sum() + self.smooth))
        
        return dice
