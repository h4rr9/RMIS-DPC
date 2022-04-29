import torch
from torch.nn import functional as F
import numpy as np


def dice_score_image(probs, target):
    '''
      compute the mean dice score for a single image

      Reminders: A false positive is a result that indicates a given condition exists, when it does not
               A false negative is a test result that indicates that a condition does not hold, while in fact it does
      Args:
          probs (tensor): probability scores of the mask being present at that pixel
          target (tensor): ground truth of the image
      Returns:
          m_dice (float): Mean dice score over classes
    '''
    ## TODO: Compute Dice Score for Each Class. Compute Mean Dice Score over Classes.

    # img_shape = (256, 320)
    # print(target.shape, prediction.shape)
    # assert(target.shape == img_shape + (n_classes,), f"got {target.shape} expected (8, 256, 320)")
    # assert(prediction.shape == (n_classes))
    
    # prediction shape is (1, 256, 320) argmax removes the dim=1, but batch size is still 1

    smooth = 1.
    target = target.squeeze(1)
    prediction = torch.empty(probs.shape)
    prediction[probs >= 0.5] = 1
    prediction[probs < 0.5] = 0
    prediction = prediction.cuda()

    TP = torch.sum(torch.mul(target == 1, prediction == 1)).item()
    FP = torch.sum(torch.mul(target == 0, prediction == 1)).item()
    FN = torch.sum(torch.mul(target == 1, prediction == 0)).item()

    if torch.sum(target) == 0:
        if FP == 0:
            return 1
        else:
            return 0

    dice = (2*TP + smooth) / (TP + FP + TP + FN + smooth)

    # give dice score of 1 if FP in that batch is 0, else 1
    # dice[(target.sum((1,2)) == 0)] = 0
    # dice[(target.sum((1,2)) == 0)*(FP == 0)] = 1

    # intersection = (prediction == target).float().sum((1,2))
    # dice = (2. * intersection + smooth)/(prediction.sum((1,2)) + target.sum((1,2)) + smooth)

    # return dice.mean()
    return dice


def iou_score_image(probs, target):
    '''
      computer the mean dice score for a single image

      Reminders: A false positive is a result that indicates a given condition exists, when it does not
               A false negative is a test result that indicates that a condition does not hold, while in fact it does
      Args:
          probs (tensor): probability scores of the mask being present at that pixel
          target (tensor): ground truth of the image
      Returns:
          m_dice (float): Mean dice score over classes
    '''
    smooth = 1
    target = target.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    prediction = torch.empty(probs.shape)
    prediction[probs >= 0.5] = 1
    prediction[probs < 0.5] = 0
    prediction = prediction.cuda()

    TP = torch.sum(torch.mul(target==0, prediction==1)).item()
    FP = torch.sum(torch.mul(target==0, prediction==1)).item()
    FN = torch.sum(torch.mul(target==1, prediction==0)).item()

    if torch.sum(target) == 0:
        if FP == 0:
            return 1
        else:
            return 0

    iou = (TP + smooth) / (TP + FP + FN + smooth)

    # boolTensor = (prediction == target).float()
    # intersection = boolTensor.sum((1, 2))
    # union = prediction.float().sum((1,2)) + target.float().sum((1,2)) - intersection
    
    # iou = (intersection + smooth) / (union + smooth)  # We smooth our devision to avoid 0/0
    
    # return iou.mean()

    return iou

def dice_score_dataset(model, dataset):
    """
    Compute the mean dice score on a set of data.
    
    Note that multiclass dice score can be defined as the mean over classes of binary
    dice score. Dice score is computed per image. Mean dice score over the dataset is the dice
    score averaged across all images.
    
    Reminders: A false positive is a result that indicates a given condition exists, when it does not
               A false negative is a test result that indicates that a condition does not hold, while in fact it does
     
    Args:
        model (UNET class): Your trained model
        dataset (DataLoader): Dataset for evaluation    
    Returns:
        m_dice (float): Mean dice score over the input dataset
    """
    ## Number of Batches and Cache over Dataset 
    scores = np.zeros(len(dataset))
    
    for idx, (img, target) in enumerate(dataset):
        ## Make Predictions
        prediction = model(img)
        scores[idx] = dice_score_image(prediction, target)
        
    ## Average Dice Score Over Images
    m_dice = scores.mean()
    return m_dice

def iou_score_dataset(model, dataset):
    """
    Compute the mean dice score on a set of data.
    
    Note that multiclass dice score can be defined as the mean over classes of binary
    dice score. Dice score is computed per image. Mean dice score over the dataset is the dice
    score averaged across all images.
    
    Reminders: A false positive is a result that indicates a given condition exists, when it does not
               A false negative is a test result that indicates that a condition does not hold, while in fact it does
     
    Args:
        model (UNET class): Your trained model
        dataloader (DataLoader): Dataset for evaluation
    
    Returns:
        m_dice (float): Mean dice score over the input dataset
    """
    ## Number of Batches and Cache over Dataset 
    scores = np.zeros(len(dataset))
    
    for idx, (img, target) in enumerate(dataset):
        ## Make Predictions
        prediction = model(img)
        scores[idx] = iou_score_image(prediction, target)
        
    ## Average Dice Score Over Images
    m_iou = scores.mean()
    return m_iou
