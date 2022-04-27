import torch
from torch.nn import functional as F
import numpy as np


def dice_score_image(prediction, target, n_classes = 2):
    '''
      computer the mean dice score for a single image

      Reminders: A false positive is a result that indicates a given condition exists, when it does not
               A false negative is a test result that indicates that a condition does not hold, while in fact it does
      Args:
          prediction (tensor): predictied labels of the image
          target (tensor): ground truth of the image
          n_classes (int): number of classes
    
      Returns:
          m_dice (float): Mean dice score over classes
    '''
    ## TODO: Compute Dice Score for Each Class. Compute Mean Dice Score over Classes.

    # img_shape = (256, 320)
    # print(target.shape, prediction.shape)
    # assert(target.shape == img_shape + (n_classes,), f"got {target.shape} expected (8, 256, 320)")
    # assert(prediction.shape == (n_classes))
    
    # prediction shape is (1, 256, 320) argmax removes the dim=1, but batch size is still 1

    dice_classes = np.zeros(n_classes)
    for cl in range(n_classes):
        target_mask = target[:, cl, ...] > 0. # shape = (1, 256, 320)

        TP = (prediction[target_mask] == cl).float().sum().item()
        FP = (prediction[~target_mask] == cl).float().sum().item()
        FN = (prediction[target_mask] != cl).float().sum().item()
        #When there is no ground truth of the class in this image
        #Give 1 dice score if False Positive pixel number is 0, 
        #give 0 dice score if False Positive pixel number is not 0 (> 0).
        if target_mask.sum() == 0.:
            dice_classes[cl] = 0. if FP > 0 else 1.
        else:
            dice_classes[cl] = 2 * TP / ((TP + FP)+ (TP + FN) )
    return dice_classes.mean()


def iou_score_image(prediction, target, n_classes = 2):
    '''
      computer the mean dice score for a single image

      Reminders: A false positive is a result that indicates a given condition exists, when it does not
               A false negative is a test result that indicates that a condition does not hold, while in fact it does
      Args:
          prediction (tensor): predictied labels of the image
          target (tensor): ground truth of the image
          n_classes (int): number of classes
    
      Returns:
          m_dice (float): Mean dice score over classes
    '''
    ## TODO: Compute Dice Score for Each Class. Compute Mean Dice Score over Classes.

    # img_shape = (256, 320)
    # print(target.shape, prediction.shape)
    # assert(target.shape == img_shape + (n_classes,), f"got {target.shape} expected (8, 256, 320)")
    # assert(prediction.shape == (n_classes))
    
    # prediction shape is (1, 256, 320) argmax removes the dim=1, but batch size is still 1

    iou_classes = np.zeros(n_classes)
    for cl in range(n_classes):
        target_mask = target[:, cl, ...] > 0. # shape = (1, 256, 320)

        TP = (prediction[target_mask] == cl).float().sum().item()
        FP = (prediction[~target_mask] == cl).float().sum().item()
        FN = (prediction[target_mask] != cl).float().sum().item()
        #When there is no ground truth of the class in this image
        #Give 1 dice score if False Positive pixel number is 0, 
        #give 0 dice score if False Positive pixel number is not 0 (> 0).
        if target_mask.sum() == 0.:
            iou_classes[cl] = 0. if FP > 0 else 1.
        else:
            iou_classes[cl] = TP / ((TP + FP + FN) )
    return iou_classes.mean()


def dice_dataset(model, dataset ,num_classes = 2, use_gpu=False):
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
        num_classes (int): Number of classes
    
    Returns:
        m_dice (float): Mean dice score over the input dataset
    """
    ## Number of Batches and Cache over Dataset 
    n_batches = len(dataset)
    scores = np.zeros(n_batches)
    n_classes = num_classes

    ## Evaluate
    model.eval()
    idx = 0
    for data in dataset:
        ## Format Data
        img, target = data
        if use_gpu:
            img = img.cuda()
            target = target.cuda()
        ## Make Predictions
        out = model(img)
        n_classes = out.shape[1]
        prediction = torch.argmax(out, dim = 1)
        scores[idx] = dice_score_image(prediction, target, n_classes)
        idx += 1
        
    ## Average Dice Score Over Images
    m_dice = scores.mean()
    return m_dice

def iou_dataset(model, dataset ,num_classes = 2, use_gpu=False):
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
        num_classes (int): Number of classes
    
    Returns:
        m_dice (float): Mean dice score over the input dataset
    """
    ## Number of Batches and Cache over Dataset 
    n_batches = len(dataset)
    scores = np.zeros(n_batches)
    n_classes = num_classes

    ## Evaluate
    model.eval()
    idx = 0
    for data in dataset:
        ## Format Data
        img, target = data
        if use_gpu:
            img = img.cuda()
            target = target.cuda()
        ## Make Predictions
        out = model(img)
        n_classes = out.shape[1]
        prediction = torch.argmax(out, dim = 1)
        scores[idx] = iou_score_image(prediction, target, n_classes)
        idx += 1
        
    ## Average Dice Score Over Images
    m_iou = scores.mean()
    return m_iou
