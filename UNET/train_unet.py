from torch import nn
import torch
from torch.nn import functional as F

# test
def test(model, test_dataloader):
    test_batches = len(test_dataloader)
    test_loss, test_iou, test_dice = 0, 0, 0

    model.eval()
    with torch.no_grad():
        for j, data in enumerate(test_dataloader, 0):
            # get inputs and labels
            inputs, labels = data
            
            # compute predictions and loss
            pred = model(inputs)
            loss = sigmoid_focal_loss(pred, labels)

def train(model,
          optimizer,
          lr,
          epochs,
          train_dataloader,
          val_dataloader,
          **kwargs):
    """
    
    """
    val_batches = len(val_dataloader)
    train_batches = len(train_dataloader)

    ##TODO: Implement a training loop
    for i in range(epochs):
        val_loss, val_IOU, val_dice = 0,0,0
        train_loss, train_IOU, train_dice = 0,0,0
        
        print("epoch: ", i)

        # train the model
        model.train()
        for j, data in enumerate(train_dataloader, 0):
            # get inputs and labels
            inputs, labels = data
            
            # compute predictions and loss
            pred = model(inputs)
            loss = sigmoid_focal_loss(pred, labels)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # evaluate the model
        model.eval()
        with torch.no_grad():
            for j, data in enumerate(val_dataloader, 0):
                # get inputs and labels
                inputs, labels = data
                
                # compute predictions and loss
                pred = model(inputs)
                loss = sigmoid_focal_loss(pred, labels)

