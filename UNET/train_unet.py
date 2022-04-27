from UNET.eval import dice_score_dataset, iou_score_dataset
import torch
import numpy as np

# test
def test(test_dataloader, model, loss_fn, cuda):
    test_batches = len(test_dataloader)
    test_loss, test_IOU, test_dice = 0, 0, 0

    model.eval()
    with torch.no_grad():
        for data in enumerate(test_dataloader):
            # get inputs and labels
            inputs, labels = data
            
            inputs = inputs.to(cuda)
            labels = labels.to(cuda)
            
            # compute predictions and loss
            pred = model(inputs)
            loss = loss_fn(pred, labels)

            test_loss += loss.item()

            # evaluate the model over validation
            test_IOU += iou_score_dataset(pred, labels)
            test_dice += dice_score_dataset(pred, labels)
        
        # per batch avg dice & iou
        test_IOU = test_IOU/test_batches
        test_dice = test_dice/test_batches
        test_loss = test_loss/test_batches

        print("Test Loss: " + test_loss)
        print("Test DICE score: " + test_dice)
        print("Test IoU score: " + test_IOU)

        np.savetxt("Test_Metrics.csv", [test_IOU, test_dice, test_loss], delimiter =", ", fmt ='%1.9f')
        
        return test_loss, test_dice, test_IOU

        
def train(train_dataloader, model, loss_fn, optimizer, train_writer, iteration, cuda): 
    """
    
    """
    train_batches = len(train_dataloader)
    total_loss = []
    total_dice = []
    total_iou = []

    train_loss, train_IOU, train_dice = 0,0,0
    
    # train the model
    model.train()
    for data in enumerate(train_dataloader):
        # get inputs and labels

        inputs, labels = data
        
        inputs = inputs.to(cuda)
        labels = labels.to(cuda)
        
        # compute predictions and loss
        pred = model(inputs)
        loss = loss_fn(pred, labels)
        
        # epoch train loss
        train_loss += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluate the model over training
        train_IOU += iou_score_dataset(pred, labels)
        train_dice += dice_score_dataset(pred, labels)
        
        if i % 5 == 0:
            train_writer.add_scalar('local/loss', train_loss, iteration)
            train_writer.add_scalar('local/dice', train_dice, iteration)
            train_writer.add_scalar('local/iou', train_IOU, iteration)
            iteration += 1

    # per batch avg dice & iou
    train_IOU = train_IOU/train_batches
    train_dice = train_dice/train_batches
    train_loss = train_loss/train_batches

    print("Train Loss: " + train_loss)
    print("Train DICE score: " + train_dice)
    print("Train IoU score: " + train_IOU)

    total_loss.append(train_loss)
    total_dice.append(train_dice)
    total_iou.append(train_IOU)

    np.savetxt("Train_Loss.csv", train_loss, delimiter =", ", fmt ='%1.9f')
    np.savetxt("Train_DICE.csv", train_dice, delimiter =", ", fmt ='%1.9f')
    np.savetxt("Train_IoU.csv", train_IOU, delimiter =", ", fmt ='%1.9f')
    
    return train_loss, train_dice, train_IOU, iteration


def val(val_dataloader, model, loss_fn, val_writer, cuda):
    """
    
    """
    val_batches = len(val_dataloader)
    total_loss = []
    total_dice = []
    total_iou = []

    val_loss, val_IOU, val_dice = 0,0,0

    # evaluate the model
    model.eval()
    with torch.no_grad():

        for data in enumerate(val_dataloader, 0):
            # get inputs and labels
            inputs, labels = data

            inputs = inputs.to(cuda)
            labels = labels.to(cuda)
            
            # compute predictions and loss
            pred = model(inputs)
            loss = loss_fn(pred, labels)

            # epoch val loss
            val_loss += loss.item()

            # evaluate the model over validation
            val_IOU += iou_score_dataset(pred, labels)
            val_dice += dice_score_dataset(pred, labels)
            

    # per batch avg dice & iou
    val_IOU = val_IOU/val_batches
    val_dice = val_dice/val_batches
    val_loss = val_loss/val_batches

    print("Validation Loss: " + val_loss)
    print("Validation DICE score: " + val_dice)
    print("Validation IoU score: " + val_IOU)

    total_loss.append(val_loss)
    total_dice.append(val_dice)
    total_iou.append(val_IOU)

    np.savetxt("Val_Loss.csv", val_loss, delimiter =", ", fmt ='%1.9f')
    np.savetxt("Val_DICE.csv", val_dice, delimiter =", ", fmt ='%1.9f')
    np.savetxt("Val_IoU.csv", val_IOU, delimiter =", ", fmt ='%1.9f')
    
    return val_loss, val_dice, val_IOU
