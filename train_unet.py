from unet import iou_score_image
from unet import dice_score_image
from unet import UNet11
from unet import DICE_Loss

import unet.transform as T

from dataset import get_data

from dpc import save_checkpoint

import os
import re
import argparse

import numpy as np

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from utils import utils

from tqdm import tqdm

plt.switch_backend('agg')

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='vgg11', type=str)
parser.add_argument('--model', default='unet11', type=str)
parser.add_argument('--dataset', default='rmis', type=str)
parser.add_argument('--data_path', default='/mnt/disks/rmis_train/', type=str)
parser.add_argument('--seq_len',
                    default=0,
                    type=int,
                    help='number of frames in each video block')
parser.add_argument('--num_seq',
                    default=0,
                    type=int,
                    help='number of video blocks')
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--ds',
                    default=0,
                    type=int,
                    help='frame downsampling rate')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    help='path of model to resume')
parser.add_argument('--pretrain',
                    default='',
                    type=str,
                    help='path of pretrained model')
parser.add_argument('--epochs',
                    default=10,
                    type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--print_freq',
                    default=5,
                    type=int,
                    help='frequency of printing output during training')
parser.add_argument('--reset_lr',
                    action='store_true',
                    help='Reset learning rate when resume training?')
parser.add_argument('--prefix',
                    default='tmp',
                    type=str,
                    help='prefix of checkpoint filename')
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--num_classes', default=1, type=int)


def main():
    args = parser.parse_args()
    cuda = torch.device('cuda')

    model = UNet11(args.num_classes)
    model.to(cuda)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.wd)
    criterion = DICE_Loss()
    args.old_lr = None

    best_dice = 0
    global iteration
    iteration = 0

    if args.resume:
        if os.path.isfile(args.resume):
            args.old_lr = float(re.search('_lr(.+?)_', args.resume).group(1))
            print("=> loading resumed checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume,
                                    map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            best_dice = checkpoint['best_dice']
            model.load_state_dict(checkpoint['state_dict'])
            if not args.reset_lr:  # if didn't reset lr, load old optimizer
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('==== Change lr from %f to %f ====' %
                      (args.old_lr, args.lr))
            print("=> loaded resumed checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("[Warning] no checkpoint found at '{}'".format(args.resume))

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(
                args.pretrain))
            checkpoint = torch.load(args.pretrain,
                                    map_location=torch.device('cpu'))
            model = utils.neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(
                args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    if args.dataset == 'rmis':
        transform = T.Compose([
            T.RandomSplit(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            # T.RandomGray(consistent=False, p=0.5),
            T.ColorJitter(brightness=0.5,
                          contrast=0.5,
                          saturation=0.5,
                          hue=0.25,
                          p=1.0),
            T.ToTensor(),
        ])

    # get training and val data
    train_loader = get_data(return_video=False,
                            video_transforms=None,
                            return_last_frame=True,
                            last_frame_transforms=transform,
                            args=args,
                            mode='train')
    val_loader = get_data(return_video=False,
                          video_transforms=None,
                          return_last_frame=True,
                          last_frame_transforms=transform,
                          args=args,
                          mode='val')

    print("loader:", len(train_loader))

    # de_noramalize = denorm()
    img_path, model_path = utils.set_path(args)

    writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(img_path, 'val'))

    # start training
    for epoch in range(args.start_epoch, args.epochs):
        print('\nEpoch {:d}\n'.format(epoch))

        train_loss, train_dice, train_iou, iteration = train(
            train_loader, model, criterion, optimizer, writer_train, iteration,
            cuda)

        val_loss, val_dice, val_iou = validate(val_loader, model, criterion,
                                               writer_val, cuda)

        # save curve
        writer_train.add_scalar('global/loss', train_loss, epoch)
        writer_train.add_scalar('global/dice', train_dice, epoch)
        writer_train.add_scalar('global/iou', train_iou, epoch)
        writer_val.add_scalar('global/loss', val_loss, epoch)
        writer_val.add_scalar('global/dice', val_dice, epoch)
        writer_val.add_scalar('global/iou', val_iou, epoch)

        # save check_point
        is_best = val_dice > best_dice
        best_dice = max(val_dice, best_dice)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'net': args.net,
                'state_dict': model.state_dict(),
                'best_dice': best_dice,
                'optimizer': optimizer.state_dict(),
                'iteration': iteration
            },
            is_best,
            filename=os.path.join(model_path,
                                  'epoch%s.pth.tar' % str(epoch + 1)),
            keep_all=False)

    print('Training from ep %d to ep %d finished' %
          (args.start_epoch, args.epochs))


# test
def test(test_dataloader, model, loss_fn, cuda):
    test_batches = len(test_dataloader)
    test_loss, test_IOU, test_dice = 0, 0, 0

    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_dataloader):
            # get inputs and labels

            inputs = inputs.to(cuda)
            labels = labels.to(cuda)

            # compute predictions and loss
            pred = model(inputs)
            loss = loss_fn(pred, labels)

            test_loss += loss.item()

            # evaluate the model over validation
            test_IOU += iou_score_image(pred, labels)
            test_dice += dice_score_image(pred, labels)

        # per batch avg dice & iou
        test_IOU = test_IOU / test_batches
        test_dice = test_dice / test_batches
        test_loss = test_loss / test_batches

        print("Test Loss: ", test_loss)
        print("Test DICE score: ", test_dice)
        print("Test IoU score: ", test_IOU)

        np.savetxt("Test_Metrics.csv", [test_IOU, test_dice, test_loss],
                   delimiter=", ",
                   fmt='%1.9f')

        return test_loss, test_dice, test_IOU


def train(train_dataloader, model, loss_fn, optimizer, train_writer, iteration,
          cuda):
    train_batches = len(train_dataloader)
    total_loss = []
    total_dice = []
    total_iou = []

    train_loss, train_IOU, train_dice = 0, 0, 0

    # train the model
    model.train()
    for i, (inputs, labels) in tqdm(enumerate(train_dataloader)):

        # get inputs and labels

        model.train()

        # print(inputs.shape)
        # print(type(inputs))
        # print('\n')
        # print(labels.shape)
        # print(type(labels))

        inputs = inputs.to(cuda)
        labels = labels.to(cuda)

        # compute predictions and loss
        pred = model(inputs).squeeze(1)

        # print("pred minimum val: ", torch.min(pred))
        # print("pred max val: ", torch.min(pred))
        # print("preds: ", pred)
        # print("target:", labels)

        loss = loss_fn(pred, labels)

        # epoch train loss
        train_loss += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluate the model over training
        train_IOU += iou_score_image(pred, labels)
        train_dice += dice_score_image(pred, labels)

        if i % 5 == 0:
            train_writer.add_scalar('local/loss', train_loss, iteration)
            train_writer.add_scalar('local/dice', train_dice, iteration)
            train_writer.add_scalar('local/iou', train_IOU, iteration)
            iteration += 1

    # per batch avg dice & iou
    train_IOU = train_IOU / train_batches
    train_dice = train_dice / train_batches
    train_loss = train_loss / train_batches

    print("Train Loss: {:.3f} ".format(train_loss))
    print("Train DICE score: {:.3f}".format(train_dice))
    print("Train IoU score: {:.3f}\n".format(train_IOU))

    total_loss.append(train_loss)
    total_dice.append(train_dice)
    total_iou.append(train_IOU)

    return train_loss, train_dice, train_IOU, iteration


def validate(val_dataloader, model, loss_fn, val_writer, cuda):
    val_batches = len(val_dataloader)
    total_loss = []
    total_dice = []
    total_iou = []

    val_loss, val_IOU, val_dice = 0, 0, 0

    # evaluate the model
    model.eval()
    with torch.no_grad():

        for i, (inputs, labels) in tqdm(enumerate(val_dataloader, 0)):

            inputs = inputs.to(cuda)
            labels = labels.to(cuda)

            # compute predictions and loss
            pred = model(inputs).squeeze(1)
            loss = loss_fn(pred, labels)

            # epoch val loss
            val_loss += loss.item()

            # evaluate the model over validation
            val_IOU += iou_score_image(pred, labels)
            val_dice += dice_score_image(pred, labels)

    # per batch avg dice & iou
    val_IOU = val_IOU / val_batches
    val_dice = val_dice / val_batches
    val_loss = val_loss / val_batches

    print("Validation loss: {:.3f}".format(val_loss))
    print("Validation dice: {:.3f}".format(val_dice))
    print("Validation iou: {:.3f}\n".format(val_IOU))

    total_loss.append(val_loss)
    total_dice.append(val_dice)
    total_iou.append(val_IOU)

    return val_loss, val_dice, val_IOU


if __name__ == '__main__':
    main()
