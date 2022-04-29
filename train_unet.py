from unet import iou_score_image
from unet import dice_score_image
from unet import iou_dice_score_image
from unet import UNet11
from unet import DICE_Loss

from dpc import AverageMeter

import unet.transform as T

from dataset import get_data

from dpc import save_checkpoint

import os
import re
import argparse
import time

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
    global args
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
            # T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            # T.RandomGray(consistent=False, p=0.5),
            # T.ColorJitter(brightness=0.5,
            #               contrast=0.5,
            #               saturation=0.5,
            #               hue=0.25,
            #               p=1.0),
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

        train_loss, train_dice, train_iou = train(train_loader, model,
                                                  criterion, optimizer, epoch,
                                                  writer_train, cuda)

        val_loss, val_dice, val_iou = validate(val_loader, model, criterion,
                                               epoch, writer_val, cuda)

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


def train(data_loader, model, loss_fn, optimizer, epoch, train_writer, cuda):
    losses = AverageMeter()
    dices = AverageMeter()
    ious = AverageMeter()
    # train the model
    model.train()

    global iteration

    for idx, (inputs, labels) in enumerate(data_loader):
        tic = time.time()

        # get inputs and labels

        model.train()

        inputs = inputs.to(cuda)
        labels = labels.to(cuda).squeeze(1)

        B, _, _ = labels.shape

        # compute predictions and loss
        pred = model(inputs).squeeze(1)
        loss = loss_fn(pred, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluate the model over training
        _iou, _dice = iou_dice_score_image(pred, labels)

        losses.update(loss.item(), B)
        dices.update(_dice, B)
        ious.update(_iou, B)

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f} ({loss.local_avg:.4f})\t'
                  'Dice: {3:.4f} IOU: {4:.4f} t:{5:.2f}\t'.format(
                      epoch,
                      idx,
                      len(data_loader),
                      _dice,
                      _iou,
                      time.time() - tic,
                      loss=losses))
            train_writer.add_scalar('local/loss', losses.val, iteration)
            train_writer.add_scalar('local/dice', dices.val, iteration)
            train_writer.add_scalar('local/iou', ious.val, iteration)
            iteration += 1

    return losses.local_avg, dices.local_avg, ious.local_avg


def validate(data_loader, model, loss_fn, epoch, writer, cuda):

    losses = AverageMeter()
    dices = AverageMeter()
    ious = AverageMeter()

    # evaluate the model
    model.eval()
    with torch.no_grad():

        for idx, (inputs, labels) in tqdm(enumerate(data_loader),
                                          total=len(data_loader)):

            inputs = inputs.to(cuda)
            labels = labels.to(cuda).squeeze(1)

            B, _, _ = labels.shape

            # compute predictions and loss
            pred = model(inputs).squeeze(1)
            loss = loss_fn(pred, labels)

            # epoch val loss
            _iou, _dice = iou_dice_score_image(pred, labels)

            losses.update(loss.item(), B)
            dices.update(_dice, B)
            ious.update(_iou, B)

    # per batch avg dice & iou
    print('[{0}/{1}] Loss {loss.local_avg:.4f}\t'
          'Dice:  {2:.4f}; IOU {3:.4f}\t'.format(epoch,
                                                 args.epochs,
                                                 dices.avg,
                                                 ious.avg,
                                                 loss=losses))

    return losses.local_avg, dices.local_avg, ious.local_avg


if __name__ == '__main__':
    main()
