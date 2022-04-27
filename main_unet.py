from UNET import unet11
from UNET import transform
from UNET import train_unet
from .main import get_data
from .main import set_path
from dpc import save_checkpoint
from dpc.utils import denorm
from dpc.backbone.resnet_2d3d import neq_load_customized

import torchvision.transforms as transforms

import os
import re
import time
import argparse
import numpy as np

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torchvision.utils as vutils

plt.switch_backend('agg')

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--model', deafult='unet11', type=str)
parser.add_argument('--dataset', default='rmis', type=str)
parser.add_argument('--data_path', required=True, type=str)

parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--ds',
                    default=3,
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
parser.add_argument('--num_classes', default=2, type=int)

def main():
    args = parser.parse_args()
    cuda = torch.device('cuda')
    
    model = unet11(args.num_classes)
    model.to(cuda)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
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
            best_acc = checkpoint['best_acc']
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
            model = neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(
                args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    # if args.dataset == 'rmis':
    #     transform = transforms.Compose([
    #         RandomSizedCrop(size=args.img_dim, consistent=True, p=1.0),
    #         RandomHorizontalFlip(consistent=True),
    #         RandomGray(consistent=False, p=0.5),
    #         ColorJitter(brightness=0.5,
    #                     contrast=0.5,
    #                     saturation=0.5,
    #                     hue=0.25,
    #                     p=1.0),
    #         ToTensor(),
    #         Normalize()
    #     ])

    # get training and val data
    train_loader = get_data(transform, args, 'train')
    val_loader = get_data(transform, args, 'val')

    de_noramalize = denorm()
    img_path, model_path = set_path(args)
    
    writer_train = SummaryWriter(logdir=os.path.join(img_path, 'train'))
    writer_val = SummaryWriter(logdir=os.path.join(img_path, 'val'))
    
    # start training
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_dice, train_iou = train(
            train_loader, model, criterion, optimizer, epoch,
            args, writer_train, cuda)
        
        val_loss, val_dice, val_iou = validate(
            val_loader, model, criterion, epoch, writer_val, cuda)

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
                # 'net': args.net,
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
