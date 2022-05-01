# modified from https://github.com/TengdaHan/DPC/blob/master/dpc/main.py

from dpc import DPC, AverageMeter
from dpc import ToTensor, Normalize
from dpc import RandomSizedCrop, RandomHorizontalFlip, RandomGray, ColorJitter

from unet import UNet11
from unet import iou_dice_score_image
from unet import DICE_Loss
import unet.transform as T

from dataset import get_data

import torchvision.transforms as transforms

import os
import re
import time
import argparse
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.optim as optim

from utils import utils

plt.switch_backend('agg')

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--dpc_model', default='dpc-rnn', type=str)
parser.add_argument('--unet_model', default='unet-11', type=str)
parser.add_argument('--dataset', default='rmis', type=str)
parser.add_argument('--data_path', required=True, type=str)
parser.add_argument('--seq_len',
                    default=5,
                    type=int,
                    help='number of frames in each video block')
parser.add_argument('--num_seq',
                    default=5,
                    type=int,
                    help='number of video blocks')
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
parser.add_argument('--feature_dim', default=33, type=int)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    args = parser.parse_args()
    cuda = torch.device('cuda')

    if args.dpc_model == 'dpc-rnn':
        dpc_model = DPC(sample_size=args.img_dim,
                        num_seq=args.num_seq,
                        seq_len=args.seq_len,
                        network=args.net,
                        upsample_size=args.feature_dim)
    else:
        raise ValueError('wrong model')

    if args.unet_model == 'unet-11':
        unet_model = UNet11(num_classes=args.num_classes)
    else:
        raise ValueError('wrong model')

    dpc_model = dpc_model.to(cuda)
    unet_model = unet_model.to(cuda)

    criterion = DICE_Loss()

    print('\n===========Check DPC Grad============')
    for name, param in dpc_model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    print('\n===========Check Unet Grad============')
    for name, param in unet_model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    params = dpc_model.parameters() + unet_model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
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
            best_dice = checkpoint['best_acc']
            dpc_model.load_state_dict(checkpoint['dpc_state_dict'])
            unet_model.load_state_dict(checkpoint['unet_state_dict'])
            if not args.reset_lr:  # if didn't reset lr, load old optimizer
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('==== Change lr from %f to %f ====' %
                      (args.old_lr, args.lr))
            print("=> loaded resumed checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("[Warning] no checkpoint found at '{}'".format(args.resume))

    if args.dataset == 'rmis':
        dpc_transform = transforms.Compose([
            RandomSizedCrop(size=args.img_dim, consistent=True, p=1.0),
            RandomHorizontalFlip(consistent=True),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5,
                        contrast=0.5,
                        saturation=0.5,
                        hue=0.25,
                        p=1.0),
            ToTensor(),
            Normalize()
        ])

        unet_transform = T.Compose([
            T.RandomSplit(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
        ])

    train_loader = get_data(return_video=True,
                            video_transforms=dpc_transform,
                            return_last_frame=True,
                            last_frame_transforms=unet_transform,
                            args=args,
                            mode='train')

    val_loader = get_data(return_video=True,
                          video_transforms=dpc_transform,
                          return_last_frame=True,
                          last_frame_transforms=unet_transform,
                          args=args,
                          mode='val')

    de_noramalize = denorm()
    img_path, model_path = utils.set_path(args)

    writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(img_path, 'val'))

    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_dice, train_iou = train(train_loader, dpc_model,
                                                  unet_model, de_noramalize,
                                                  criterion, optimizer, epoch,
                                                  args, writer_train, cuda)
        val_loss, val_dice, val_iou = validate(val_loader, dpc_model,
                                               unet_model, criterion, epoch,
                                               args, cuda)

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
                'dpc_state_dict': dpc_model.state_dict(),
                'unet_state_dict': unet_model.state_dict(),
                'best_acc': best_dice,
                'optimizer': optimizer.state_dict(),
                'iteration': iteration
            },
            is_best,
            filename=os.path.join(model_path,
                                  'epoch%s.pth.tar' % str(epoch + 1)),
            keep_all=False)

    print('Training from ep %d to ep %d finished' %
          (args.start_epoch, args.epochs))


def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size()  # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)


def train(data_loader, dpc_model, unet_model, de_normalize, criterion,
          optimizer, epoch, args, writer, device):
    losses = AverageMeter()
    dices = AverageMeter()
    ious = AverageMeter()

    dpc_model.train()
    unet_model.train()
    global iteration

    for idx, (input_seq, img, target) in enumerate(data_loader):
        tic = time.time()

        B = input_seq.size(0)

        input_seq = input_seq.to(device)
        img = img.to(device)
        target = target.to(device).squeeze(1)

        dpc_features = dpc_model(input_seq)
        # visualize
        del input_seq

        pred = unet_model(img, dpc_features).squeeze(1)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _iou, _dice = iou_dice_score_image(pred, target)

        losses.update(loss.item(), B)
        dices.update(_dice, B)
        ious.update(_iou, B)

        del loss

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f} ({loss.local_avg:.4f})\t'
                  'Dice: {3:.4f} IOU: {4:.4f} T:{5:.2f}\t'.format(
                      epoch,
                      idx,
                      len(data_loader),
                      _dice,
                      _iou,
                      time.time() - tic,
                      loss=losses))
            writer.add_scalar('local/loss', losses.val, iteration)
            writer.add_scalar('local/dice', dices.val, iteration)
            writer.add_scalar('local/iou', ious.val, iteration)
            iteration += 1

            iteration += 1

    return losses.local_avg, dices.local_avg, ious.local_avg


def validate(data_loader, dpc_model, unet_model, criterion, epoch, args,
             device):
    dices = AverageMeter()
    losses = AverageMeter()
    ious = AverageMeter()

    dpc_model.eval()
    unet_model.eval()

    with torch.no_grad():
        for idx, (input_seq, img, target) in tqdm(enumerate(data_loader),
                                                  total=len(data_loader)):
            input_seq = input_seq.to(device)
            img = img.to(device)
            target = target.to(device).squeeze(1)
            B = input_seq.size(0)
            dpc_features = dpc_model(input_seq)
            del input_seq

            pred = unet_model(img, dpc_features).squeeze(1)
            loss = criterion(pred, target)

            # epoch val loss
            _iou, _dice = iou_dice_score_image(pred, target)

            losses.update(loss.item(), B)
            dices.update(_dice, B)
            ious.update(_iou, B)

            # [B, P, SQ, B, N, SQ]

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
