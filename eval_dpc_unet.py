from unet import iou_dice_score_image
from unet import UNet11
from unet import DICE_Loss

from dpc import AverageMeter
from dpc import RandomSizedCrop, ToTensor, Normalize
from dpc import DPC

import unet.transform as T

from dataset import get_data

import os
import re
import argparse

import torch
from utils import utils

import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='rmis', type=str)
parser.add_argument('--data_path', default='/mnt/disks/rmis_test/', type=str)
parser.add_argument('--dpc_model', default='dpc-rnn', type=str)
parser.add_argument('--unet_model', default='unet-11', type=str)
parser.add_argument('--load_weights',
                    default='',
                    type=str,
                    help='path of model to load')
parser.add_argument('--seq_len',
                    default=5,
                    type=int,
                    help='number of frames in each video block')
parser.add_argument('--num_seq',
                    default=5,
                    type=int,
                    help='number of video blocks')
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--ds',
                    default=3,
                    type=int,
                    help='frame downsampling rate')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--feature_dim', default=33, type=int)
parser.add_argument('--num_classes', default=1, type=int)
parser.add_argument('--net', default='resnet18', type=str)


def main():

    global args
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

    if args.dataset == 'rmis':
        dpc_transforms = transforms.Compose([
            RandomSizedCrop(size=args.img_dim, consistent=True, p=1.0),
            ToTensor(),
            Normalize()
        ])

        unet_transform = T.Compose([
            T.ToTensor(),
        ])

    # load the saved weights
    if args.load_weights:
        if os.path.isfile(args.load_weights):
            args.old_lr = float(
                re.search('_lr(.+?)_', args.load_weights).group(1))
            print("=> loading resumed checkpoint '{}'".format(
                args.load_weights))
            checkpoint = torch.load(args.load_weights,
                                    map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            dpc_model.load_state_dict(checkpoint['dpc_state_dict'])
            unet_model.load_state_dict(checkpoint['unet_state_dict'])
            print("=> loaded resumed checkpoint '{}' (epoch {})".format(
                args.load_weights, checkpoint['epoch']))
        else:
            print("[Warning] no checkpoint found at '{}'".format(args.resume))

    test_loader = get_data(return_video=True,
                           video_transforms=dpc_transforms,
                           return_last_frame=True,
                           last_frame_transforms=unet_transform,
                           args=args,
                           mode='test')

    losses = AverageMeter()
    dices = AverageMeter()
    ious = AverageMeter()

    # evaluate the model
    dpc_model.eval()
    unet_model.eval()
    with torch.no_grad():

        for idx, (input_seq, img, labels) in enumerate(test_loader):

            input_seq = input_seq.to(cuda)
            img = img.to(cuda)
            labels = labels.to(cuda).squeeze(1)

            B, H, W = labels.shape

            # compute predictions and loss
            inputsLeft = img[..., :H]
            inputsRight = img[..., W - H:W]

            dpc_features = dpc_model(input_seq)

            predLeft = unet_model(inputsLeft, dpc_features).squeeze(1)
            predRight = unet_model(inputsRight, dpc_features).squeeze(1)

            pred = utils.create_full_mask(predLeft, predRight)

            loss = criterion(pred, labels)

            # epoch val loss
            _iou, _dice = iou_dice_score_image(pred, labels)

            losses.update(loss.item(), B)
            dices.update(_dice, B)
            ious.update(_iou, B)

    # per batch avg dice & iou
    print('[Loss {loss.local_avg:.4f}\t'
          'Dice:  {0:.4f}; IOU {1:.4f}\t'.format(dices.avg,
                                                 ious.avg,
                                                 loss=losses))

    # return losses.local_avg, dices.local_avg, ious.local_avg


if __name__ == '__main__':
    main()
