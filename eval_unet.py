from unet import iou_score_image
from unet import dice_score_image
from unet import iou_dice_score_image
from unet import UNet11
from unet import DICE_Loss

from dpc import AverageMeter

import unet.transform as T

from dataset import get_data

import os
import re
import argparse

import torch
from utils import utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='rmis', type=str)
parser.add_argument('--data_path', default='/mnt/disks/rmis_test/', type=str)
parser.add_argument('--load_weights',
                    default='',
                    type=str,
                    help='path of model to load')
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
parser.add_argument('--batch_size', default=15, type=int)
parser.add_argument('--num_classes', default=1, type=int)


def main():

    global args
    args = parser.parse_args()
    cuda = torch.device('cuda')

    model = UNet11(args.num_classes)
    model.to(cuda)

    criterion = DICE_Loss()

    if args.dataset == 'rmis':
        transform = T.Compose([T.ToTensor()])

    # load the saved weights
    if os.path.isfile(args.load_weights):
        args.old_lr = float(re.search('_lr(.+?)_', args.load_weights).group(1))
        print("=> loading weights of trained model '{}'".format(
            args.load_weights))
        checkpoint = torch.load(args.load_weights,
                                map_location=torch.device('cpu'))
        # args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # if not args.reset_lr:  # if didn't reset lr, load old optimizer
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded weights '{}' (epoch {})".format(
            args.load_weights, checkpoint['epoch']))
    else:
        print("[Warning] no weights found at '{}'".format(args.load_weights))

    test_loader = get_data(return_video=False,
                           video_transforms=None,
                           return_last_frame=True,
                           last_frame_transforms=transform,
                           args=args,
                           mode='test')

    losses = AverageMeter()
    dices = AverageMeter()
    ious = AverageMeter()

    # evaluate the model
    model.eval()
    with torch.no_grad():

        for idx, (inputs, labels) in enumerate(test_loader):

            inputs = inputs.to(cuda)
            labels = labels.to(cuda).squeeze(1)

            B, H, W = labels.shape

            # compute predictions and loss
            inputsLeft = inputs[..., :H]
            inputsRight = inputs[..., W - H:W]

            predLeft = model(inputsLeft).squeeze(1)
            predRight = model(inputsRight).squeeze(1)

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
