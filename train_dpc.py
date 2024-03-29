# modified from https://github.com/TengdaHan/DPC/blob/master/dpc/main.py

from dpc import save_checkpoint, denorm, calc_topk_accuracy
from dpc import DPC_RNN, AverageMeter
from dpc import ToTensor, Normalize
from dpc import RandomSizedCrop, RandomHorizontalFlip, RandomGray, ColorJitter

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
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from utils import utils

plt.switch_backend('agg')

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--model', default='dpc-rnn', type=str)
parser.add_argument('--dataset', default='rmis', type=str)
parser.add_argument('--data_path', required=True, type=str)
parser.add_argument('--seq_len',
                    default=5,
                    type=int,
                    help='number of frames in each video block')
parser.add_argument('--num_seq',
                    default=8,
                    type=int,
                    help='number of video blocks')
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


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    args = parser.parse_args()
    cuda = torch.device('cuda')

    if args.model == 'dpc-rnn':
        model = DPC_RNN(sample_size=args.img_dim,
                        num_seq=args.num_seq,
                        seq_len=args.seq_len,
                        network=args.net,
                        pred_step=args.pred_step)

    else:
        raise ValueError('wrong model')

    model = nn.DataParallel(model)
    model = model.to(cuda)

    criterion = nn.CrossEntropyLoss()

    if args.train_what == 'last':
        for name, param in model.module.resnet.named_parameters():
            param.requires_grad = False

    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    args.old_lr = None

    best_acc = 0
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
            model = utils.neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(
                args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    if args.dataset == 'rmis':
        transform = transforms.Compose([
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

    train_loader = get_data(return_video=True,
                            video_transforms=transform,
                            return_last_frame=False,
                            last_frame_transforms=None,
                            args=args,
                            mode='train')

    val_loader = get_data(return_video=True,
                          video_transforms=transform,
                          return_last_frame=False,
                          last_frame_transforms=None,
                          args=args,
                          mode='val')

    de_noramalize = denorm()
    img_path, model_path = utils.set_path(args)

    writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(img_path, 'val'))

    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_acc, train_accuracy_list = train(
            train_loader, model, de_noramalize, criterion, optimizer, epoch,
            args, writer_train, cuda)
        val_loss, val_acc, val_accuracy_list = validate(
            val_loader, model, criterion, epoch, args, cuda)

        # save curve
        writer_train.add_scalar('global/loss', train_loss, epoch)
        writer_train.add_scalar('global/accuracy', train_acc, epoch)
        writer_val.add_scalar('global/loss', val_loss, epoch)
        writer_val.add_scalar('global/accuracy', val_acc, epoch)
        writer_train.add_scalar('accuracy/top1', train_accuracy_list[0], epoch)
        writer_train.add_scalar('accuracy/top3', train_accuracy_list[1], epoch)
        writer_train.add_scalar('accuracy/top5', train_accuracy_list[2], epoch)
        writer_val.add_scalar('accuracy/top1', val_accuracy_list[0], epoch)
        writer_val.add_scalar('accuracy/top3', val_accuracy_list[1], epoch)
        writer_val.add_scalar('accuracy/top5', val_accuracy_list[2], epoch)

        # save check_point
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'net': args.net,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
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


def train(data_loader, model, de_normalize, criterion, optimizer, epoch, args,
          writer, device):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.train()
    global iteration

    for idx, input_seq in enumerate(data_loader):
        tic = time.time()
        input_seq = input_seq.to(device)
        B = input_seq.size(0)
        [score_, mask_] = model(input_seq)
        # visualize
        if (iteration == 0) or (iteration == args.print_freq):
            if B > 2:
                input_seq = input_seq[0:2, :]
            writer.add_image(
                'input_seq',
                de_normalize(
                    vutils.make_grid(input_seq.transpose(
                        2, 3).contiguous().view(-1, 3, args.img_dim,
                                                args.img_dim),
                                     nrow=args.num_seq * args.seq_len)),
                iteration)
        del input_seq

        if idx == 0:
            target_, (_, B2, NS, NP, SQ) = process_output(mask_)

        # score is a 6d tensor: [B, P, SQ, B2, N, SQ]
        # similarity matrix is computed inside each gpu, thus here B == num_gpu * B2
        score_flattened = score_.view(B * NP * SQ, B2 * NS * SQ)
        target_flattened = target_.contiguous().view(B * NP * SQ,
                                                     B2 * NS * SQ).to(device)
        target_flattened = target_flattened.to(int).argmax(dim=1)

        loss = criterion(score_flattened, target_flattened)
        top1, top3, top5 = calc_topk_accuracy(score_flattened,
                                              target_flattened, (1, 3, 5))

        accuracy_list[0].update(top1.item(), B)
        accuracy_list[1].update(top3.item(), B)
        accuracy_list[2].update(top5.item(), B)

        losses.update(loss.item(), B)
        accuracy.update(top1.item(), B)

        del score_

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del loss

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f} ({loss.local_avg:.4f})\t'
                  'Acc: top1 {3:.4f}; top3 {4:.4f}; top5 {5:.4f} T:{6:.2f}\t'.
                  format(epoch,
                         idx,
                         len(data_loader),
                         top1,
                         top3,
                         top5,
                         time.time() - tic,
                         loss=losses))

            writer.add_scalar('local/loss', losses.val, iteration)
            writer.add_scalar('local/accuracy', accuracy.val, iteration)

            iteration += 1

    return losses.local_avg, accuracy.local_avg, [
        i.local_avg for i in accuracy_list
    ]


def validate(data_loader, model, criterion, epoch, args, device):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.eval()

    with torch.no_grad():
        for idx, input_seq in tqdm(enumerate(data_loader),
                                   total=len(data_loader)):
            input_seq = input_seq.to(device)
            B = input_seq.size(0)
            [score_, mask_] = model(input_seq)
            del input_seq

            if idx == 0:
                target_, (_, B2, NS, NP, SQ) = process_output(mask_)

            # [B, P, SQ, B, N, SQ]
            score_flattened = score_.view(B * NP * SQ, B2 * NS * SQ)
            target_flattened = target_.contiguous().view(
                B * NP * SQ, B2 * NS * SQ).to(device)
            target_flattened = target_flattened.to(int).argmax(dim=1)

            loss = criterion(score_flattened, target_flattened)
            top1, top3, top5 = calc_topk_accuracy(score_flattened,
                                                  target_flattened, (1, 3, 5))

            losses.update(loss.item(), B)
            accuracy.update(top1.item(), B)

            accuracy_list[0].update(top1.item(), B)
            accuracy_list[1].update(top3.item(), B)
            accuracy_list[2].update(top5.item(), B)

    print('[{0}/{1}] Loss {loss.local_avg:.4f}\t'
          'Acc: top1 {2:.4f}; top3 {3:.4f}; top5 {4:.4f} \t'.format(
              epoch, args.epochs, *[i.avg for i in accuracy_list],
              loss=losses))
    return losses.local_avg, accuracy.local_avg, [
        i.local_avg for i in accuracy_list
    ]


if __name__ == '__main__':
    main()
