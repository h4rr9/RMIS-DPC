# from https://github.com/TengdaHan/DPC/blob/master/eval/model_3d_lc.py

import math

from .backbone import select_resnet, ConvGRU

import torch
import torch.nn as nn
import torch.nn.functional as F


class DPC(nn.Module):

    def __init__(
        self,
        sample_size,
        num_seq=5,
        seq_len=5,
        network='resnet18',
        dropout=0.5,
    ):
        super(DPC, self).__init__()
        torch.cuda.manual_seed(666)
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        print('=> Using RNN + FC model ')

        print('=> Use 2D-3D %s!' % network)
        self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 32))
        track_running_stats = True

        self.backbone, self.param = select_resnet(
            network, track_running_stats=track_running_stats)
        self.param['num_layers'] = 1
        self.param['hidden_size'] = self.param['feature_size']

        print('=> using ConvRNN, kernel_size = 1')
        self.agg = ConvGRU(input_size=self.param['feature_size'],
                           hidden_size=self.param['hidden_size'],
                           kernel_size=1,
                           num_layers=self.param['num_layers'])
        self._initialize_weights(self.agg)

    def forward(self, block):
        # seq1: [B, N, C, SL, W, H]
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B * N, C, SL, H, W)
        feature = self.backbone(block)
        del block
        feature = F.relu(feature)

        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=1)
        feature = feature.view(B, N, self.param['feature_size'],
                               self.last_size,
                               self.last_size)  # [B*N,D,last_size,last_size]
        context, _ = self.agg(feature)
        context = context[:, -1, :].unsqueeze(1)

        return context

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet_3d.py


if __name__ == "__main__":

    mymodel = DPC(sample_size=128)
    x = torch.FloatTensor(4, 5, 3, 5, 128, 128)

    if torch.cuda.is_available():
        mymodel = mymodel.cuda()
        x = x.cuda()

    nn.init.normal_(x)
    __import__('ipdb').set_trace()
    c = mymodel(x)
    print(c.shape)
