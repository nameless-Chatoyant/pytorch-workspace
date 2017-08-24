import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from loss import FocalLoss
from cfgs.config import cfg

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.channels = cfg.channels
        self.channels.append(cfg.class_num)
        self.AtrousConvs = nn.ModuleList()
        for layer_idx, dilation in enumerate(cfg.dilations):
            AtrousConv = []
            if layer_idx == 0:
                ch_in = 3
            else:
                ch_in = self.channels[layer_idx - 1]
            AtrousConv.append(nn.Conv2d(in_channels = ch_in,
                            out_channels = self.channels[layer_idx],
                            kernel_size = cfg.kernel_size[layer_idx],
                            stride = 1,
                            padding = 0,
                            dilation = cfg.dilations[layer_idx],
                            bias = not cfg.with_bn))
            if cfg.with_bn: 
                AtrousConv.append(nn.BatchNorm1d(self.channels[layer_idx]))
            if layer_idx != len(cfg.dilations) - 1:
                AtrousConv.append(nn.ReLU())
            AtrousConv = nn.Sequential(*AtrousConv)
            self.AtrousConvs.append(AtrousConv)

    def forward(self, x):
        print('Input:', x.size())
        for conv_idx,AtrousConv in enumerate(self.AtrousConvs):
            x = F.pad(x, (cfg.dilations[conv_idx], cfg.dilations[conv_idx], cfg.dilations[conv_idx], cfg.dilations[conv_idx]))
            x = AtrousConv(x)
            print('Conv{}:'.format(conv_idx), x.size())
        return x

if __name__ == '__main__':
    net = Net()
    print(net)