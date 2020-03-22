'''
Original FastDepth: Fast Monocular Depth Estimation on Embedded Systems by Wofk et al.

Code from https://github.com/dwofk/fast-depth/, licensed under MIT license
'''

import os
import torch
import torch.nn as nn
import torchvision.models
import collections
import math
import numpy as np
import torch.nn.functional as F

class MobileNet(nn.Module):
    def __init__(self, relu6=True):
        super(MobileNet, self).__init__()

        def relu(relu6):
            if relu6:
                return nn.ReLU6(inplace=True)
            else:
                return nn.ReLU(inplace=True)

        def conv_bn(inp, oup, stride, relu6):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                relu(relu6),
            )

        def conv_dw(inp, oup, stride, relu6):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                relu(relu6),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                relu(relu6),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2, relu6), 
            conv_dw( 32,  64, 1, relu6),
            conv_dw( 64, 128, 2, relu6),
            conv_dw(128, 128, 1, relu6),
            conv_dw(128, 256, 2, relu6),
            conv_dw(256, 256, 1, relu6),
            conv_dw(256, 512, 2, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 1024, 2, relu6),
            conv_dw(1024, 1024, 1, relu6),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def weights_init(m):
    # Initialize kernel weights with Gaussian distributions
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,in_channels,kernel_size,stride=1,padding=padding,bias=False,groups=in_channels),
          nn.BatchNorm2d(in_channels),
          nn.ReLU(inplace=True),
        )


def pointwise(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        )


def final_pointwise(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.Sigmoid()
        )


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.num_ch_enc = np.array([32,64,128,128,256,256,512,512,512,512,512,512,1024, 1024])
        mobilenet = MobileNet()
        mobilenet.apply(weights_init)
        for i in range(14):
            setattr( self, 'conv{}'.format(i), mobilenet.model[i])

    def forward(self, x):
        for i in range(14):
            layer = getattr(self, 'conv{}'.format(i))
            x = layer(x)
            # print("{}: {}".format(i, x.size()))
            if i==1:
                x1 = x
            elif i==3:
                x2 = x
            elif i==5:
                x3 = x
        
        features = {
            'x': x,
            'x1': x1,
            'x2': x2,
            'x3': x3
        }
        return features


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        kernel_size = 5
        self.params = params
        self.decode_conv1 = nn.Sequential(
            depthwise(1024, kernel_size),
            pointwise(1024, 512))
        self.decode_conv2 = nn.Sequential(
            depthwise(512, kernel_size),
            pointwise(512, 256))
        self.decode_conv3 = nn.Sequential(
            depthwise(256, kernel_size),
            pointwise(256, 128))
        self.decode_conv4 = nn.Sequential(
            depthwise(128, kernel_size),
            pointwise(128, 64))
        self.decode_conv5 = nn.Sequential(
            depthwise(64, kernel_size),
            pointwise(64, 32))
        self.decode_conv6 = final_pointwise(32, 1)
        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)
        weights_init(self.decode_conv6)

    def forward(self, features):
        x = features['x']
        for i in range(1,6):
            layer = getattr(self, 'decode_conv{}'.format(i))
            x = layer(x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if i==4:
                x = x + features['x1']
            elif i==3:
                x = x + features['x2']
            elif i==2:
                x = x + features['x3']
        x = self.decode_conv6(x)
        self.outputs = {}
        assert self.params['scales'] == [0], 'MobileNet outputs a single depth! No multiple scales are allowed'
        self.outputs[("disp", 0)] = x
        return self.outputs
