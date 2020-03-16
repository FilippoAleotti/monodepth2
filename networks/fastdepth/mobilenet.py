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
import networks.fastdepth.imagenet as imagenet

class Identity(nn.Module):
    # a dummy identity module
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, stride=2):
        super(Unpool, self).__init__()

        self.stride = stride

        # create kernel [1, 0; 0, 0]
        self.mask = torch.zeros(1, 1, stride, stride)
        self.mask[:,:,0,0] = 1

    def forward(self, x):
        assert x.dim() == 4
        num_channels = x.size(1)
        return F.conv_transpose2d(x,
            self.mask.detach().type_as(x).expand(num_channels, 1, -1, -1),
            stride=self.stride, groups=num_channels)

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


def pointwise(in_channels, out_channels, relu=True):
    layers = []
    layers.append(nn.Conv2d(in_channels,out_channels,1,1,0,bias=False))
    layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        #self.output_size = params['output_size']
        self.num_ch_enc = np.array([3,32,64,128,128,256,256,512,512,512,512,512,512,1024])
        pretrained = params['pretrained']
        mobilenet = imagenet.MobileNet()
        if pretrained:
            raise ValueError('Not available yet...')
            pretrained_path = os.path.join('imagenet', 'results', 'imagenet.arch=mobilenet.lr=0.1.bs=256', 'model_best.pth.tar')
            checkpoint = torch.load(pretrained_path)
            state_dict = checkpoint['state_dict']

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            mobilenet.load_state_dict(new_state_dict)
        else:
            mobilenet.apply(weights_init)

        for i in range(14):
            setattr( self, 'conv{}'.format(i), mobilenet.model[i])


    def forward(self, x):
        for i in range(14):
            layer = getattr(self, 'conv{}'.format(i))
            x = layer(x)
            if i==1:
                x1 = x
            elif i==3:
                x2 = x
            elif i==5:
                x3 = x
        features = {
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'x': x
        }
        return features

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        kernel_size = 5
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
        self.decode_conv6 = pointwise(32, 1, relu=False)
        self.sigmoid =  nn.Sigmoid()

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
        if self.params['supervised'] == False:
            x = self.sigmoid(x)

        self.outputs = {}
        assert self.params['scales'] == [0], 'MobileNet outputs a single depth! No multiple scales are allowed'
        self.outputs[("disp", 0)] = x
        return self.outputs


