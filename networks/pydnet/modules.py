import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d_leaky(channel_in, channel_out, kernel_size=3, stride=1, padding=1, dilation=1, relu=True):
    '''
        2D leaky relu convolution
    '''
    layers= [
        nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride=stride, 
                    padding=padding, dilation=dilation, bias=True),
    ]
    if relu is True:
        layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)

def bilinear_upsampling_by_convolution(channel_in, channel_out, factor=2):
    '''
        Upsampling the input using bilinear + conv by a desired factor
    '''
    return nn.Sequential(
        Bilinear(scale_factor=factor),
        conv2d_leaky(channel_in, channel_out)
    )

class Bilinear(nn.Module):
    ''' interpolate wrapper, as suggested in
        https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588
    '''
    def __init__(self, scale_factor=None, size=None):
        super(Bilinear, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = 'bilinear'
        
        if scale_factor is None and size is None:
            raise ValueError('size or factor must be defined in Bilinear')
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, size=self.size, mode=self.mode, align_corners=False)
        return x

def reset_bias(module):
    if module.bias is not None:
        module.bias.data.zero_()
    return module