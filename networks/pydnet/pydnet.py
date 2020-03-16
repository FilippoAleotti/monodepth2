import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from networks.pydnet.modules import *
from networks.utils import get_size, value_or_default


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        channel_encoder_features    = [3,16,32,64,96,128,192]
        self.num_ch_enc = np.array([3,16,32,64,96,128,192])
        self.conv1 = self.encoder_block(channel_encoder_features[0], channel_encoder_features[1])
        self.conv2 = self.encoder_block(channel_encoder_features[1], channel_encoder_features[2])
        self.conv3 = self.encoder_block(channel_encoder_features[2], channel_encoder_features[3])
        self.conv4 = self.encoder_block(channel_encoder_features[3], channel_encoder_features[4])
        self.conv5 = self.encoder_block(channel_encoder_features[4], channel_encoder_features[5])
        self.conv6 = self.encoder_block(channel_encoder_features[5], channel_encoder_features[6])

        self.initialize()

    def encoder_block(self, channel_in, channel_out):
        '''
            Define an encoder block
        '''
        layers = []
        layers.append(conv2d_leaky(channel_in=channel_in,  channel_out=channel_out,  kernel_size=3, dilation=1, stride=2, relu=True))
        layers.append(conv2d_leaky(channel_in=channel_out, channel_out=channel_out,  kernel_size=3, dilation=1, stride=1, relu=True))
        return nn.Sequential(*layers)


    def forward(self,input_batch):
        encoder_features_1 = self.conv1(input_batch)
        encoder_features_2 = self.conv2(encoder_features_1)
        encoder_features_3 = self.conv3(encoder_features_2)
        encoder_features_4 = self.conv4(encoder_features_3)
        encoder_features_5 = self.conv5(encoder_features_4)
        encoder_features_6 = self.conv6(encoder_features_5)
        return [encoder_features_6, encoder_features_5, encoder_features_4, encoder_features_3, encoder_features_2, encoder_features_1]


    def initialize(self):
        ''' 
        Define how to initialize variables.
        Default: 
            - kaiming normal for conv2d,transpose2d and conv3d
            - 1 for BatchNorm 2d or 3d
            - 0 for Linear
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        channel_encoder_features    = [3,16,32,64,96,128,192]
        self.scales = value_or_default(params, 'scales', default= range(3,0,-1))
        num_ch = 8
        self.upsample_features_6 = bilinear_upsampling_by_convolution(num_ch, factor=2)
        self.upsample_features_5 = bilinear_upsampling_by_convolution(num_ch, factor=2)
        self.upsample_features_4 = bilinear_upsampling_by_convolution(num_ch, factor=2)
        self.upsample_features_3 = bilinear_upsampling_by_convolution(num_ch, factor=2)
        self.upsample_features_2 = bilinear_upsampling_by_convolution(num_ch, factor=2)

        self.estimator_6 = self.estimator(channel_encoder_features[6])            #H/64
        self.estimator_5 = self.estimator(channel_encoder_features[5] + num_ch)   #H/32
        self.estimator_4 = self.estimator(channel_encoder_features[4] + num_ch)   #H/16
        self.estimator_3 = self.estimator(channel_encoder_features[3] + num_ch)   #H/8
        self.estimator_2 = self.estimator(channel_encoder_features[2] + num_ch)   #H/4
        self.estimator_1 = self.estimator(channel_encoder_features[1] + num_ch)   #H/2

        self.get_disp_6 = self.get_disparity()
        self.get_disp_5 = self.get_disparity()
        self.get_disp_4 = self.get_disparity()
        self.get_disp_3 = self.get_disparity()
        self.get_disp_2 = self.get_disparity()
        self.get_disp_1 = self.get_disparity()

        self.initialize()

    def forward(self, encoder_features):
        encoder_features_6, encoder_features_5, encoder_features_4, encoder_features_3, encoder_features_2, encoder_features_1 = encoder_features

        estimator_6   = self.estimator_6(encoder_features_6)
        upsampled_features_6 = self.upsample_features_6(estimator_6)
        disp_6 = self.get_disp_6(estimator_6)

        features_5 = torch.cat((encoder_features_5, upsampled_features_6), 1)
        estimator_5   = self.estimator_5(features_5)
        upsampled_features_5 = self.upsample_features_5(estimator_5)
        disp_5 = self.get_disp_5(estimator_5)

        features_4 = torch.cat((encoder_features_4, upsampled_features_5), 1)
        estimator_4   = self.estimator_4(features_4)
        upsampled_features_4 = self.upsample_features_4(estimator_4)
        disp_4 = self.get_disp_4(estimator_4)

        features_3 = torch.cat((encoder_features_3, upsampled_features_4), 1)
        estimator_3 = self.estimator_3(features_3)
        upsampled_features_3 = self.upsample_features_3(estimator_3)
        disp_3 = self.get_disp_3(estimator_3)

        features_2 = torch.cat((encoder_features_2, upsampled_features_3), 1)
        estimator_2   = self.estimator_2(features_2)
        upsampled_features_2 = self.upsample_features_2(estimator_2)
        disp_2 = self.get_disp_2(estimator_2)

        features_1 = torch.cat((encoder_features_1, upsampled_features_2), 1)
        estimator_1 = self.estimator_1(features_1)
        disp_1 = self.get_disp_1(estimator_1)

        predictions = [None, disp_1, disp_2, disp_3, disp_4, disp_5, disp_6]
        self.outputs = {}

        for i in range(5, -1, -1):
            if i in self.scales:
                self.outputs[("disp", i)] = predictions[i]
        return self.outputs


    def get_current_size(self, previous_disp):
        '''
            Given previous disparity, get current disparity size [h,w]
        '''
        previous_shape = get_size(previous_disp)
        return [x*2 for x in previous_shape]


    def upsample_previous_prediction(self, prediction, desired_shape):
        '''
            Upsample previous prediction
        '''
        previous_upsampled = nn.functional.interpolate(prediction, size=desired_shape, mode='bilinear', align_corners=False)
        return previous_upsampled


    def estimator(self, channel_in):
        '''
            Shared estimator specification
        '''
        layers = []
        layers.append(conv2d_leaky(channel_in=channel_in,  channel_out=96,  kernel_size=3, dilation=1, stride=1, relu=True))
        layers.append(conv2d_leaky(channel_in=96,  channel_out=64,  kernel_size=3, dilation=1, stride=1, relu=True))
        layers.append(conv2d_leaky(channel_in=64,  channel_out=32,  kernel_size=3, dilation=1, stride=1, relu=True))
        layers.append(conv2d_leaky(channel_in=32,  channel_out=8,   kernel_size=3, dilation=1, stride=1, relu=True))
        return nn.Sequential(*layers)


    def get_disparity(self):
        layers = []
        layers.append(conv2d_leaky(channel_in=8,  channel_out=1,  kernel_size=3, dilation=1, stride=1, relu=False))
        if self.params['supervised'] == False:
            # NOTE: in case of unsupervised training,
            # apply sigmoid function as last activation
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)


    def initialize(self):
        ''' 
        Define how to initialize variables.
        Default: 
            - kaiming normal for conv2d,transpose2d and conv3d
            - 1 for BatchNorm 2d or 3d
            - 0 for Linear
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)