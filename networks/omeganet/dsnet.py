import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from networks.utils import get_size, value_or_default


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        channel_encoder_features    = [16,32,64,128,256]
        self.num_ch_enc = np.array(channel_encoder_features)
        self.conv1 = self.encoder_block(3, channel_encoder_features[0]) # H/2
        self.conv2 = self.encoder_block(channel_encoder_features[0], channel_encoder_features[1]) # H/4
        self.conv3 = self.encoder_block(channel_encoder_features[1], channel_encoder_features[2]) # H/8
        self.conv4 = self.encoder_block(channel_encoder_features[2], channel_encoder_features[3]) # H/16
        self.conv5 = self.encoder_block(channel_encoder_features[3], channel_encoder_features[4]) # H/32

        self.initialize()

    def encoder_block(self, inp, out):
        '''
            Define an encoder block
        '''
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=inp, out_channels=out, kernel_size=3, stride=2),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=out, out_channels=out, kernel_size=3, stride=1),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )


    def forward(self,input_batch):
        encoder_features_0 = input_batch
        encoder_features_1 = self.conv1(input_batch)
        encoder_features_2 = self.conv2(encoder_features_1)
        encoder_features_3 = self.conv3(encoder_features_2)
        encoder_features_4 = self.conv4(encoder_features_3)
        encoder_features_5 = self.conv5(encoder_features_4)
        return [encoder_features_5, encoder_features_4, encoder_features_3, encoder_features_2, encoder_features_1]


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
        channel_encoder_features    = params['num_ch_enc']
        self.scales = value_or_default(params, 'scales', default= range(3,0,-1))
        num_ch = 16

        self.depth_upsampling_5 = self.depth_upsampling()
        self.depth_upsampling_4 = self.depth_upsampling()
        self.depth_upsampling_3 = self.depth_upsampling()
        self.depth_upsampling_2 = self.depth_upsampling()

        self.upsample_features_5 = self.feature_upsampling(num_ch)
        self.upsample_features_4 = self.feature_upsampling(num_ch)
        self.upsample_features_3 = self.feature_upsampling(num_ch)
        self.upsample_features_2 = self.feature_upsampling(num_ch)

        self.estimator_5 = self.estimator(channel_encoder_features[4])            #H/32
        self.estimator_4 = self.estimator(channel_encoder_features[3] + num_ch)   #H/16
        self.estimator_3 = self.estimator(channel_encoder_features[2] + num_ch)   #H/8
        self.estimator_2 = self.estimator(channel_encoder_features[1] + num_ch)   #H/4
        self.estimator_1 = self.estimator(channel_encoder_features[0] + num_ch)   #H/2

        self.get_disp_5 = self.get_disparity()
        self.get_disp_4 = self.get_disparity()
        self.get_disp_3 = self.get_disparity()
        self.get_disp_2 = self.get_disparity()
        self.get_disp_1 = self.get_disparity()

        self.initialize()

    def forward(self, encoder_features):
        encoder_features_5, encoder_features_4, encoder_features_3, encoder_features_2, encoder_features_1 = encoder_features

        estimator_5   = self.estimator_5(encoder_features_5)
        disp_5 = self.get_disp_5(estimator_5)
        up_disp_5 = self.depth_upsampling_5(disp_5)
        up_feat_5 = self.upsample_features_5(estimator_5)

        features_4 = torch.cat((encoder_features_4, up_feat_5), 1)
        estimator_4   = self.estimator_4(features_4)
        disp_4 = self.get_disp_4(estimator_4) + up_disp_5
        up_disp_4 = self.depth_upsampling_4(disp_4)
        up_feat_4 = self.upsample_features_4(estimator_4)

        features_3 = torch.cat((encoder_features_3, up_feat_4), 1)
        estimator_3   = self.estimator_3(features_3)
        disp_3 = self.get_disp_3(estimator_3) + up_disp_4
        up_disp_3 = self.depth_upsampling_3(disp_3)
        up_feat_3 = self.upsample_features_3(estimator_3)

        features_2 = torch.cat((encoder_features_2, up_feat_3), 1)
        estimator_2   = self.estimator_2(features_2)
        disp_2 = self.get_disp_3(estimator_2) + up_disp_3
        up_disp_2 = self.depth_upsampling_2(disp_2)
        up_feat_2 = self.upsample_features_2(estimator_2)

        features_1 = torch.cat((encoder_features_1, up_feat_2), 1)
        estimator_1   = self.estimator_1(features_1)
        disp_1 = self.get_disp_1(estimator_1) + up_disp_2

        predictions = [None, disp_1, disp_2, disp_3, disp_4, disp_5]
        self.outputs = {}

        for i in range(5, -1, -1):
            if i in self.scales:
                self.outputs[("disp", i)] = predictions[i]
        return self.outputs


    def feature_upsampling(self, inp):
        return nn.Sequential(
            Up(mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=3, stride=1),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True)
        )


    def depth_upsampling(self):
        return Up(mode='bilinear')


    def get_current_size(self, previous_disp):
        '''
            Given previous disparity, get current disparity size [h,w]
        '''
        previous_shape = get_size(previous_disp)
        return [x*2 for x in previous_shape]


    def conv_block(self, inp, out):
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=inp, out_channels=out, kernel_size=3, stride=1),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )


    def estimator(self, inp):
        '''
            Shared estimator specification
        '''
        return nn.Sequential(
            self.conv_block(inp, 64),
            self.conv_block(64, 48),
            self.conv_block(48, 32),
            self.conv_block(32, 16)
        )


    def get_disparity(self):
        return nn.Sequential(
            self.conv_block(16, 64),
            self.conv_block(64, 32),
            self.conv_block(32, 16),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1),
            nn.Sigmoid()
        )


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


class Up(nn.Module):
    def __init__(self, mode):
        super(Up, self).__init__()
        self.interp = nn.functional.interpolate
        self.mode = mode
        
    def forward(self, x):
        if self.mode == 'bilinear':
            x = self.interp(x, scale_factor=2, mode=self.mode, align_corners=False)
        else:
            x = self.interp(x, scale_factor=2, mode=self.mode)
        return x