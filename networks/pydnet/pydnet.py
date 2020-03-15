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

        last_channel_shared_decoder = 32
        self.conv1 = self.encoder_block(channel_encoder_features[0], channel_encoder_features[1])
        self.conv2 = self.encoder_block(channel_encoder_features[1], channel_encoder_features[2])
        self.conv3 = self.encoder_block(channel_encoder_features[2], channel_encoder_features[3])
        self.conv4 = self.encoder_block(channel_encoder_features[3], channel_encoder_features[4])
        self.conv5 = self.encoder_block(channel_encoder_features[4], channel_encoder_features[5])
        self.conv6 = self.encoder_block(channel_encoder_features[5], channel_encoder_features[6])


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
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                reset_bias(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                reset_bias(m)
            elif isinstance(m, nn.Linear):
                reset_bias(m)

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        channel_encoder_features    = [3,16,32,64,96,128,192]
        last_channel_shared_decoder = 32
        self.scales = value_or_default(params, 'scales', default= range(3,0,-1))
        
        self.upsample_features_6 = bilinear_upsampling_by_convolution(last_channel_shared_decoder, last_channel_shared_decoder, factor=2)
        self.upsample_features_5 = bilinear_upsampling_by_convolution(last_channel_shared_decoder, last_channel_shared_decoder, factor=2)
        self.upsample_features_4 = bilinear_upsampling_by_convolution(last_channel_shared_decoder, last_channel_shared_decoder, factor=2)
        self.upsample_features_3 = bilinear_upsampling_by_convolution(last_channel_shared_decoder, last_channel_shared_decoder, factor=2)
        self.upsample_features_2 = bilinear_upsampling_by_convolution(last_channel_shared_decoder, last_channel_shared_decoder, factor=2)

        channel_in = channel_encoder_features[6]
        self.shared_estimator_6 = self.shared_estimator(channel_in)   #H/64
        disparity_channels = last_channel_shared_decoder # 32 channels of shared encoder
        self.disparity_6 = self.disparity_estimator(channel_in=disparity_channels)

        channel_in = channel_encoder_features[5] + last_channel_shared_decoder
        self.shared_estimator_5 = self.shared_estimator(channel_in)   #H/32
        disparity_channels += 1 # added 1 previous disp
        self.disparity_5 = self.disparity_estimator(channel_in=disparity_channels)

        channel_in = channel_encoder_features[4] + last_channel_shared_decoder
        self.shared_estimator_4 = self.shared_estimator(channel_in)   #H/16
        disparity_channels += 1 
        self.disparity_4 = self.disparity_estimator(channel_in=disparity_channels)

        channel_in = channel_encoder_features[3] + last_channel_shared_decoder
        self.shared_estimator_3 = self.shared_estimator(channel_in)   #H/8
        disparity_channels += 1 
        self.disparity_3 = self.disparity_estimator(channel_in=disparity_channels)

        channel_in = channel_encoder_features[2] + last_channel_shared_decoder
        self.shared_estimator_2 = self.shared_estimator(channel_in)   #H/4
        disparity_channels += 1 
        self.disparity_2 = self.disparity_estimator(channel_in=disparity_channels)

        channel_in = channel_encoder_features[1] + last_channel_shared_decoder
        self.shared_estimator_1 = self.shared_estimator(channel_in)   #H/2
        disparity_channels += 1
        self.disparity_1 = self.disparity_estimator(channel_in=disparity_channels)


    def forward(self, encoder_features):
        encoder_features_6, encoder_features_5, encoder_features_4, encoder_features_3, encoder_features_2, encoder_features_1 = encoder_features
        previous_prediction = []
        shared_estimator_6   = self.shared_estimator_6(encoder_features_6)
        upsampled_features_6 = self.upsample_features_6(shared_estimator_6)
        disp_6 = self.disparity_6(shared_estimator_6)
        previous_prediction.append(disp_6)
        
        features_5 = torch.cat((encoder_features_5, upsampled_features_6),1)
        shared_estimator_5   = self.shared_estimator_5(features_5)
        upsampled_features_5 = self.upsample_features_5(shared_estimator_5)
        size_disp_5 = self.get_current_size(disp_6)
        previous_disp_upsampled = self.upsample_previous_predictions(previous_prediction, size_disp_5)
        disparity_features = torch.cat((shared_estimator_5, previous_disp_upsampled),1)
        disp_5 = self.disparity_5(disparity_features)
        previous_prediction.append(disp_5)

        features_4 = torch.cat((encoder_features_4, upsampled_features_5),1)
        shared_estimator_4   = self.shared_estimator_4(features_4)
        upsampled_features_4 = self.upsample_features_4(shared_estimator_4)
        size_disp_4 = self.get_current_size(disp_5)
        previous_disp_upsampled = self.upsample_previous_predictions(previous_prediction, size_disp_4)
        disparity_features = torch.cat((shared_estimator_4, previous_disp_upsampled),1)
        disp_4 = self.disparity_4(disparity_features)
        previous_prediction.append(disp_4)

        features_3 = torch.cat((encoder_features_3, upsampled_features_4),1)
        shared_estimator_3 = self.shared_estimator_3(features_3)
        upsampled_features_3 = self.upsample_features_3(shared_estimator_3)
        size_disp_3 = self.get_current_size(disp_4)
        previous_disp_upsampled = self.upsample_previous_predictions(previous_prediction, size_disp_3)
        disparity_features = torch.cat((shared_estimator_3, previous_disp_upsampled),1)
        disp_3 = self.disparity_3(disparity_features)
        previous_prediction.append(disp_3)

        features_2 = torch.cat((encoder_features_2, upsampled_features_3),1)
        shared_estimator_2   = self.shared_estimator_2(features_2)
        upsampled_features_2 = self.upsample_features_2(shared_estimator_2)
        size_disp_2 = self.get_current_size(disp_3)
        previous_disp_upsampled = self.upsample_previous_predictions(previous_prediction, size_disp_2)
        disparity_features = torch.cat((shared_estimator_2, previous_disp_upsampled),1)
        disp_2 = self.disparity_2(disparity_features)
        previous_prediction.append(disp_2)

        features_1 = torch.cat((encoder_features_1, upsampled_features_2),1)
        shared_estimator_1 = self.shared_estimator_1(features_1)
        size_disp_1 = self.get_current_size(disp_2)
        previous_disp_upsampled = self.upsample_previous_predictions(previous_prediction, size_disp_1)
        disparity_features = torch.cat((shared_estimator_1, previous_disp_upsampled),1)
        disp_1 = self.disparity_1(disparity_features)

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




    def upsample_previous_predictions(self, predictions, desired_shape):
        '''
            Given all the previous prediction, upsample it through bilinear and concat
            them together along channel dimension
        '''
        upsampled_prediction = []
        for previous_prediction in predictions:
            previous_upsampled = nn.functional.interpolate(previous_prediction, size=desired_shape, mode='bilinear', align_corners=False)
            upsampled_prediction.append(previous_upsampled)
        tensor = torch.cat(upsampled_prediction,1)
        return tensor


    def shared_estimator(self, channel_in):
        '''
            Shared estimator specification
        '''
        layers = []
        layers.append(conv2d_leaky(channel_in=channel_in,  channel_out=96,  kernel_size=3, dilation=1, stride=1, relu=True))
        layers.append(conv2d_leaky(channel_in=96,  channel_out=96,  kernel_size=3, dilation=1, stride=1, relu=True))
        layers.append(conv2d_leaky(channel_in=96,  channel_out=64,  kernel_size=3, dilation=1, stride=1, relu=True))
        layers.append(conv2d_leaky(channel_in=64,  channel_out=32,  kernel_size=3, dilation=1, stride=1, relu=True))
        return nn.Sequential(*layers)


    def disparity_estimator(self, channel_in):
        '''
            Estimate disparity
        '''
        layers = []
        layers.append(conv2d_leaky(channel_in=channel_in,  channel_out=32,  kernel_size=3, dilation=1, stride=1, relu=True))
        layers.append(conv2d_leaky(channel_in=32,  channel_out=8,   kernel_size=3, dilation=1, stride=1, relu=True))
        layers.append(conv2d_leaky(channel_in=8,   channel_out=1,   kernel_size=3, dilation=1, stride=1, relu=False))
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
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                reset_bias(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                reset_bias(m)
            elif isinstance(m, nn.Linear):
                reset_bias(m)