# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import torch
import torch.utils.data as data
from torchvision import transforms
from itertools import chain
import json
from datasets.nyu_dataset import NYUDataset



class DIODEDataset(NYUDataset):
    """ DIODE dataset
    """


    def check_and_tuplize_tokens(self, tokens, valid_tokens):
        if not isinstance(tokens, (tuple, list)):
            tokens = (tokens, )
        for split in tokens:
            assert split in valid_tokens
        return tokens


    def get_image_path(self, frame_id, line):
        f_str = "{}.png".format(line)
        image_path = os.path.join(self.data_path, f_str)
        return image_path


    def get_depth(self, frame_id, line, do_flip):
        depth_str = "{}_depth.npy".format(line)
        mask_str = "{}_depth_mask.npy".format(line)

        depth_path = os.path.join(self.data_path, depth_str)
        mask_path = os.path.join(self.data_path, mask_str)

        depth_gt = np.load(depth_path).squeeze()
        mask = np.load(mask_path)

        depth_gt *= mask

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt