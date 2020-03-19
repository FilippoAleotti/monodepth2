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
import numbers
import random
import cv2
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image  # using pillow-simd for increased speed
from datasets.mono_dataset import pil_loader

class NYUDataset(data.Dataset):
    """ NYU dataset
    """
    def __init__(self, data_path,
                 filenames,
                 height,
                 width,
                 is_train=False,
                 img_ext='.png'):

        super(NYUDataset, self).__init__()
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.interp = Image.ANTIALIAS
        self.is_train = is_train
        self.img_ext = img_ext
        
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
        self.resize_img =  transforms.Resize((self.height, self.width), interpolation=self.interp)
        self.resize_depth =  cv2.resize

    def preprocess(self, rgb_img, depth_gt, color_aug):
        """Crop colour images and gt and augment if required
        """
        inputs = {}
        targets = {}
        h, w = depth_gt.squeeze().shape
        depth_gt = self.resize_depth(depth_gt, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        rgb_img = self.resize_img(rgb_img)
        gt = np.expand_dims(depth_gt, -1)
        inputs['color'] = self.to_tensor(rgb_img)
        inputs['color_aug'] = self.to_tensor(color_aug(rgb_img))
        targets['depth'] = self.to_tensor(gt)
        return inputs, targets

    def __len__(self):
        return len(self.filenames)

    def get_color(self, frame_id, line, do_flip):
        color = self.loader(self.get_image_path(frame_id, line))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>)          for raw colour images,
            ("color_aug", <frame_id>)      for augmented colour images,
            "depth"                              for ground truth depth maps.
        """
        inputs  = {}
        targets = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].strip()

        rgb_img = self.get_color(index, line, do_flip)
        depth_gt = self.get_depth(index, line, do_flip)
        
        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        inputs, targets = self.preprocess(rgb_img, depth_gt, color_aug)

        return inputs, targets

    def get_image_path(self, frame_id, line):
        f_str = "{:06d}{}".format(frame_id, self.img_ext)
        image_path = os.path.join(self.data_path, 'rgb', f_str)
        return image_path

    def get_depth(self, frame_id, line, do_flip):
        ''' NOTE: nyu depth are stored as npy files (no 16bit images) '''
        f_str = "{:06d}{}".format(frame_id, '.npy')
        image_path = os.path.join(self.data_path, 'depth', f_str)
        depth_gt = np.load(image_path)
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt