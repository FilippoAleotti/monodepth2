# Mixed dataset

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

class MIXEDDataset(data.Dataset):
    """ MIXED dataset
    """
    def __init__(self, data_path,
                 filenames,
                 height,
                 width,
                 is_train=False,
                 img_ext='.png'):

        super(MIXEDDataset, self).__init__()
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.is_train = is_train
        self.img_ext = img_ext
        
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

    def preprocess(self, rgb_img, inv_depth_gt, color_aug):
        """Crop colour images and gt and augment if required
        """
        inputs = {}
        targets = {}
        rgb_img = np.asarray(rgb_img)
        inputs['color'] = self.to_tensor(rgb_img)
        inputs['color_aug'] = self.to_tensor(color_aug(rgb_img))
        targets['inverse_depth'] = self.to_tensor(inv_depth_gt)
        targets['depth'] = self.to_tensor(1.0/ (0.00001 + inv_depth_gt))
        targets['mask'] = self.to_tensor((inv_depth_gt > 0).astype(np.float32))

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

        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].strip()

        rgb_img = self.get_color(index, line, do_flip)
        depth_gt = self.get_depth(index, line, do_flip)

        # no color augmentation
        color_aug = (lambda x: x)

        inputs, targets = self.preprocess(rgb_img, depth_gt, color_aug)

        return inputs, targets

    def get_image_path(self, frame_id, line):
        f_str = "{}{}".format(line, self.img_ext)
        image_path = os.path.join(self.data_path, 'frames', f_str)
        return image_path

    def get_depth(self, frame_id, line, do_flip):
        ''' NOTE: gt inverse depth are stored in 16 bit png images '''
        f_str = "{}{}".format(line, '.png')
        image_path = os.path.join(self.data_path, 'depth', f_str)
        inv_depth_gt = cv2.imread(image_path, -1) / 256.0
        inv_depth_gt = np.expand_dims(inv_depth_gt, -1)
        if do_flip:
            inv_depth_gt = np.fliplr(inv_depth_gt)
        inv_depth_gt = inv_depth_gt.astype(np.float32)
        return inv_depth_gt
