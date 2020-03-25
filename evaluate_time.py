from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import math
import time
import matplotlib.pyplot as plt
from layers import disp_to_depth
from utils import readlines, count_parameters, color_map, create_dir
from options import MonodepthOptions
import datasets
from networks import factory

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
splits_dir = os.path.join(os.path.dirname(__file__), "splits")


class Result(object):
    def __init__(self):
        self.data_time, self.gpu_time = 0, 0
        self.max_fps = 0

    def set_to_worst(self):
        self.data_time, self.gpu_time = 0, 0
        self.max_fps = 0

    def update(self, gpu_time, data_time, max_fps):
        self.data_time, self.gpu_time = data_time, gpu_time
        self.max_fps = max_fps

    def evaluate(self, output, target):
        self.data_time = 0
        self.gpu_time = 0


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0
        self.sum_data_time, self.sum_gpu_time = 0, 0
        self.max_fps = 0

    def update(self, gpu_time, data_time, n):
        self.count += n
        self.sum_data_time += n*data_time
        self.sum_gpu_time += n*gpu_time
        curr_fps = 1.0 / gpu_time
        if curr_fps >= self.max_fps:
            self.max_fps = curr_fps

    def average(self):
        avg = Result()
        avg.update(self.sum_gpu_time / self.count, self.sum_data_time / self.count, self.max_fps)
        return avg


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    n=1
    average_meter = AverageMeter()
    do_warm_up = False
    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    encoder_dict = torch.load(encoder_path)

    dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                        encoder_dict['height'], encoder_dict['width'],
                                        [0], 4, is_train=False)
    dataloader = DataLoader(dataset, n, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)
    dataloader_warming_up = DataLoader(dataset, n, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)
    encoder_params = {
        'num_layers': opt.num_layers,
        'pretrained': False
    }
    encoder = factory.get_encoder(opt.architecture)(params=encoder_params)
    decoder_params = {
        'num_ch_enc': encoder.num_ch_enc,
        'supervised': False
    }
    depth_decoder = factory.get_decoder(opt.architecture)(params=decoder_params)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    print("-> Computing timings with size {}x{}".format(encoder_dict['width'], encoder_dict['height']))
    if do_warm_up:
        print("-> Warming up")
        with torch.no_grad():
            for data in dataloader_warming_up:
                input_color = data[("color", 0, 0)].cuda()
        print("-> Warming up ended! ")
    else:
        print("-> Deleating cache")
        torch.cuda.empty_cache()
    print("-> Starting timing evaluation")
    end = time.time()
    with torch.no_grad():
        for data in dataloader:
            end = time.time()
            input_color = data[("color", 0, 0)].cuda()
            data_time = time.time() - end
            end = time.time()
            output = depth_decoder(encoder(input_color))
            gpu_time = time.time() - end
            average_meter.update(gpu_time, data_time, n=n)

    avg = average_meter.average()
    print("\n  " + ("{:>8} | " * 5).format("ARCHITECTURE", "DATA-TIME","GPU-TIME", "MAX FPS", "MEAN FPS"))
    print(("{} &{: 8.3f}&{: 8.3f}&{: 8.3f}&{: 8.3f}").format(opt.architecture, avg.data_time * 1000., avg.gpu_time * 1000., avg.max_fps, 1/avg.gpu_time) + "\\\\")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
