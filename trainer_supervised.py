# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import json
from utils import *
from layers import *
import datasets
from networks import factory

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.architecture)
        self.models = {}
        self.parameters_to_train = []
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.num_scales = len(self.opt.scales)

        print('=> building depth encoder')
        encoder_params = {
            'num_layers': self.opt.num_layers,
            'pretrained': self.opt.weights_init == "pretrained"
        }
        print_params('encoder', encoder_params)
        self.models["encoder"] = factory.get_encoder(self.opt.architecture)(params=encoder_params)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        
        print('=> building depth decoder')
        decoder_params = {
            'num_ch_enc': self.models["encoder"].num_ch_enc,
            'scales': self.opt.scales,
            'supervised': True
        }
        print_params('decoder', decoder_params)
        self.models["depth"] = factory.get_decoder(self.opt.architecture)(params=decoder_params)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        num_params = (count_parameters(self.models["depth"]) + count_parameters(self.models['encoder'])) / 1000000
        print("=>  # Depth network parameters: {:.2f} M".format(num_params))


        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.architecture)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)
        run_tensorboard(logdir=self.opt.log_dir, port=self.opt.training_port)
        print('Tensorboard now running!')

        # data
        datasets_dict = {"diode": datasets.DIODEDataset,
                         "nyu": datasets.NYUDataset}
        print("dataset => "+ self.opt.dataset)
        self.dataset = datasets_dict[self.opt.dataset]
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        # NOTE: in pytorch 1.1.0 and later, first optimizer.step and then lr_scheduler.step
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, [inputs, targets] in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs, targets)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                self.compute_depth_losses(targets, outputs,  losses)
                self.log("train", inputs, outputs, targets, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs, gt):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        
        for key, ipt in gt.items():
            gt[key] = ipt.to(self.device)

        features = self.models["encoder"](inputs["color_aug"])
        outputs = self.models["depth"](features)

        losses = self.compute_losses(outputs, gt['inverse_depth'])
        return outputs, losses

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs, targets = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs, targets = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs, targets)
            
            self.compute_depth_losses(targets, outputs, losses)
            self.log("val", inputs, outputs, targets, losses)
            del inputs, outputs, losses

        self.set_train()

    def compute_losses(self, outputs, gt):
        """Compute the loss error using HuBer 
        """
        losses = {}
        total_loss = 0
        mask = gt > 0
        mask.detach_()

        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            loss = F.smooth_l1_loss(disp[mask], gt, reduction="mean")
            total_loss += loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, depth_gt, outputs, losses):
        """Compute depth metrics, to allow monitoring during training
        """
        depth_pred = outputs[("disp", 0)]
        depth_pred = depth_pred.detach().data
        gt = depth_gt['inverse_depth'].detach().data
        average_meter = AverageMeter()
        for i in range(gt.shape[0]):
            pred_i = depth_pred[i,:,:,:]
            gt_i = gt[i,:,:,:]
            mask_i = gt_i > 0
            result = Result()
            result.evaluate(pred_i[mask_i], 1/gt_i[mask_i])
            average_meter.update(result, pred_i.size(0))
            
        average=average_meter.average()
        losses = {
            'RMSE': average.rmse,
            'MAE': average.mae,
            'Delta1': average.delta1,
            'REL': average.absrel,
            'Lg10': average.lg10
        }
        return losses

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, targets, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                writer.add_image("color/{}".format(j), inputs[("color")][j].data, self.step)
                writer.add_image("color_aug/{}".format(j), inputs[("color_aug")][j].data, self.step)
                writer.add_image("disp_{}/{}".format(s, j), color_map(outputs[("disp", s)][j], cmap='jet'), self.step)
            writer.add_image("gt/{}".format(j), color_map(targets["inverse_depth"][j].data, cmap='jet'), self.step)
            writer.add_image("gt_mask/{}".format(j), (targets["inverse_depth"][j].data > 0)* 255, self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")