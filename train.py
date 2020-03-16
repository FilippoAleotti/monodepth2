# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer_selfsupervised import Trainer as SELF_Trainer
from trainer_supervised import Trainer as SUP_Trainer

from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    if 'kitti' in opts.dataset:
        trainer = SELF_Trainer(opts)
    else:
        trainer = SUP_Trainer(opts)
    trainer.train()
