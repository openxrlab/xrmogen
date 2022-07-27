# @Author: zcy
# @Date:   2022-04-20 17:05:14
# @Last Modified by:   zcy
# @Last Modified time: 2022-06-15 17:02:42

import json
import os

import imageio
import numpy as np
import torch
from mmcv.runner import get_dist_info
from mmcv.runner.hooks import HOOKS, Hook

from .utils import calculate_ssim, img2mse, mse2psnr, to8b


@HOOKS.register_module()
class TestHook(Hook):
    """In test phase, calculate metrics over all testset.

    ndown: multiscales for mipnerf, set to 0 for others
    """
    def __init__(self,
                 ndown=1,
                 save_img=False,
                 dump_json=False,
                 save_folder='test'):
        # self.ndown = ndown
        # self.dump_json = dump_json
        # self.save_img = save_img
        # self.save_folder = save_folde
        pass 

    def before_val_epoch(self, runner):
        pass

    def after_val_iter(self, runner):
        pass

    def after_val_epoch(self, runner):
        pass
