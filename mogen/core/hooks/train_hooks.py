
import json
import os

import imageio
import numpy as np
import torch
from mmcv.runner import get_dist_info
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class PassEpochNumberToModelHook(Hook):
    """In test phase, calculate metrics over all testset.

    ndown: multiscales for mipnerf, set to 0 for others
    """
    def __init__(self,
                 ):

        pass 

    def before_train_epoch(self, runner):
        """
            prepare experiment folder
            experiments
        """
        runner.model.module._epoch = runner.epoch




