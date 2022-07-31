# @Author: zcy
# @Date:   2022-04-20 17:05:14
# @Last Modified by:   zcy
# @Last Modified time: 2022-06-15 17:03:30

import os

import imageio
import numpy as np
import torch
from mmcv.runner import get_dist_info
from mmcv.runner.hooks import HOOKS, Hook, CheckpointHook
import mmcv


@HOOKS.register_module()
class SaveDancePKLHook(Hook):

    def __init__(self, save_folder='validation'):
        self.save_folder = save_folder
        # pass

    def after_val_epoch(self, runner):
        rank, _ = get_dist_info()
        if rank == 0:
            cur_epoch = runner.epoch
            dance_poses = runner.outputs['output_pose']
            print(len(dance_poses), flush=True)


            store_dir = os.path.join(runner.work_dir, self.save_folder, 'epoch' + str(cur_epoch))
            os.makedirs(store_dir, exist_ok=True)

            for key in dance_poses:
                mmcv.dump(dance_poses[key].cpu().data.numpy(), os.path.join(store_dir, key + '.pkl'))

@HOOKS.register_module()
class SetValPipelineHook(Hook):
    """pass val dataset's pipeline to network."""
    def __init__(self, valset=None):
        self.val_pipeline = valset.pipeline

    def before_run(self, runner):  # only run once
        runner.model.module.set_val_pipeline(self.val_pipeline)
        del self.val_pipeline