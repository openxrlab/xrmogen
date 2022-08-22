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
        self.count = 0


    def before_val_epoch(self, runner):
        """
            prepare experiment folder
            experiments
        """
        self.dance_results = {}

    def after_val_iter(self, runner):
        rank, _ = get_dist_info()
        if rank == 0:
            dance_poses, dance_name = runner.outputs['output_pose'], runner.outputs['file_name']
            self.dance_results[dance_name] = dance_poses

    def after_val_epoch(self, runner):
        rank, _ = get_dist_info()
        if rank == 0:
            cur_epoch = runner.epoch

            print(len(self.dance_results), flush=True)


            store_dir = os.path.join(runner.work_dir, self.save_folder, 'epoch' + str(cur_epoch))
            os.makedirs(store_dir, exist_ok=True)

            for key in self.dance_results:
                mmcv.dump(self.dance_results[key].cpu().data.numpy(), os.path.join(store_dir, key + '.pkl'))

            # need to manually add 1 here 
            runner._epoch += 1

@HOOKS.register_module()
class SetValPipelineHook(Hook):
    """pass val dataset's pipeline to network."""
    def __init__(self, valset=None):
        self.val_pipeline = valset.pipeline

    def before_run(self, runner):  # only run once
        runner.model.module.set_val_pipeline(self.val_pipeline)
        del self.val_pipeline