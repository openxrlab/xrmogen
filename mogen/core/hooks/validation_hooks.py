# @Author: zcy
# @Date:   2022-04-20 17:05:14
# @Last Modified by:   zcy
# @Last Modified time: 2022-06-15 17:03:30

import os

import imageio
import numpy as np
import torch
from mmcv.runner import get_dist_info
from mmcv.runner.hooks import HOOKS, Hook

from .utils import calculate_ssim, img2mse, mse2psnr, to8b



@HOOKS.register_module()
class SaveDancePKL(Hook):
    """save testset's render results with spiral poses 在每次val_step()之后调用
    用于保存test数据集的环型pose渲染图片 这些图片是没有groundtruth的 以视频方式保存."""
    def __init__(self, save_folder='validation'):
        # self.save_folder = save_folder
        pass

    def after_val_epoch(self, runner):
        pass
        # rank, _ = get_dist_info()
        # if rank == 0:
        #     cur_iter = runner.epoch
        #     dance_poses = np.stack(runner.outputs['spiral_rgbs'], 0)


        #     spiral_dir = os.path.join(runner.work_dir, self.save_folder)
        #     os.makedirs(spiral_dir, exist_ok=True)

            

        #     imageio.mimwrite(os.path.join(spiral_dir,
        #                                   '{}_rgb.mp4'.format(cur_iter)),
        #                      to8b(spiral_rgbs),
        #                      fps=30,
        #                      quality=8)
        #     imageio.mimwrite(os.path.join(spiral_dir,
        #                                   '{}_disp.mp4'.format(cur_iter)),
        #                      to8b(spiral_disps / np.max(spiral_disps)),
        #                      fps=30,
        #                      quality=8)

