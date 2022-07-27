import time
import warnings

import mmcv
import torch
from mmcv.runner import EpochBasedRunner, IterBasedRunner
from mmcv.runner.iter_based_runner import IterLoader
from mmcv.runner.utils import get_host_info


class DanceTrainRunner(IterBasedRunner):
    """KiloNerfDistillTrainRunner Iter-based Runner.

    This runner uses iter_loaders as a member variable which will be changed in
    the distill cycle.
    """
    def run(self, data_loaders, workflow, max_iters=None, **kwargs):
        pass


class DanceTrainRunner(IterBasedRunner):
    pass


class DanceTestRunner(EpochBasedRunner):
    pass
