import os
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import init_dist

from xrmogen.models.builder import build_dance_models
from xrmogen.utils import get_root_logger
from .helper import build_dataloader, get_optimizer, get_runner, register_hooks


def train_mogen(cfg):
    """Train model entry function.

    Args:
        cfg (dict): The config dict for training.

    1. build dataloader
    2. optimizer
    3. build model
    4. build runnder
    """

    train_loader, trainset = build_dataloader(cfg, mode='train')
    val_loader, valset = build_dataloader(cfg, mode='val')
    dataloaders = [train_loader, val_loader]

    network = build_dance_models(cfg.model)

    optimizer = get_optimizer(network, cfg)

    if cfg.distributed:
        print('init_dist...', flush=True)
        init_dist('slurm', **cfg.get('dist_param', {}))
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        network = MMDistributedDataParallel(
            network.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        network = MMDataParallel(network.cuda(), device_ids=[0])

    Runner = get_runner(cfg.train_runner)
    runner = Runner(
        network,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=get_root_logger(log_level=cfg.log_level),
        meta=None)

    runner.timestamp = cfg.get('timestamp', None)

    # register hooks
    print('register hooks...', flush=True)

    # standard training hooks (default hooks of mmcv)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    # self-defined hooks
    register_hooks(cfg.train_hooks, **locals())

    # resume_from是载入ckpt和runner的训练信息，load_checkpoint只载入ckpt
    if cfg.get('resume_from', None):
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from', None) and os.path.exists(cfg.load_from):
        runner.load_checkpoint(cfg.load_from)
    runner_kwargs = dict()

    print('start train...', flush=True)
    runner.run(dataloaders, cfg.workflow, cfg.max_epochs, **runner_kwargs)
