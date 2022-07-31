from mmcv import Config
from .test import test_mogen
from .train import train_mogen

__all__ = ['run_mogen']


def run_mogen(args):
    cfg = Config.fromfile(args.config)
    if args.test_only:
        cfg['model']['cfg']['phase'] = 'test'
        test_mogen(cfg)
    else:
        train_mogen(cfg)
