from mogen.models import *
from mogen.models.builder import build_dance_models
from mmcv import Config

config_path = 'configs/bailando.py'

cfg = Config.fromfile(config_path)

# print(cfg)

network = build_dance_models(cfg.model)

print(network)