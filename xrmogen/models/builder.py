# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

DANCE_MODELS = MODELS


def build_dance_models(cfg):
    # print(cfg.keys())
    return DANCE_MODELS.build(cfg)
