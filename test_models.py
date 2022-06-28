import torch
from models import SepVQVAE
import yaml
from easydict import EasyDict



with open('configs/test_model.yaml') as f:
    config = yaml.load(f)

config = EasyDict(config)

model = SepVQVAE(config.structure)


