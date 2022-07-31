from mogen.models import *
from mogen.datasets.builder import build_dataset
from mmcv import Config
from mogen.core.apis.helper import build_dataloader

config_path = 'configs/test_aistpp.py'

cfg = Config.fromfile(config_path)
# print(cfg)
dataloader, _ = build_dataloader(cfg, mode='test')

cc= 0

print(_[2])

for ii, data in enumerate(dataloader):
    print(ii, data)
    print(ii, data['music'].size())
    break
# print(dataset[2]['music'].size(), dataset[2]['dance'].size(),)