import torch
from dataset.music_dance_dataset import MusicDanceDataset
import yaml
from easydict import EasyDict



with open('configs/test_dataset.yaml') as f:
    config = yaml.load(f)

config = EasyDict(config)

train_dataset = MusicDanceDataset(config.train_data)
test_dataset = MusicDanceDataset(config.test_data)

print(len(train_dataset), train_dataset[3][0].shape, train_dataset[4][1].shape)
print(len(test_dataset), test_dataset[1][0].shape, test_dataset[1][1].shape, test_dataset[1][2])
