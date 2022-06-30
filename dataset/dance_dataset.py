
from dataset.load_data import load_train_data_aist, load_test_data_aist
import numpy as np

from torch.utils.data import Dataset


class DanceDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dances = None
        self.mode = cfg.mode
        self._init_load()

    def _init_load(self):
        if self.mode == 'train':
            _, dances, fnames = load_train_data_aist(self.cfg)
        elif self.mode == 'test':
            _, dances, fnames = load_test_data_aist(self.cfg)
        self.dances = dances
        self.fnames = fnames

    def __len__(self):
        return len(self.dances)

    def __getitem__(self, index):
        return self.dances[index]
