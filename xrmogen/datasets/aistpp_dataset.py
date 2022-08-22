
from .load_data.load_music_dance_data import load_train_data_aist, load_test_data_aist
""" Define the paired music-dance dataset. """
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
from .builder import DATASETS
from .pipeline.compose import Compose

@DATASETS.register_module()
class AISTppDataset(Dataset):
    def __init__(self, data_config, pipeline):
        self.cfg = data_config
        self.dances = None
        self.musics = None
        self.mode = self.cfg.mode
        self._init_load()
        self.pipeline = Compose(pipeline)


    def _init_load(self):
        if self.mode == 'train':
            musics, dances, fnames = load_train_data_aist(self.cfg)
        elif self.mode == 'test':
            musics, dances, fnames = load_test_data_aist(self.cfg)
        
        print(len(musics), musics[0].shape, len(dances), dances[0].shape, len(fnames))
        self.musics = musics
        self.dances = dances
        self.fnames = fnames

    def __len__(self):
        return len(self.musics)

    def __getitem__(self, index):
        data = {
            'music': self.musics[index],
            'dance': self.dances[index],
            'file_names': self.fnames[index],
        }
        return self.pipeline(data)


