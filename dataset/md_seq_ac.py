""" music-dance paired data for actor critic learning """

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset


class MoDaSeqAC(Dataset):
    def __init__(self, musics, dances, beats, interval=None):
        # if dances is not None:
        ups, downs = dances
        assert (len(musics) == (len(ups))), \
            'the number of dances should be equal to the number of musics'

        music_data = []
        dance_data_up = []
        dance_data_down = []
        beat_data = []
        mask_data = []

        ups, downs = dances

        for (np_music, np_dance_up, np_dance_down, beat) in zip(musics, ups, downs, beats):
            if interval is not None:
                seq_len, dim = np_music.shape
                for i in range(0, seq_len-interval+1):
                    if i == 0:
                        mask = np.ones([interval - 2], dtype=np.float32)
                    else:
                        mask = np.zeros([interval - 2], dtype=np.float32)
                        mask[-1] = 1.0

                    music_sub_seq = np_music[i: i + interval]
                    dance_sub_seq_up = np_dance_up[i: i + interval]
                    dance_sub_seq_down = np_dance_down[i: i + interval]
                    beat_this = beat[i*8:i*8+interval*8]
                    if len(beat_this) is not 8*interval:
                        for iii in range(8*interval - len(beat_this)):
                            beat_this = np.append(beat_this, beat_this[-1:])
                    if len(music_sub_seq) == interval  and len(dance_sub_seq_up) == interval:
                        music_data.append(music_sub_seq)
                        dance_data_up.append(dance_sub_seq_up)
                        dance_data_down.append(dance_sub_seq_down)
                        beat_data.append(beat_this)
                        mask_data.append(mask)


        self.musics = music_data
        self.dances_up = dance_data_up
        self.dances_down = dance_data_down
        self.beat_data = beat_data
        self.mask_data = mask_data


    def __len__(self):
        return len(self.musics)

    def __getitem__(self, index):
        return self.musics[index], self.dances_up[index], self.dances_down[index], self.beat_data[index], self.mask_data[index]

