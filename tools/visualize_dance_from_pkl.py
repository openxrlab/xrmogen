# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


from email.mime import audio
import os
import json
import argparse
import numpy as np

import numpy

from tqdm import tqdm

import os
import mmcv

import numpy as np
from xrmogen.data_structure.keypoints import Keypoints
from xrmogen.core.visualization import visualize_keypoints3d

def visualizeAndWritefromPKL(pkl_root, audio_path=None):

    video_root= os.path.join(pkl_root, 'video')
    if not os.path.exists(video_root):
        os.mkdir(video_root)

    music_names = sorted(os.listdir(audio_path))

    for pkl_name in tqdm(os.listdir(pkl_root), desc='Generating Videos'):

        if not pkl_name.endswith('.pkl'):
            continue
        print(pkl_name, flush=True)
        result = mmcv.load(os.path.join(pkl_root, pkl_name))

        np_dance = result[0]
        print(np_dance.shape)
        
        kps3d_arr = np_dance.reshape([np_dance.shape[0], 24, 3])
       
        kps3d_arr_w_conf = np.concatenate(
                (kps3d_arr, np.ones_like(kps3d_arr[..., 0:1])),
                axis=-1
            )
        kps3d_arr_w_conf = np.expand_dims(kps3d_arr_w_conf, axis=1)
        # mask array in shape (n_frame, n_person, n_kps)
        kps3d_mask = np.ones_like(kps3d_arr_w_conf[..., 0])
        convention = 'smpl'
        keypoints3d = Keypoints(
            kps=kps3d_arr_w_conf,
            mask=kps3d_mask,
            convention=convention
        )
        visualize_keypoints3d(
            keypoints=keypoints3d,
            output_path=os.path.join(video_root, pkl_name.split('.pkl')[0] + '.mp4')
        )
        dance_name = pkl_name.split('.pkl')[0]
    
        # video + audio
        name = dance_name.split(".")[0]
        if 'cAll' in name:
            music_name = name[-9:-5] + '.wav'

        if music_name in music_names:
            audio_dir_ = os.path.join(audio_path, music_name)
            name_w_audio = name + "_audio"
            cmd_audio = f"ffmpeg -i {video_root}/{name}.mp4 -i {audio_dir_} -map 0:v -map 1:a -c:v copy -shortest -y {video_root}/{name_w_audio}.mp4 -loglevel quiet"
            os.system(cmd_audio)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visulize from recorded pkl')
    parser.add_argument('--pkl_root', type=str)
    parser.add_argument('--audio_path', type=str, default='')
    args = parser.parse_args()

    visualizeAndWritefromPKL(args.pkl_root, args.audio_path)
