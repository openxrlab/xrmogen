import numpy as np
from xrmogen.data_structure.keypoints import Keypoints
from xrmogen.core.visualization import visualize_keypoints3d
from xrprimer.utils.log_utils import get_logger


# input_path = '/home/share_data/xrmogen/wulala.npy'
# kps3d_arr = np.load(input_path, allow_pickle=True)
# kps3d_arr = kps3d_arr.reshape(
#         kps3d_arr.shape[0], -1, 3)[:200, ...][:, :24, :]

input_path = '/home/share_data/xrmogen/psyduck.json.pkl.npy'
kps3d_arr = np.load(input_path, allow_pickle=True).item()['pred_position']
kps3d_arr = kps3d_arr.reshape(
        kps3d_arr.shape[0], -1, 3)[:200, ...]

kps3d_arr_w_conf = np.concatenate(
    (kps3d_arr, np.ones_like(kps3d_arr[..., 0:1])),
    axis=-1
)
kps3d_arr_w_conf = np.expand_dims(kps3d_arr_w_conf, axis=1)
logger = get_logger()
logger.info(kps3d_arr_w_conf.shape)
# mask array in shape (n_frame, n_person, n_kps)
kps3d_mask = np.ones_like(kps3d_arr_w_conf[..., 0])
convention = 'smpl_45' if kps3d_arr_w_conf.shape[2] == 45 \
    else 'smpl'
keypoints3d = Keypoints(
    kps=kps3d_arr_w_conf,
    mask=kps3d_mask,
    convention=convention
)
visualize_keypoints3d(
    keypoints=keypoints3d,
    output_path='/home/share_data/xrmogen/visualize_keypoints3d.mp4'
)