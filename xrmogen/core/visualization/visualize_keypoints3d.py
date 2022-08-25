# yapf: disable
import numpy as np
from typing import Union

from xrmogen.data_structure.keypoints import Keypoints
from scipy.spatial.transform import Rotation as scipy_Rotation

try:
    from mmhuman3d.core.visualization.visualize_keypoints3d import (
        visualize_kp3d,
    )
    has_mmhuman3d = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_mmhuman3d = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception
# yapf: enable


def visualize_keypoints3d(
        keypoints: Keypoints,
        output_path: str,
        return_array: bool = False) -> Union[None, np.ndarray]:
    """Visualize 3d keypoints, powered by mmhuman3d.

    Args:
        keypoints (Keypoints):
            An keypoints3d instance of Keypoints.
        output_path (str):
            Path to the output file. Either a video path
            or a path to an image folder.
        return_array (bool, optional):
            Whether to return the visualized image array.
            Defaults to False.

    Returns:
        Union[None, np.ndarray]:
            If return_array it returns an array of images,
            else return None.
    """
    logger = keypoints.logger
    if not has_mmhuman3d:
        logger.error(import_exception)
        raise ImportError
    # prepare keypoints data
    keypoints_np = keypoints.to_numpy()
    kps3d = keypoints_np.get_keypoints()[..., :3].copy()
    rotation = scipy_Rotation.from_euler('zxy', [180, 0, 180], degrees=True)
    kps3d = rotation.apply(
        kps3d.reshape(-1, 3)).reshape(
        keypoints_np.get_frame_number(), keypoints_np.get_person_number(),
        keypoints_np.get_keypoints_number(), 3)
    if keypoints_np.get_person_number() == 1:
        kps3d = np.squeeze(kps3d, axis=1)
    kps_convention = keypoints_np.get_convention()
    kps_mask = keypoints_np.get_mask()
    mm_kps_mask = np.sign(np.sum(np.abs(kps_mask), axis=(0, 1)))
    vis_arr = visualize_kp3d(
        kp3d=kps3d,
        output_path=output_path,
        data_source=kps_convention,
        mask=mm_kps_mask,
        return_array=return_array,
        fps=60)
    return vis_arr
