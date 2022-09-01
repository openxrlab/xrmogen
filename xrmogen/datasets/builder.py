from mmcv.utils import Registry, build_from_cfg

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def build_dataset(cfg, default_args=None):
    """Build a dataset from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.
    Returns:
        Dataset: The constructed dataset.
    """
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset
