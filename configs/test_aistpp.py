import os
from datetime import datetime

num_gpus = 1

traindata_cfg = dict( 
    data_dir='/mnt/lustre/syli/dance/Bailando/data/aistpp_test_full_wav',
    rotmat=False,
    seq_len=240,
    mode='train',
    move=8,
    external_wav='/mnt/lustre/syli/dance/Bailando/data/aistpp_music_feat_7.5fps',
    external_wav_rate=8
)

testdata_cfg = dict( 
    data_dir='/mnt/lustre/syli/dance/Bailando/data/aistpp_test_full_wav',
    rotmat=False,
    mode='test',
    move=8,
    external_wav='/mnt/lustre/syli/dance/Bailando/data/aistpp_music_feat_7.5fps',
    external_wav_rate=8
)

train_pipeline = [
    dict(type='ToTensor', enable=True, keys=['music', 'dance'],),
]
test_pipeline = [
    dict(type='ToTensor', enable=True, keys=['music', 'dance'],),
]

data = dict(
    train_loader=dict(batch_size=32, num_workers=8),
    train=dict(
        type='AISTppDataset',
        data_config=traindata_cfg,
        pipeline=train_pipeline,
    ),
    val_loader=dict(batch_size=1, num_workers=8),
    val=dict(
        type='AISTppDataset',
        data_config=testdata_cfg,
        pipeline=test_pipeline,
    ),
    test_loader=dict(batch_size=1, num_workers=8),
    test=dict(
        type='AISTppDataset',
        data_config=testdata_cfg,
        pipeline=test_pipeline,
    ),
)