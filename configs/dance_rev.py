from email import policy
import os
from datetime import datetime



num_gpus = 1

## optimizer
method = 'animatable_nerf'
phase = 'motion vqvae'

# optimizer
optimizer = dict(type='Adam', lr=1e-4, betas=[0.9, 0.999])
optimizer_config = dict(grad_clip=None)

lr_rate = 1e-4
max_epochs = 10
evalute_config = dict()
lr_config = dict(policy='step', step=[2, 3], gamma=0.1, by_epoch=True)
checkpoint_config = dict(interval=1, by_epoch=True)
log_level = 'INFO'
log_config = dict(interval=10,  by_epoch=False, hooks=[dict(type='TextLoggerHook')])
workflow = [('train', 2), ('val', 1)]
# workflow = [('val', 1)]
# hooks
# 'params' are numeric type value, 'variables' are variables in local environment
train_hooks = [
    dict(type='PassEpochNumberToModelHook',
         params=dict()),
    dict(type='SaveDancePKLHook',
         params=dict()),
]

test_hooks = [
    dict(type='SaveDancePKLHook',
         params=dict(save_folder='test')),
]

# runner
train_runner = dict(type='DanceTrainRunner')
test_runner = dict(type='DanceTestRunner')

# runtime settings
num_gpus = 1
distributed = 0  # 是否多卡，mmcv对dp多卡支持不好，故而要么单卡要么ddp多卡
work_dir = './try_dr/'.format(phase)  # noqa
timestamp = datetime.now().strftime("%d-%b-%H-%M")


## dataset

traindata_cfg = dict( 
    data_dir='/mnt/lustre/syli/dance/Bailando/data/aistpp_train_wav',
    rotmat=False,
    seq_len=240,
    mode='train',
    move=1
)

testdata_cfg = dict( 
    data_dir='/mnt/lustre/syli/dance/Bailando/data/aistpp_test_full_wav',
    rotmat=False,
    mode='test',
    move=1
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
load_from = os.path.join(work_dir, 'latest.pth')

##### model
model = dict(
    type='DanceRevolution',
    model_config=dict(
    #ChoreoGrapher Configs
        max_seq_len=4500,
        d_frame_vec=438,
        frame_emb_size=200,
        n_layers=2,
        n_head=8,
        d_k=64,
        d_v=64,
        d_model=200,
        d_inner=1024,
        dropout=0.1,
        d_pose_vec=72,
        pose_emb_size=72,
        condition_step=10,
        sliding_windown_size=100,
        lambda_v=0.01,
        cuda=True,
        rotmat=False
    )
)

