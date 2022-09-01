from email import policy
import os
from datetime import datetime

num_gpus = 1

## optimizer
method = 'bailando'
phase = 'gpt'

# optimizer
optimizer = dict(type='Adam', lr=3e-4, betas=[0.5, 0.999])
optimizer_config = dict(grad_clip=None)

lr_rate = 3e-4
max_epochs = 500
evalute_config = dict()
lr_config = dict(policy='step', step=[250, 400], gamma=0.1)
checkpoint_config = dict(interval=20, by_epoch=True)
log_level = 'INFO'
log_config = dict(interval=10,  by_epoch=False, hooks=[dict(type='TextLoggerHook')])
workflow = [('train', 20), ('val', 1)]

# hooks
# 'params' are numeric type value, 'variables' are variables in local environment
train_hooks = [
    dict(type='SaveDancePKLHook',
         params=dict()),
]

test_hooks = [
    dict(type='SaveTestDancePKLHook',
         params=dict(save_folder='test')),
]

# runner
train_runner = dict(type='DanceTrainRunner')
test_runner = dict(type='DanceTestRunner')

# runtime settings
num_gpus = 1
distributed = 0  # multi-gpu
work_dir = './bailando_test/'.format(phase)  # noqa
timestamp = datetime.now().strftime("%d-%b-%H-%M")


load_from = os.path.join('./example/bailando.pth')

## dataset

traindata_cfg = dict( 
    data_dir='data/aistpp_train_wav',
    rotmat=False,
    seq_len=240,
    mode='train',
    move=8,
    external_wav='data/aistpp_music_feat_7.5fps',
    external_wav_rate=8
)

testdata_cfg = dict( 
    data_dir='data/aistpp_test_full_wav',
    rotmat=False,
    mode='test',
    move=8,
    external_wav='data/aistpp_music_feat_7.5fps',
    external_wav_rate=8
)

train_pipeline = [
    dict(type='ToTensor', enable=True, keys=['music', 'dance'],),
]
test_pipeline = [
    dict(type='ToTensor', enable=True, keys=['music', 'dance'],),
]

data = dict(
    train_loader=dict(batch_size=32, num_workers=0),
    train=dict(
        type='AISTppDataset',
        data_config=traindata_cfg,
        pipeline=train_pipeline,
    ),
    val_loader=dict(batch_size=1, num_workers=0),
    val=dict(
        type='AISTppDataset',
        data_config=testdata_cfg,
        pipeline=test_pipeline,
    ),
    test_loader=dict(batch_size=1, num_workers=0),
    test=dict(
        type='AISTppDataset',
        data_config=testdata_cfg,
        pipeline=test_pipeline,
    ),
)


##### model

model = dict(
    type='Bailando',
    model_config=dict(
        bailando_phase='gpt',
        vqvae=dict( 
            up_half=dict(
                levels=1,
                downs_t=[3,],
                strides_t =[2,],
                emb_width=512,
                l_bins=512,
                l_mu=0.99,
                commit=0.02,
                hvqvae_multipliers=[1,],
                width=512,
                depth=3,
                m_conv=1.0,
                dilation_growth_rate=3,
                sample_length=240,
                use_bottleneck=True,
                joint_channel=3,
                vqvae_reverse_decoder_dilation=True
            ),
            down_half=dict(
                levels=1,
                downs_t=[3,],
                strides_t =[2,],
                emb_width =512,
                l_bins =512,
                l_mu =0.99,
                commit =0.02,
                hvqvae_multipliers =[1,],
                width=512,
                depth=3,
                m_conv =1.0,
                dilation_growth_rate =3,
                sample_length=240,
                use_bottleneck=True,
                joint_channel=3,
                vqvae_reverse_decoder_dilation=True
            ),
            use_bottleneck=True,
            joint_channel=3,
        ),

        gpt=dict(
            block_size=29,
            base=dict(
                embd_pdrop=0.1,
                resid_pdrop=0.1,
                attn_pdrop=0.1,
                vocab_size_up=512,
                vocab_size_down=512,
                block_size=29,
                n_layer=6,
                n_head=12,
                n_embd=768 ,
                n_music=438,
                n_music_emb=768
            ),
            head=dict(
                embd_pdrop=0.1,
                resid_pdrop=0.1,
                attn_pdrop=0.1,
                vocab_size=512,
                block_size=29,
                n_layer=6,
                n_head=12,
                n_embd=768,
                vocab_size_up=512,
                vocab_size_down=512 
            ),
            n_music=438,
            n_music_emb=768
        )
    )
)
  