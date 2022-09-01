# 教程 1: 如何编写配置（config）文件

XRMoGen 用 python 编写配置（config）文件。配置文件中包含所有实验需要的配置，包括训练和测试流程、使用的模型参数、使用的数据集参数，以及其他超参。
XRMoGen 提供的所有配置文件都放置在 `$PROJECT/configs` 文件夹下。

<!-- TOC -->

- [教程 1: 如何编写配置文件](#教程-1-如何编写配置文件)
  - [配置文件组成部分](#配置文件组成部分)

<!-- TOC -->

## 配置文件组成部分
配置文件的内容大体包含3个部分:
* 训练
* 模型
* 数据

我们以对Bailando模型的训练config来举例说明：

* 训练/测试配置
    训练配置部分包含了控制训练过程的各类参数，包括optimizer
    ```python

    ## optimizer
    method = 'bailando'
    phase = 'motion vqvae'

    # optimizer
    optimizer = dict(type='Adam', lr=3e-5, betas=[0.5, 0.999])
    optimizer_config = dict(grad_clip=None)

    lr_rate = 3e-5
    max_epochs = 500
    evalute_config = dict()
    lr_config = dict(policy='step', step=[100, 200], gamma=0.1, by_epoch=True)
    checkpoint_config = dict(interval=20, by_epoch=True)
    log_level = 'INFO'
    log_config = dict(interval=10,  by_epoch=False, hooks=[dict(type='TextLoggerHook')])
    ```

    workflow表示train和validation的频率
    ```python
    workflow = [('train', 20), ('val', 1)]
    ```

    mmcv框架下，包括在标准dataloader之外向model传输信息、或者将网络结果存储下来这种IO需要通过hook来实现，
    config里要定义所需的hooks
    ```python
    train_hooks = [
        dict(type='SaveDancePKLHook',
            params=dict()),
    ]

    test_hooks = [
        dict(type='SaveDancePKLHook',
            params=dict(save_folder='test')),
    ]
    ```

    mmcv框架下，训练和测试都是要通过一个runner实例来实现，在config中还需要定义runner以及它的参数。
    ```python
    # runner
    train_runner = dict(type='DanceTrainRunner')
    test_runner = dict(type='DanceTestRunner')
    ```

    此外还有其他参数，包括work_dir，即实验的文件夹
    ```python
    # runtime settings
    num_gpus = 1
    distributed = 0  # 是否多卡，mmcv对dp多卡支持不好，故而要么单卡要么ddp多卡
    work_dir = './trytry2/'.format(phase)  # noqa
    timestamp = datetime.now().strftime("%d-%b-%H-%M")
    ```





* 模型：
    模型部分定义了构建相应模型所需要的所有参数
    ```python
        model = dict(
        type='Bailando',
        model_config=dict(
            bailando_phase='motion vqvae',
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
    ```

* 数据
    数据部分的配置信息，定义了数据集类型，数据的处理流程，batchsize等等信息。
    ```python  
    traindata_cfg = dict(
        data_dir='/mnt/lustre/syli/dance/Bailando/data/aistpp_train_wav',
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
    ```
