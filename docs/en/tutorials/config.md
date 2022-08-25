# Tutorial 1: How to write a config file


In XRMoGen, configuration (config) files are implemented in python. 
A config file contains the configuration required for all experiments, including training and testing pipelines, model, dataset, and other hyperparameters.
All configuration files provided by XRMoGen are under the `$PROJECT/configs` folder.

<!-- TOC -->

- [Tutorial 1: How to write a config file](#tutorial-1-how-to-write-a-config-file)
  - [配置文件组成部分](#配置文件组成部分)

<!-- TOC -->

## Components
We can logically divide the configuration file into components:
* training
* model
* data

Let's take the training config of the Bailando model as an example:

* train/test

    The training part contains various parameters that control the training process, e.g., the optimizer
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

    `workflow` defines the frequency of training and validation
    ```python
    workflow = [('train', 20), ('val', 1)]
    ```

    Under the mmcv framework, IOs of training and test, like transmitting information to the model outside the standard dataloader, or storing network results,  need to be implemented through hooks.
    Required hooks are decalared in config

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

    Under the mmcv framework, training and testing are implemented through a `runner` instance. The `runner` instance and its parameters need to be defined in the config.
    ```python
    # runner
    train_runner = dict(type='DanceTrainRunner')
    test_runner = dict(type='DanceTestRunner')
    ```

    Besides, define `work_dir`, the experiment root, in config files
    ```python
    # runtime settings
    num_gpus = 1
    distributed = 0  # Whether multi-gpu; mmcv does not support dp multi-card well, so either single card or ddp multi-card
    work_dir = './experiments_first_try/'.format(phase)  # noqa
    timestamp = datetime.now().strftime("%d-%b-%H-%M")
    ```





* Model:
    The model section defines a model with all required parameters
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

* Data:    
    The data part defines the data set type, data processing flow, batch size and other information.
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
