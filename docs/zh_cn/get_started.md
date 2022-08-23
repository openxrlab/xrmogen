# 上手指南

我们提供一份指南，旨在帮用户快速了解以及使用XRMoGen。
关于安装部分的介绍，参见[installation.md](installation.md)。


<!-- TOC -->

- [Getting Started](#getting-started)
  - [Datasets](#datasets)
  - [Build a Model](#build-a-model)
    - [XRMoGen Composition](#xrmogen-composition)
    - [Write a new dance generation model](#write-a-new-dance-generation-model)
  - [Train a Model](#train-a-model)
    - [Iteration Controls](#epoch-controls)
    - [Train](#train)
    - [Test](#test)
    - [Visualize](#visualize)
  - [Tutorials](#tutorials)

<!-- TOC -->

## 数据集

我们推荐使用预提取特征的音乐和动作数据，参见[dataset_preparation.md](dataset_preparation.md)。
下载后解压至$PROJECT/data。为了方便在生成舞蹈后合成有音乐的视频，建议将原始音乐（.mov）下载到同一目录的的musics文件夹下：


```
xrmogen
├── mogen
├── docs
├── configs
├── data
│   ├── aistpp_train_wav
│   ├── aistpp_test_full_wav
│   ├── aistpp_music_feat_7.5fps
│   ├── aist_features_zero_start
│   ├── musics
├── ...
```


## Build a Model

### XRMoGen Composition

Currently, XRMoGen contains two dance generation algorithms

- Bailando: Siyao *et al.*, Bailando: 3D Dance Generation by Actor-Critic GPT with Choreographic Memory, CVPR 2022
- DanceRevolution: Huang *et al.*, Dance Revolution: Long-Term Dance Generation with Music via Curriculum Learning, ICLR 2021


The model structure can be customized through config files.

### Write a new dance generation model

To implement a new method, your model need to contain following functions/medhotds to fit the training/test pipeline:


- `train_step()`: forward method of the training mode.
- `val_step()`: forward method of the testing mode.
- regestered as a dance model


To be specific, if we want to implement a new model, there are several things to do.

1. create a new file in `mogen/models/dance_models/my_model.py`.

    ```python
    from ..builder import NETWORKS
    from ...builder import DANCE_MODELS

    @DANCE_MODELS.register_module()
    class MyDanceModel(nn.Module):

        def __init__(self, model_config):
            super().__init__()
        
        def forward(self, ...):
            ....

        def train_step(self, data, optimizer, **kwargs):
            ....

        def val_step(self, data, optimizer=None, **kwargs):
            ....
    ```

2. Import the model in `mogen/models/__init__.py`

    ```python
    from .my_model import MyDanceModel
    ```

3. write a config file that defines the model as


    ```python
    model = dict(
        type='MyDanceModel',
        ....
    ```



## Train a Model

### Epoch Controls

XRMoGen uses `mmcv.runner.EpochBasedRunner` to control training and test.

In the training mode, the `max_epochs` in config file decide how many epochs to train. 
In test mode, `max_epochs` is forced to change to 1, which represents only 1 epoch to test.

Validation frequency is set as `workflow` of config file:
```python
 workflow = [('train', 20), ('val', 1)]
```

### Train
For example, to train Bailando (Motion VQVAE phase),

```shell
python main.py --config configs/config/bailando_motion_vqvae.py 
```

Arguments are:
- `--config`: config file path.


### Test
To test relevant model, add `--test_only` tag after the config path:

```shell
python main.py --config configs/config/bailando_motion_vqvae.py --test_only
```

To Compute the quantitative scores:
```python
python tools/eval_quantitative_scores.py --pkl_root [GENERATED_DANCE_PKL_ROOT] --gt_root [GROUND_TRUTH_FEATURES=data/aist_features_zero_start] --music_feature_root [MUSIC_FEATURE_ROOT=aistpp_test_full_wav]

```

### Visualize

```python
python tools/visualize_dance_from_pkl.py --pkl_root [GENERATED_DANCE_PKL_ROOT] --audio_path data/musics/
```


## Tutorials
Currently, we provide some tutorials for users to learn about
* [configs](tutorials/config.md)
* [data pipeline](tutorials/data_pipeline.md)
* [model](tutorials/model.md)
