# 上手指南

我们提供一份指南，旨在帮用户快速了解以及使用XRMoGen。
关于安装部分的介绍，参见[installation.md](installation.md)。


<!-- TOC -->

- [上手指南](#上手指南)
  - [数据集](#数据集)
  - [构建模型](#构建模型)
    - [XRMoGen 构成](#xrmogen-构成)
    - [如何构建新的模型](#如何构建新的模型)
  - [模型训练](#模型训练)
    - [Epoch 控制](#epoch-控制)
    - [训练](#训练)
    - [测试](#测试)
    - [可视化](#可视化)
  - [教程](#教程)

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


## 构建模型

### XRMoGen 构成

目前，XRMoGen中包含以下两种舞蹈生成的方法：

- Bailando: Siyao *et al.*, Bailando: 3D Dance Generation by Actor-Critic GPT with Choreographic Memory, CVPR 2022
- DanceRevolution: Huang *et al.*, Dance Revolution: Long-Term Dance Generation with Music via Curriculum Learning, ICLR 2021


模型结构由配置文件（config）来控制。

### 如何构建新的模型

要实现新的方法，模型需要包含以下的函数/方法以适应当前的训练/测试流程：

- `train_step()`: 模型训练的正向过程；
- `val_step()`: 模型测试的正向过程；
- 将模型注册为一个DANCE_MODELS


具体来说，如果我们想实现一个新模型，有几件事要做：

1. 在`mogen/models/dance_models/`下创建一个新模型`my_model.py`：

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

2. 在`mogen/models/__init__.py`中导入新模型：

    ```python
    from .my_model import MyDanceModel
    ```

3. 将新模型的构建参数写进需要的config文件：


    ```python
    model = dict(
        type='MyDanceModel',
        ....
    ```



## 模型训练

### Epoch 控制

XRMoGen 使用 `mmcv.runner.EpochBasedRunner` （以epoch为单位）去训练和测试模型.

在训练模式下， config文件中的 `max_epochs` 参数决定了模型被训练多少epoch。
在测试模式下, `max_epochs` 被强制设置为1，即将测试数据测试一遍。

训练、测试交替的频率在config文件中的 `workflow` 设定:
```python
 workflow = [('train', 20), ('val', 1)]
```

### 训练
比如，为了训练Bailando模型 (Motion VQVAE phase)，运行以下命令

```shell
python main.py --config configs/config/bailando_motion_vqvae.py
```

参数:
- `--config`: config 文件路径


### 测试
测试相应的模型，只需要在config路径后添加 `--test_only`:

```shell
python main.py --config configs/config/bailando_motion_vqvae.py --test_only
```

对生成的舞蹈动作计算量化指标：
```python
python tools/eval_quantitative_scores.py --pkl_root [生成的动作PKL文件夹] --gt_root data/aist_features_zero_start --music_feature_root data/aistpp_test_full_wav

```

### 可视化

```python
python tools/visualize_dance_from_pkl.py --pkl_root [生成的动作PKL文件夹] --audio_path data/musics/
```


## 教程
目前我们提供以下教程
* [configs](tutorials/config.md)
* [data pipeline](tutorials/data_pipeline.md)
* [model](tutorials/model.md)
