# 教程 2: 如何设计数据处理流程

在本教程中，我们将介绍一些有关数据前处理流水线的概念以及设计的方法。

<!-- TOC -->

- [教程 2: 如何设计数据处理流程](#教程-2-如何设计数据处理流程)
  - [数据处理流程的基本概念](#数据处理流程的基本概念)
  - [Design of Data Pipelines](#design-of-data-pipelines)

<!-- TOC -->

## 数据处理流程的基本概念
数据流水线（pipeline）是对送入网络前的数据的前处理。通常来说，对于图像任务，前处理经常涉及到剪裁、形变、颜色、添加噪声等。在mmcv系列代码中被命名为```pipeline```。

前处理流程在config文件中被定义:
```python
train_pipeline = [
    dict(type='ToTensor', enable=True, keys=['music', 'dance'],),
]
```
对于舞蹈生成动作，由于目前普遍的算法都尚无对音乐或动作序列做前处理，因此只有ToTensor一项流程，即将音乐/舞蹈序列转化成torch.tensor。

pipeline通常在datasets文件夹下定义。一个自定义pipeline的模板如下

```python
@PIPELINES.register_module()
class PipelineA:
    """get viewdirs from rays_d
    """
    def __init__(self, enable=True, **kwargs):
        self.enable = enable

    def __call__(self, results):
        """get viewdirs
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        # process on results
        return results
```
