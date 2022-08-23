# Tutorial 2: How to use Data Pipelines

In this tutorial, we concept of data pipelines, and how to customize and extend your own data pipelines for the project.

<!-- TOC -->

- [Tutorial 2: How to use Data Pipelines](#tutorial-2-how-to-use-data-pipelines)
  - [Concept of Data Pipelines](#concept-of-data-pipelines)
  - [Design of Data Pipelines](#design-of-data-pipelines)

<!-- TOC -->

## Concept of Data Pipelines
Data Pipeline is a modular form for data process. We make common data processing operations into python class, which named ```pipeline```.
For example, in image tasks, pre-processing often involves cropping, deformation, color, adding noise, etc. It is named ````pipeline``` in the mmcv series of codes.




The preprocessing flow is defined in the config file:
```python
train_pipeline = [
    dict(type='ToTensor', enable=True, keys=['music', 'dance'],),
]
```
For dance generation movements, since there is no pre-processing of music or action sequences in the current general algorithms, there is only one process - ToTensor \- which converts music/dance sequences into torch.tensor.


Pipelines are usually defined under the datasets folder. A template for a custom pipeline is as follows.

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
