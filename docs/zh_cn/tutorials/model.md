# Tutorial 3: Model

In this tutorial, we will give a brief introduction to the dance generation models contained in XRMoGen and their interface.

<!-- TOC -->

- [Tutorial 3: Model](#tutorial-3-model)
  - [Dance Generation Models](#the-design-of-nerf-model)



<!-- TOC -->

## Dance Generation Models

Currently, XRMoGen contains two dance generation algorithms

- Bailando: Siyao *et al.*, Bailando: 3D Dance Generation by Actor-Critic GPT with Choreographic Memory, CVPR 2022
- DanceRevolution: Huang *et al.*, Dance Revolution: Long-Term Dance Generation with Music via Curriculum Learning, ICLR 2021

In both models, interfaces for training and validation/test of the runner are implemented:
```python

@DANCE_MODELS.register_module()
class MyDanceModel(nn.Module):

    ....

    def train_step(self, data, optimizer, **kwargs):
        ....

    def val_step(self, data, optimizer=None, **kwargs):
        ....
```
Input `data` is temporally paired music fueatures and 3D human pose sequence. For training, both music features and 3D poses are used for supervised learning.
For test, only music features with the starting pose (code) are used to generate dance.

Output of validation is a dictionary, where `output_pose` is generated dance with size of (nframes, njoints=24, ndim=3). `file_name` is a string of the file name of the corresponding output pose.

```python
  outputs = {
      'output_pose': results[0],
      'file_name': data['file_names'][0]
  }
```

Output pose will be stored in `.pkl` format after validation.