# Installation


<!-- TOC -->

We provide some tips for XRMoGen installation in this file.


- [Installation](#installation)
  - [Requirements](#requirements)
  - [Prepare environment](#prepare-environment)
      - [a. Create a conda virtual environment and activate it.](#a-create-a-conda-virtual-environment-and-activate-it)
      - [b. Install PyTorch and torchvision](#b-install-pytorch-and-torchvision)
      - [c. Install Other Needed Python Packages](#c-install-other-needed-python-packages)
  - [Another option: Docker Image](#another-option-docker-image)


<!-- TOC -->
<!-- 
## 安装依赖包

- Linux
- Python 3.7+
- PyTorch 1.6+
- CUDA 10.0+
- GCC 7.5+
- build-essential: Install by `apt-get install -y build-essential git ninja-build ffmpeg libsm6 libxext6 libgl1`
- [mmcv-full](https://github.com/open-mmlab/mmcv)
- Numpy
- ffmpeg
- [opencv-python 3+](https://github.com/dmlc/decord): 可通过 `pip install opencv-python>=3` 安装
- [imageio](https://github.com/dmlc/decord): 可通过 `pip install imageio` 安装
- [scikit-image](https://github.com/dmlc/decord): 可通过 `pip install scikit-image` 安装
- [spconv](https://github.com/dmlc/decord): 从支持的版本中选择跟你本地cuda版本一致的安装, 比如 `pip install spconv-cu113`
- [pytorch3d](https://github.com/dmlc/decord): 可通过 `pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"` 安装
 -->


## Prepare Environment
<!-- 
#### a. 安装系统依赖库.

```shell
sudo apt install libgl-dev freeglut3-dev build-essential git ninja-build ffmpeg libsm6 libxext6 libgl1
``` -->

#### a. Create a conda virtual environment and activate it.

```shell
conda create -n xrnerf python=3.7 -y
conda activate xrnerf
```

#### b. Install PyTorch and torchvision

1. Check the version of pytorch-cuda，and select a suitable on at [here](https://pytorch.org/get-started/previous-versions/) 
2. Use the  `conda install` command for corresponding version to install PyTorch and Torchvision。

#### c.  Install Other Needed Python Packages
* ```pip install -r requirements.txt```
<!-- * 根据[官方说明](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)，安装 ```mmcv-full```
* 安装 ```spconv```, 比如 ```pip install spconv-cu111```. 值得注意的是只有部分cuda版本是支持的, 具体请查看 [官方说明](https://github.com/traveller59/spconv)
* 通过 ```pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"``` 安装 ```pytorch3d```
* 通过 ```pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch``` 安装 ```tcnn``` 
* 查看[官方说明](https://github.com/creiser/kilonerf#option-b-build-cuda-extension-yourself) 安装 ```kilo-cuda``` -->
  
<!-- #### e. 安装cuda扩展
* 为了支持instant-ngp算法，需要编译安装cuda扩展 ```raymarch```, 查看[具体教程](../../extensions/ngp_raymarch/README.md) -->


## Another option: Docker Image

XRMogen provides a [Dockerfile](../../Dockerfile) to build the docker image directly

```shell
docker build -f ./Dockerfile --rm -t xrmogen .
```

**Attention** Users need to make sure that  [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) is properly installed.

Run the following command to start the docker image
```shell
docker run --gpus all -it xrmogen /workspace
```
Open a teiminal in your host computer, copy project into docker container
```shell
docker cp ProjectPath/xrmogen [DOCKER_ID]:/workspace
```
where [DOCKER_ID] is the docker id that can be obtained by
```
docker ps -a
```
  
<!-- ## 安装验证

为了验证 XRNerf 和所需的依赖包是否已经安装成功，可以运行单元测试模块

```shell
coverage run --source xrnerf/models -m pytest -s test/models && coverage report -m
```

注意，运行单元测试模块前需要额外安装 ```coverage``` 和 ```pytest``` 
```
pip install coverage pytest -i https://pypi.tuna.tsinghua.edu.cn/simple
``` -->

