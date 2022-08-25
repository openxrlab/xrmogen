# 安装

本文档提供了安装 XRNerf 的相关步骤。

<!-- TOC -->

- [安装](#安装)
  <!-- - [安装依赖包](#安装依赖包) -->
  - [准备环境](#准备环境)
      - [a. 创建并激活 conda 虚拟环境.](#a-创建并激活-conda-虚拟环境)
      - [b. 安装 PyTorch 和 torchvision](#b-安装-pytorch-和-torchvision)
      - [c. 安装其他python包](#c-安装其他python包)

  - [利用 Docker 镜像安装 XRMoGen](#利用-docker-镜像安装-xrmogen)
  <!-- - [安装验证](#安装验证) -->

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


## 准备环境
<!-- 
#### a. 安装系统依赖库.

```shell
sudo apt install libgl-dev freeglut3-dev build-essential git ninja-build ffmpeg libsm6 libxext6 libgl1
``` -->

#### a. 创建并激活 conda 虚拟环境.

```shell
conda create -n xrnerf python=3.7 -y
conda activate xrnerf
```

#### b. 安装 PyTorch 和 torchvision

1. 查看pytorch-cuda版本匹配表，选择合适的版本 [here](https://pytorch.org/get-started/previous-versions/) 
2. 用对应`conda install` 命令安装对应版本的PyTorch以及Torchvision。

#### c. 安装其他python包
* ```pip install -r requirements.txt```
<!-- * 根据[官方说明](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)，安装 ```mmcv-full```
* 安装 ```spconv```, 比如 ```pip install spconv-cu111```. 值得注意的是只有部分cuda版本是支持的, 具体请查看 [官方说明](https://github.com/traveller59/spconv)
* 通过 ```pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"``` 安装 ```pytorch3d```
* 通过 ```pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch``` 安装 ```tcnn``` 
* 查看[官方说明](https://github.com/creiser/kilonerf#option-b-build-cuda-extension-yourself) 安装 ```kilo-cuda``` -->
  
<!-- #### e. 安装cuda扩展
* 为了支持instant-ngp算法，需要编译安装cuda扩展 ```raymarch```, 查看[具体教程](../../extensions/ngp_raymarch/README.md) -->


## 利用 Docker 镜像安装 XRMoGen

XRMogen 提供一个 [Dockerfile](../../Dockerfile) 可以直接创建 docker 镜像

```shell
docker build -f ./Dockerfile --rm -t xrmogen .
```

**注意** 用户需要确保已经安装了 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)。

运行以下命令，进入镜像:
```shell
docker run --gpus all -it xrmogen /workspace
```
在本机上(非docker镜像机内)开启一个终端，将项目文件(包括数据集)复制进docker镜像机
```shell
docker cp ProjectPath/xrmogen [DOCKER_ID]:/workspace
```
其中[DOCKER_ID] 是镜像的id, 通过下面命令确定
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

