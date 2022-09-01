# Data Preparation

In this document we describe the data involved in XRMoGen and how to use it.

<!-- TOC -->

- [Data Preparation](#data-preparation)
  - [Download Data](#download-data)
      - [Data Structure](#data-structure)


<!-- TOC -->

## Download Data

The dataset used in the dance generation task is AIST++. It contains [original music](https://aistdancedb.ongaaccel.jp/database_download/) and motion annotations(https://google.github.io/aistplusplus_dataset/download.html). In XRMoGen, we provide extracted musical features (438 dimensions) and 3D coordinates of human key points extracted based on the SMPL model (download [here](https://openxrlab-share.oss-cn-hongkong.aliyuncs.com/xrmogen/data.zip)).


## Data Structure
We recommend using the extracted data. Please download it and unzip it to $PROJECT/data. In order to facilitate the synthesis of a video with music after the dance is generated, it is needed to download the original music (.mov) to the musics folder in the same directory:


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
