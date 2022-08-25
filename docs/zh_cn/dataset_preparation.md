# 数据准备

我们在此文档中介绍mogen所用到的数据以及使用方法。

<!-- TOC -->

- [数据准备](#数据准备)
  - [获取数据](#获取数据)
      - [数据结构](#数据结构)


<!-- TOC -->

## 获取数据

mogen舞蹈生成任务中采用的数据集是AIST++。包含[原始音乐](https://aistdancedb.ongaaccel.jp/database_download/)，和[动作标注](https://google.github.io/aistplusplus_dataset/download.html)。在xrmogen中，我们提供抽取的音乐特征(438维）以及基于smpl模型提取的人体关键点三维坐标（[此处]()下载）。


## 数据结构
我们推荐使用提取特征后的数据，下载后解压至$PROJECT/data。并且为了方便在生成舞蹈后合成有音乐的视频，建议将原始音乐（.mov）下载到同一目录的的musics文件夹下：


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


