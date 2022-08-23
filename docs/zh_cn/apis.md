# API
## run_mogen
输入: args, 运行参数
目的: 根据运行参数，训练或测试动作生成模型

## train_mogen
输入: cfg, mmcv.Config
目的: 根据运行参数，训练动作生成模型

## test_mogen
输入: cfg, mmcv.Config
目的: 根据运行参数，测试动作生成模型

## parse_args
输入: args, 运行参数
目的: 把运行参数转换成mmcv.Config
