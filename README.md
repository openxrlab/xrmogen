
# TODO
- [x] 完成train/test api
- [x] 完成 runner (epoch runner)
- [x] 完成test和validation的hook，将3d pose positions 存到pkl
- [x] 完成config 
- [x] 完成dance revolution 的代码风格化转变
- [x] 完成tools的代码优化
- [ ] 完成各项测试（在当前框架下的重新训练、测试、resume等）
- [ ] dockerfile

## Environment
````PyTorch >= 1.6.0````

    pip install -r requirements.txt

## Data preparation

See [dataset_preparation.md](docs/en/dataset_preparation.md)


## Training
    sh srun_train.sh [CONFIG_PATH] [your node name] 1

## Test
    sh srun_test.sh [CONFIG_PATH] [your node name] 1
