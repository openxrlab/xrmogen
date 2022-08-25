## Environment
````PyTorch >= 1.6.0````

    pip install -r requirements.txt

## Data preparation

See [dataset_preparation.md](docs/en/dataset_preparation.md)


## Training
    sh srun_train.sh [CONFIG_PATH] [your node name] 1

## Test
    sh srun_test.sh [CONFIG_PATH] [your node name] 1
