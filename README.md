
# TODO
- [x] 完成train/test api
- [x] 完成 runner (epoch runner)
- [x] 完成test和validation的hook，将3d pose positions 存到pkl
- [x] 完成config 
- [x] 完成dance revolution 的代码风格化转变
- [x] 完成tools的代码优化
- [ ] 完成各项测试（在当前框架下的重新训练、测试、resume等）
- [ ] docker
## Benchmark

| 方法  | FID_k $\downarrow$ | FID_g $\downarrow$ | DIV_k $\uparrow$ | DIV_g $\uparrow$  | BA $\uparrow$ |
| :--- | :----: | :----: | :----: | :----: | :----: |
| Bailando | 28.16 | 9.62 | 7.83 | 6.34 | 0.2332 |
| FACT | 35.35 | 22.11 | 5.94 | 6.18 | 0.2209 |
| FACT reimpl @ 45epochs| 59.19 | 18.12 | 9.55 | 5.56 | -- |
| DanceRevolution | 73.42 | 25.92 | 3.52 | 4.87 | 0.19.50 |
| DanceRevolution reimpl @ 10epochs | 67.48 | 27.08 | 3.61 | 4.55 | -- |
| DanceNet | 69.18 | 25.49 | 2.86 | 2.85 | 0.1430 |


## 日志

见docs/update.txt

## Environment
````PyTorch == 1.6.0````

## Data preparation

下载[预处理数据](https://drive.google.com/file/d/1EGJeBE1fE59ByjxR_-ipwV6Dz-Cx-stT/view?usp=sharing) 到 ./data 文件夹中.

## 训练SOTA （判别式模型）

````sh srun_fact.sh configs/fact.yaml train [your node name] 1````
    

## 训练

The training of Bailando comprises of 4 steps in the following sequence. If you are using the slurm workload manager, you can directly run the corresponding shell. Otherwise, please remove the 'srun' parts. Our models are all trained with single NVIDIA V100 GPU. * A kind reminder: the quantization code does not fit multi-gpu training
<!-- If you are using the slurm workload manager, run the code as

If not, run -->

### Step 1: Train pose VQ-VAE (without global velocity)

    sh srun.sh configs/sep_vqvae.yaml train [your node name] 1

### Step 2: Train glabal velocity branch of pose VQ-VAE

    sh srun.sh configs/sep_vqvae_root.yaml train [your node name] 1

### Step 3: Train motion GPT

    sh srun_gpt_all.sh configs/cc_motion_gpt.yaml train [your node name] 1

### Step 4: Actor-Critic finetuning on target music 

    sh srun_actor_critic.sh configs/actor_critic.yaml train [your node name] 1

## 测试

To test with our pretrained models, please download the weights from [here](https://drive.google.com/file/d/1Fi0TIiBV6EQAQrBU0IOnlke2Nu4IcutC/view?usp=sharing) (Google Drive) or separately downloads the four weights from [[weight 1]](https://www.jianguoyun.com/p/DcicSkIQ6OS4CRiH8LYE)|[[weight 2]](https://www.jianguoyun.com/p/DTi-B1wQ6OS4CRjonbwEIAA)|[[weight 3]](https://www.jianguoyun.com/p/Dde220EQ6OS4CRiD8LYE)|[[weight4]](https://www.jianguoyun.com/p/DRHA80cQ6OS4CRiC8LYE) (坚果云) into ./experiments folder.

### 1. Generate dancing results

To test the VQ-VAE (with or without global shift as you indicated in config):

````sh srun.sh configs/sep_vqvae.yaml eval [node name] 1````

To test GPT:

````sh srun_gpt_all.sh configs/cc_motion_gpt.yaml eval [node name] 1````
   
To test final restuls:
    
````sh srun_actor_critic.sh configs/actor_critic.yaml eval [node name] 1````

### 2. Dance quality evaluations

After generating the dance in the above step, run the following codes.

### Step 1: Extract the (kinetic & manual) features of all AIST++ motions (ONLY do it by once):
    
````python extract_aist_features.py````


### Step 2: compute the evaluation metrics:

````python utils/metrics_new.py````

It will show exactly the same values reported in the paper. To fasten the computation, comment Line 184 of utils/metrics_new.py after computed the ground-truth feature once. To test another folder, change Line 182 to your destination, or kindly modify this code to a "non hard version" :)

## Choreographic for music in the wild

Bailando is trained on AIST++, which is not able to cover all musics in the wild. For example, musics in AIST++ do not contain lyrics, and could be relatively simple than dance musics in our life. So, to fill the gap, our solution is to finetune the pretrained Bailando on the music(s) for several epochs using the "actor-critic learning" process in our paper.  

To do so, make a folder named "./extra/" and put your songs (should be mp3 file) into it (not too many for one time), and extract the features as

````sh prepare_demo_data.sh````
    
Then, run the reinforcement learning code as

````sh srun_actor_critic.sh configs/actor_critic_demo.yaml train [node name] 1````




