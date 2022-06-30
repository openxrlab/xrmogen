# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" This script handling the training process. """
import os
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from dataset.music_dance_dataset import MusicDanceDataset
# from models.gpt2 import condGPT2

from utils.log import Logger
from utils.functional import str2bool, load_data, load_data_aist, check_data_distribution,visualizeAndWrite,load_test_data_aist,load_test_data
from torch.optim import *
import warnings
from tqdm import tqdm
import itertools
import pdb
import numpy as np
import models
import datetime
warnings.filterwarnings('ignore')

import torch.nn.functional as F
# a, b, c, d = check_data_distribution('/mnt/lustre/lisiyao1/dance/dance2/DanceRevolution/data/aistpp_train')

import matplotlib.pyplot as plt



class MCTall():
    def __init__(self, args):
        self.config = args
        torch.backends.cudnn.benchmark = True
        self._build()

    def train(self):
        vqvae = self.model.eval()
        gpt = self.model2.train()

        config = self.config
        # data = self.config.data
        # criterion = nn.MSELoss()
        training_data = self.training_data
        test_loader = self.test_loader
        optimizer = self.optimizer
        log = Logger(self.config, self.expdir)
        updates = 0
        
        checkpoint = torch.load(config.vqvae_weight)
        vqvae.load_state_dict(checkpoint['model'], strict=False)

        if hasattr(config, 'init_weight') and config.init_weight is not None and config.init_weight is not '':
            print('Use pretrained model!')
            print(config.init_weight)  
            checkpoint = torch.load(config.init_weight)
            gpt.load_state_dict(checkpoint['model'], strict=False)
        # self.model.eval()

        random.seed(config.seed)
        torch.manual_seed(config.seed)
        #if args.cuda:
        torch.cuda.manual_seed(config.seed)
        self.device = torch.device('cuda' if config.cuda else 'cpu')


        # Training Loop
        for epoch_i in range(1, config.epoch + 1):
            log.set_progress(epoch_i, len(training_data))

            for batch_i, batch in enumerate(training_data):

                music_seq, pose_seq  = batch 

                music_seq = music_seq.to(self.device)
                pose_seq = pose_seq.to(self.device)

                pose_seq[:, :, :3] = 0
                optimizer.zero_grad()

                music_ds_rate = config.ds_rate if not hasattr(config, 'external_wav') else config.external_wav_rate
                music_ds_rate = config.music_ds_rate if hasattr(config, 'music_ds_rate') else music_ds_rate
                music_relative_rate = config.music_relative_rate if hasattr(config, 'music_relative_rate') else config.ds_rate
                
                music_seq = music_seq[:, :, :config.structure_generate.n_music//music_ds_rate].contiguous().float()

                b, t, c = music_seq.size()
                music_seq = music_seq.view(b, t//music_ds_rate, c*music_ds_rate)

                if hasattr(config, 'music_normalize') and config.music_normalize:
                    print('Normalize!')
                    music_seq = music_seq / ( t//music_ds_rate * 1.0 )

                with torch.no_grad():
                    quants_pred = vqvae.module.encode(pose_seq)
                    if isinstance(quants_pred, tuple):
                        quants_input = tuple(quants_pred[ii][0][:, :-1].clone().detach() for ii in range(len(quants_pred)))
                        quants_target = tuple(quants_pred[ii][0][:, 1:].clone().detach() for ii in range(len(quants_pred)))
                    else:
                        quants = quants_pred[0]
                        quants_input = quants[:, :-1].clone().detach()
                        quants_target = quants[:, 1:].clone().detach()

                output, loss = gpt(quants_input, music_seq[:, config.ds_rate//music_relative_rate:], quants_target)
                loss.backward()

                # update parameters
                optimizer.step()

                stats = {
                    'updates': updates,
                    'loss': loss.item()
                }

                log.update(stats)
                updates += 1

            checkpoint = {
                'model': gpt.state_dict(),
                'config': config,
                'epoch': epoch_i
            }

            # # Save checkpoint
            if epoch_i % config.save_per_epochs == 0 or epoch_i == 1:
                filename = os.path.join(self.ckptdir, f'epoch_{epoch_i}.pt')
                torch.save(checkpoint, filename)
            # Eval
            if epoch_i % config.test_freq == 0:
                with torch.no_grad():
                    print("Evaluation...")
                    gpt.eval()
                    results = []
                    random_id = 0  # np.random.randint(0, 1e4)
                    quants_out = {}
                    for i_eval, batch_eval in enumerate(tqdm(test_loader, desc='Generating Dance Poses')):
                        # Prepare data
                        music_seq, pose_seq = batch_eval
                        music_seq = music_seq.to(self.device)
                        pose_seq = pose_seq.to(self.device)
                        
                        quants = vqvae.module.encode(pose_seq)

                        if isinstance(quants, tuple):
                            x = tuple(quants[i][0][:, :1] for i in range(len(quants)))
                        else:
                            x = quants[0][:, :1]
                        
                        music_ds_rate = config.ds_rate if not hasattr(config, 'external_wav') else config.external_wav_rate
                        music_ds_rate = config.music_ds_rate if hasattr(config, 'music_ds_rate') else music_ds_rate
                        music_relative_rate = config.music_relative_rate if hasattr(config, 'music_relative_rate') else config.ds_rate
                        music_seq = music_seq[:, :, :config.structure_generate.n_music//music_ds_rate].contiguous().float()

                        b, t, c = music_seq.size()
                        music_seq = music_seq.view(b, t//music_ds_rate, c*music_ds_rate)
                        music_seq = music_seq[:, config.ds_rate//music_relative_rate:]

                        if hasattr(config, 'music_normalize') and config.music_normalize:
                            music_seq = music_seq / ( t//music_ds_rate * 1.0 )

                        zs = gpt.module.sample(x, cond=music_seq, shift=config.sample_shift if hasattr(config, 'sample_shift') else None)

                        pose_sample = vqvae.module.decode(zs)

                        if config.global_vel:
                            
                            global_vel = pose_sample[:, :, :3].clone()
                            pose_sample[:, 0, :3] = 0
                            for iii in range(1, pose_sample.size(1)):
                                pose_sample[:, iii, :3] = pose_sample[:, iii-1, :3] + global_vel[:, iii-1, :]

                        if isinstance(zs, tuple):
                            quants_out[self.dance_names[i_eval]] = tuple(zs[ii][0].cpu().data.numpy()[0] for ii in range(len(zs)))
                        else:
                            quants_out[self.dance_names[i_eval]] = zs[0].cpu().data.numpy()[0]
                    
                        results.append(pose_sample)

                    visualizeAndWrite(results, config, self.visdir, self.dance_names, epoch_i, quants_out)
                gpt.train()
            self.schedular.step()

    def eval(self):
        with torch.no_grad():
            vqvae = self.model.eval()
            gpt = self.model2.eval()

            config = self.config
            # data = self.config.data
            # criterion = nn.MSELoss()

            
            checkpoint = torch.load(config.vqvae_weight)
            vqvae.load_state_dict(checkpoint['model'], strict=False)

            # config = self.config
            # model = self.model.eval()
            epoch_tested = config.testing.ckpt_epoch

            checkpoint = torch.load(config.vqvae_weight)
            vqvae.load_state_dict(checkpoint['model'], strict=False)

            ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
            self.device = torch.device('cuda' if config.cuda else 'cpu')
            print("Evaluation...")
            checkpoint = torch.load(ckpt_path)
            gpt.load_state_dict(checkpoint['model'])
            gpt.eval()

            results = []
            random_id = 0  # np.random.randint(0, 1e4)
            # quants = {}
            quants_out = {}
            for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc='Generating Dance Poses')):
                # Prepare data
                # pose_seq_eval = map(lambda x: x.to(self.device), batch_eval)
                if hasattr(config, 'demo') and config.demo:
                    music_seq = batch_eval.to(self.device)
                    quants = ([torch.ones(1, 1,).to(self.device).long() * 423], [torch.ones(1, 1,).to(self.device).long() * 12])
                else:
                    music_seq, pose_seq = batch_eval
                    music_seq = music_seq.to(self.device)
                    pose_seq = pose_seq.to(self.device)
                
                    quants = vqvae.module.encode(pose_seq)
                # print(pose_seq.size())
                if isinstance(quants, tuple):
                    x = tuple(quants[i][0][:, :1].clone() for i in range(len(quants)))
                else:
                    x = quants[0][:, :1].clone()
                
                if hasattr(config, 'random_init_test') and config.random_init_test:
                    if isinstance(quants, tuple):
                        for iij in range(len(x)):
                            x[iij][:, 0] = torch.randint(512, (1, ))
                    else:
                        x[:, 0] = torch.randint(512, (1, ))

                music_ds_rate = config.ds_rate if not hasattr(config, 'external_wav') else config.external_wav_rate
                music_ds_rate = config.music_ds_rate if hasattr(config, 'music_ds_rate') else music_ds_rate
                music_relative_rate = config.music_relative_rate if hasattr(config, 'music_relative_rate') else config.ds_rate
                
                music_seq = music_seq[:, :, :config.structure_generate.n_music//music_ds_rate].contiguous().float()
                b, t, c = music_seq.size()
                music_seq = music_seq.view(b, t//music_ds_rate, c*music_ds_rate)
                music_relative_rate = config.music_relative_rate if hasattr(config, 'music_relative_rate') else config.ds_rate
                

                music_seq = music_seq[:, config.ds_rate//music_relative_rate:]
                # it is just music_seq[:, 1:], ignoring the first music feature
                
                if hasattr(config, 'music_normalize') and config.music_normalize:
                    music_seq = music_seq / ( t//music_ds_rate * 1.0 )
                # print(music_seq.size())

                # block_size = gpt.module.get_block_size()

                zs = gpt.module.sample(x, cond=music_seq, shift=config.sample_shift if hasattr(config, 'sample_shift') else None)

                pose_sample = vqvae.module.decode(zs)

                if config.global_vel:
                    print('!!!!!')
                    global_vel = pose_sample[:, :, :3].clone()
                    pose_sample[:, 0, :3] = 0
                    for iii in range(1, pose_sample.size(1)):
                        pose_sample[:, iii, :3] = pose_sample[:, iii-1, :3] + global_vel[:, iii-1, :]

                results.append(pose_sample)
                if isinstance(zs, tuple):
                    quants_out[self.dance_names[i_eval]] = tuple(zs[ii][0].cpu().data.numpy()[0] for ii in range(len(zs))) 
                else:
                    quants_out[self.dance_names[i_eval]] = zs[0].cpu().data.numpy()[0]

            visualizeAndWrite(results, config, self.evaldir, self.dance_names, epoch_tested, quants_out)
    def visgt(self,):
        config = self.config
        print("Visualizing ground truth")

        results = []
        random_id = 0  # np.random.randint(0, 1e4)
        
        for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc='Generating Dance Poses')):
            # Prepare data
            # pose_seq_eval = map(lambda x: x.to(self.device), batch_eval)
            _, pose_seq_eval = batch_eval

            results.append(pose_seq_eval)
            # moduel.module.encode

            # quants = model.module.encode(src_pos_eval)[0].cpu().data.numpy()[0]

                    # exit()
        # weights = np.histogram(all_quants, bins=1, range=[0, config.structure.l_bins], normed=False, weights=None, density=None)
        visualizeAndWrite(results, config,self.gtdir, self.dance_names, 0)

    def analyze_code(self,):
        config = self.config
        print("Analyzing codebook")

        epoch_tested = config.testing.ckpt_epoch
        ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model'])
        model = self.model.eval()

        training_data = self.training_data
        all_quants = None

        torch.cuda.manual_seed(config.seed)
        self.device = torch.device('cuda' if config.cuda else 'cpu')
        random_id = 0  # np.random.randint(0, 1e4)
        
        for i_eval, batch_eval in enumerate(tqdm(self.training_data, desc='Generating Dance Poses')):
            # Prepare data
            # pose_seq_eval = map(lambda x: x.to(self.device), batch_eval)
            pose_seq_eval = batch_eval.to(self.device)

            quants = model.module.encode(pose_seq_eval)[0].cpu().data.numpy()
            all_quants = np.append(all_quants, quants.reshape(-1)) if all_quants is not None else quants.reshape(-1)

        print(all_quants)
                    # exit()
        # visualizeAndWrite(results, config,self.gtdir, self.dance_names, 0)
        plt.hist(all_quants, bins=config.structure.l_bins, range=[0, config.structure.l_bins])

        log = datetime.datetime.now().strftime('%Y-%m-%d')
        plt.savefig(self.histdir1 + '/hist_epoch_' + str(epoch_tested)  + '_%s.jpg' % log)   #图片的存储
        plt.close()

    def sample(self,):
        config = self.config
        print("Analyzing codebook")

        epoch_tested = config.testing.ckpt_epoch
        ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model'])
        model = self.model.eval()

        quants = {}

        results = []

        if hasattr(config, 'analysis_array') and config.analysis_array is not None:
            # print(config.analysis_array)
            names = [str(ii) for ii in config.analysis_array]
            print(names)
            for ii in config.analysis_array:
                print(ii)
                zs =  [(ii * torch.ones((1, self.config.sample_code_length), device='cuda')).long()]
                print(zs[0].size())
                pose_sample = model.module.decode(zs)
                if config.global_vel:
                    global_vel = pose_sample[:, :, :3]
                    pose_sample[:, 0, :3] = 0
                    for iii in range(1, pose_sample.size(1)):
                        pose_sample[:, iii, :3] = pose_sample[:, iii-1, :3] + global_vel[:, iii-1, :]

                quants[str(ii)] = zs[0].cpu().data.numpy()[0]

                results.append(pose_sample)
        else:
            names = ['rand_seq_' + str(ii) for ii in range(10)]
            for ii in range(10):
                zs = [torch.randint(0, self.config.structure.l_bins, size=(1, self.config.sample_code_length), device='cuda')]
                pose_sample = model.module.decode(zs)
                quants['rand_seq_' + str(ii)] = zs[0].cpu().data.numpy()[0]
                if config.global_vel:
                    global_vel = pose_sample[:, :, :3]
                    pose_sample[:, 0, :3] = 0
                    for iii in range(1, pose_sample.size(1)):
                        pose_sample[:, iii, :3] = pose_sample[:, iii-1, :3] + global_vel[:, iii-1, :]
                results.append(pose_sample)
        visualizeAndWrite(results, config, self.sampledir, names, epoch_tested, quants)

    def _build(self):
        config = self.config
        self.start_epoch = 0
        self._dir_setting()
        self._build_model()
        if not(hasattr(config, 'need_not_train_data') and config.need_not_train_data):
            self._build_train_loader()
        if not(hasattr(config, 'need_not_test_data') and config.need_not_train_data):      
            self._build_test_loader()
        self._build_optimizer()

    def _build_model(self):
        """ Define Model """
        config = self.config 
        if hasattr(config.structure, 'name') and hasattr(config.structure_generate, 'name'):
            print(f'using {config.structure.name} and {config.structure_generate.name} ')
            model_class = getattr(models, config.structure.name)
            model = model_class(config.structure)

            model_class2 = getattr(models, config.structure_generate.name)
            model2 = model_class2(config.structure_generate)
        else:
            raise NotImplementedError("Wrong Model Selection")
        
        model = nn.DataParallel(model)
        model2 = nn.DataParallel(model2)
        self.model2 = model2.cuda()
        self.model = model.cuda()

    def _build_train_loader(self):

        config = self.config
        trainset = MusicDanceDataset(config.train_data)

        self.training_data = torch.utils.data.DataLoader(
            trainset,
            num_workers=8,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True)



    def _build_test_loader(self):

        config = self.config
        testset = MusicDanceDataset(config.test_data)
        self.test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=1,
            shuffle=False
        )
        self.dance_names = testset.fnames

    def _build_optimizer(self):
        #model = nn.DataParallel(model).to(device)
        config = self.config.optimizer
        try:
            optim = getattr(torch.optim, config.type)
        except Exception:
            raise NotImplementedError('not implemented optim method ' + config.type)

        self.optimizer = optim(itertools.chain(self.model2.module.parameters(),
                                             ),
                                             **config.kwargs)
        self.schedular = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **config.schedular_kwargs)

    def _dir_setting(self):
        # data = self.config.data
        self.expname = self.config.expname
        self.experiment_dir = os.path.join("./", "experiments")
        self.expdir = os.path.join(self.experiment_dir, self.expname)

        if not os.path.exists(self.expdir):
            os.mkdir(self.expdir)

        self.visdir = os.path.join(self.expdir, "vis")  # -- imgs, videos, jsons
        if not os.path.exists(self.visdir):
            os.mkdir(self.visdir)

        self.jsondir = os.path.join(self.visdir, "jsons")  # -- imgs, videos, jsons
        if not os.path.exists(self.jsondir):
            os.mkdir(self.jsondir)

        self.histdir = os.path.join(self.visdir, "hist")  # -- imgs, videos, jsons
        if not os.path.exists(self.histdir):
            os.mkdir(self.histdir)

        self.imgsdir = os.path.join(self.visdir, "imgs")  # -- imgs, videos, jsons
        if not os.path.exists(self.imgsdir):
            os.mkdir(self.imgsdir)

        self.videodir = os.path.join(self.visdir, "videos")  # -- imgs, videos, jsons
        if not os.path.exists(self.videodir):
            os.mkdir(self.videodir)
        
        self.ckptdir = os.path.join(self.expdir, "ckpt")
        if not os.path.exists(self.ckptdir):
            os.mkdir(self.ckptdir)

        self.evaldir = os.path.join(self.expdir, "eval")
        if not os.path.exists(self.evaldir):
            os.mkdir(self.evaldir)

        self.gtdir = os.path.join(self.expdir, "gt")
        if not os.path.exists(self.gtdir):
            os.mkdir(self.gtdir)

        self.jsondir1 = os.path.join(self.evaldir, "jsons")  # -- imgs, videos, jsons
        if not os.path.exists(self.jsondir1):
            os.mkdir(self.jsondir1)

        self.histdir1 = os.path.join(self.evaldir, "hist")  # -- imgs, videos, jsons
        if not os.path.exists(self.histdir1):
            os.mkdir(self.histdir1)

        self.imgsdir1 = os.path.join(self.evaldir, "imgs")  # -- imgs, videos, jsons
        if not os.path.exists(self.imgsdir1):
            os.mkdir(self.imgsdir1)

        self.videodir1 = os.path.join(self.evaldir, "videos")  # -- imgs, videos, jsons
        if not os.path.exists(self.videodir1):
            os.mkdir(self.videodir1)

        self.sampledir = os.path.join(self.evaldir, "samples")  # -- imgs, videos, jsons
        if not os.path.exists(self.sampledir):
            os.mkdir(self.sampledir)




        
def prepare_dataloader(music_data, dance_data, batch_size):
    data_loader = torch.utils.data.DataLoader(
        MoDaSeq(music_data, dance_data),
        num_workers=8,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
                # collate_fn=paired_collate_fn,
    )

    return data_loader










 


