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
from dataset.dataset import MusicDanceDataset
import models
from utils.log import Logger
from utils.functional import visualizeAndWrite
from torch.optim import *
import warnings
from tqdm import tqdm
import itertools
import pdb


warnings.filterwarnings('ignore')

# a, b, c, d = check_data_distribution('/mnt/lustre/lisiyao1/dance/dance2/DanceRevolution/data/aistpp_train')



class Music2Dance():
    def __init__(self, args):


        self.config = args
        torch.backends.cudnn.benchmark = True
        self._build()

    def train(self):
        model = self.model.train()
        config = self.config
        data = self.config.data
        criterion = nn.MSELoss()
        training_data = self.training_data
        test_loader = self.test_loader
        optimizer = self.optimizer
        log = Logger(self.config, self.expdir)
        updates = 0
        
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        #if args.cuda:
        torch.cuda.manual_seed(config.seed)
        self.device = torch.device('cuda' if config.cuda else 'cpu')
        if hasattr(config, 'init_weight') and config.init_weight is not '':
            model.load_state_dict(torch.load(config.init_weight)['model'])

        # Training Loop
        for epoch_i in range(1, config.epoch + 1):
            log.set_progress(epoch_i, len(training_data))

            model.train()
            
            for batch_i, batch in enumerate(training_data):

                music_seq, pose_seq  = batch 

                aud_seq = music_seq.to(self.device)
                pose_seq = pose_seq.to(self.device)

                gold_seq = pose_seq[:, 120:140] 
                src_aud = aud_seq[:, :]

                src_pos = pose_seq[:, :120]


                optimizer.zero_grad()

                output = model(src_aud, src_pos)


                loss = criterion(output[:, :20], gold_seq)
                loss.backward()

                # update parameters
                optimizer.step()

                stats = {
                    'updates': updates,
                    'loss': loss.item()
                }
                #if epoch_i % self.config.log_per_updates == 0:
                log.update(stats)
                updates += 1

            checkpoint = {
                'model': model.state_dict(),
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
                    model = model.eval()
                    results = []
                    
                    for i_eval, batch_eval in enumerate(tqdm(test_loader, desc='Generating Dance Poses')):
                        # Prepare data
                        aud_seq_eval, pose_seq_eval = batch_eval
                        aud_seq_eval = aud_seq_eval.to(self.device)
                        pose_seq_eval = pose_seq_eval.to(self.device)
                        
                        src_pos_eval = pose_seq_eval[:, :120] #

                        pose_seq_out = model.generate(aud_seq_eval, src_pos_eval)  # first 20 secs
                        results.append(pose_seq_out)
                   
                        # exit()

                    visualizeAndWrite(results, config, self.visdir, self.dance_names,epoch_i, None)
                    
                model.train()
            self.schedular.step()



    def eval(self):
        with torch.no_grad():
            config = self.config

            epoch_tested = config.testing.ckpt_epoch

            ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")

            print("Evaluation...")
            checkpoint = torch.load(ckpt_path)
            self.model.load_state_dict(checkpoint['model'])
            self.model.eval()

            results = []
            random_id = 0  # np.random.randint(0, 1e4)
            
            for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc='Generating Dance Poses')):
                # Prepare data
                aud_seq_eval, pose_seq_eval = batch_eval
                aud_seq_eval = aud_seq_eval.to(self.device)
                pose_seq_eval = pose_seq_eval.to(self.device)
                
                src_aud_eval = aud_seq_eval[:, :] # arbituary length when test
                src_pos_eval = pose_seq_eval[:, :120] #

                pose_seq_out = self.model.generate(aud_seq_eval, src_pos_eval)
                results.append(pose_seq_out)

            visualizeAndWrite(results, config, self.evaldir, self.dance_names, epoch_tested)

    def _build(self):
        config = self.config
        self.start_epoch = 0
        self._dir_setting()
        self._build_model()
        self._build_train_loader()
        self._build_test_loader()
        self._build_optimizer()

    def _build_model(self):
        """ Define Model """
        config = self.config
        if hasattr(config.structure, 'name'):
            print(f'using {config.structure.name}')
            model_class = getattr(models, config.structure.name)
            model = model_class(config.structure)
        else:
            raise NotImplementedError("Wrong Model Selection")
        
        model = nn.DataParallel(model)
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

        self.optimizer = optim(itertools.chain(self.model.module.parameters(),
                                             ),
                                             **config.kwargs)
        self.schedular = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **config.schedular_kwargs)

    def _dir_setting(self):
        data = self.config.data
        self.expname = self.config.expname
        self.experiment_dir = os.path.join(os.getcwd(), "experiments")
        self.expdir = os.path.join(self.experiment_dir, self.expname)

        if not os.path.exists(self.expdir):
            os.mkdir(self.expdir)

        self.visdir = os.path.join(self.expdir, "vis")  # -- imgs, videos, jsons
        if not os.path.exists(self.visdir):
            os.mkdir(self.visdir)

        self.jsondir = os.path.join(self.visdir, "jsons")  # -- imgs, videos, jsons
        if not os.path.exists(self.jsondir):
            os.mkdir(self.jsondir)

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

        self.jsondir1 = os.path.join(self.evaldir, "jsons")  # -- imgs, videos, jsons
        if not os.path.exists(self.jsondir1):
            os.mkdir(self.jsondir1)

        self.jsondir1 = os.path.join(self.visdir, "jsons")  # -- imgs, videos, jsons
        if not os.path.exists(self.jsondir1):
            os.mkdir(self.jsondir1)

        self.jsondir1 = os.path.join(self.evaldir, "pkl")  # -- imgs, videos, jsons
        if not os.path.exists(self.jsondir1):
            os.mkdir(self.jsondir1)

        self.imgsdir1 = os.path.join(self.evaldir, "imgs")  # -- imgs, videos, jsons
        if not os.path.exists(self.imgsdir1):
            os.mkdir(self.imgsdir1)

        self.videodir1 = os.path.join(self.evaldir, "videos")  # -- imgs, videos, jsons
        if not os.path.exists(self.videodir1):
            os.mkdir(self.videodir1)



        






