
import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from .vqvae.sep_vqvae_root import SepVQVAER
from .gpt.cross_cond_gpt import CrossCondGPT


from ...builder import DANCE_MODELS

@DANCE_MODELS.register_module()
class Bailando(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.bailando_phase = config['bailando_phase']
        self.vqvae = SepVQVAER(config.vqvae)
        self.gpt = CrossCondGPT(config.gpt)
    
    def train_step(self, data, optimizer, **kwargs):
        train_phase = self.bailando_phase

        music_seq, pose_seq  = data 
        optimizer.zero_grad()

        if train_phase == 'motion vqvae':
            self.vqvae.train()
            pose_seq[:, :, :3] = 0
            _, loss, metrics = self.vqvae(pose_seq, train_phase)

        elif train_phase == 'global velocity':
            self.vqvae.train()
            pose_seq[:, :-1, :3] = pose_seq[:, 1:, :3] - pose_seq[:, :-1, :3]
            pose_seq[:, -1, :3] = pose_seq[:, -2, :3]
            pose_seq = pose_seq.clone().detach()

            _, loss, metrics = self.vqvae(pose_seq, train_phase)

        elif train_phase == 'gpt':
            pose_seq[:, :, :3] = 0
            self.vqvae.eval()
            self.gpt.train()
            with torch.no_grad():
                quants_pred = self.vqvae.encode(pose_seq)
                if isinstance(quants_pred, tuple):
                    quants_input = tuple(quants_pred[ii][0][:, :-1].clone().detach() for ii in range(len(quants_pred)))
                    quants_target = tuple(quants_pred[ii][0][:, 1:].clone().detach() for ii in range(len(quants_pred)))
                else:
                    quants = quants_pred[0]
                    quants_input = quants[:, :-1].clone().detach()
                    quants_target = quants[:, 1:].clone().detach()

            _, loss = self.gpt(quants_input, music_seq[:, 1:], quants_target)

        else:
            raise NotImplementedError

        stats = {
            'loss': loss.item()
        }
        outputs = {
            'loss': loss.item(),
            'log_vars': stats,
        }

        loss.backward()
        optimizer.step()
        return outputs

    def val_step(self, data, optimizer, **kwargs):
        return self.test_step(data, optimizer, **kwargs)

    
    def test_step(self, data, optimizer, **kwargs):
        test_phase = self.bailando_phase

        music_seq, pose_seq  = data 
        self.eval()
        results = []

        pose_seq[:, :, :3] = 0
        with torch.no_grad():
            if test_phase == 'motion vqvae':
                pose_seq[:, :, :3] = 0
                pose_seq_out, _, _ = self.vqvae(pose_seq, test_phase)

            elif test_phase == 'global velocity':
                pose_seq[:, :-1, :3] = pose_seq[:, 1:, :3] - pose_seq[:, :-1, :3]
                pose_seq[:, -1, :3] = pose_seq[:, -2, :3]
                pose_seq = pose_seq.clone().detach()

                pose_seq_out, _, _ = self.vqvae(pose_seq, test_phase)

                global_vel = pose_seq_out[:, :, :3].clone()
                pose_seq_out[:, 0, :3] = 0
                for iii in range(1, pose_seq_out.size(1)):
                    pose_seq_out[:, iii, :3] = pose_seq_out[:, iii-1, :3] + global_vel[:, iii-1, :]

            elif test_phase == 'gpt':
                
                quants = self.vqvae.module.encode(pose_seq)

                if isinstance(quants, tuple):
                    x = tuple(quants[i][0][:, :1].clone() for i in range(len(quants)))
                else:
                    x = quants[0][:, :1].clone()

                zs = self.gpt.module.sample(x, cond=music_seq)

                pose_sample = self.vqvae.module.decode(zs)

                global_vel = pose_sample[:, :, :3].clone()
                pose_sample[:, 0, :3] = 0
                for iii in range(1, pose_sample.size(1)):
                    pose_sample[:, iii, :3] = pose_sample[:, iii-1, :3] + global_vel[:, iii-1, :]

                results.append(pose_sample)
            else:
                raise NotImplementedError

        outputs = {
            'output_pose': results
        }


    

