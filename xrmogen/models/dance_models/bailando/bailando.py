
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

    def __init__(self, model_config):
        super().__init__()
        self.bailando_phase = model_config['bailando_phase']
        self.vqvae = SepVQVAER(model_config.vqvae)
        self.gpt = CrossCondGPT(model_config.gpt)
        
        # self.val_results = {}
    
    def train_step(self, data, optimizer, **kwargs):
        train_phase = self.bailando_phase

        music_seq, pose_seq  = data['music'], data['dance'] 

        optimizer.zero_grad()

        if train_phase == 'motion vqvae':
            self.vqvae.train()
            pose_seq[:, :, :3] = 0
            out, loss, metrics = self.vqvae(pose_seq, train_phase)

        elif train_phase == 'global velocity':
            self.vqvae.train()
            pose_seq[:, :-1, :3] = pose_seq[:, 1:, :3] - pose_seq[:, :-1, :3]
            pose_seq[:, -1, :3] = pose_seq[:, -2, :3]
            pose_seq = pose_seq.clone().detach()

            out, loss, metrics = self.vqvae(pose_seq, train_phase)

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

            out, loss = self.gpt(quants_input, music_seq[:, 1:], quants_target)

        else:
            raise NotImplementedError

        stats = {
            'loss': loss.item()
        }
        outputs = {
            'loss': loss,
            'log_vars': stats,
            'num_samples': out.size(1)
        }

        # loss.backward()
        # optimizer.step()
        return outputs

    def val_step(self, data, optimizer, **kwargs):
        return self.test_step(data, optimizer, **kwargs)

    
    def test_step(self, data, optimizer, **kwargs):
        test_phase = self.bailando_phase

        music_seq, pose_seq = data['music'], data['dance']
        self.eval()
        
        results = []

        pose_seq[:, :, :3] = 0
        with torch.no_grad():
            if test_phase == 'motion vqvae':
                
                # print(pose_seq[0, 7, 6], )
                pose_seq[:, :, :3] = 0
                pose_seq_out, _, _ = self.vqvae(pose_seq, test_phase)
                results.append(pose_seq_out) 

            elif test_phase == 'global velocity':
                pose_seq[:, :-1, :3] = pose_seq[:, 1:, :3] - pose_seq[:, :-1, :3]
                pose_seq[:, -1, :3] = pose_seq[:, -2, :3]
                pose_seq = pose_seq.clone().detach()

                pose_seq_out, _, _ = self.vqvae(pose_seq, test_phase)

                n, t, c = pose_seq_out.size()
                pose_seq_out = pose_seq_out.view(n, t, c//3, 3)
                global_vel = pose_seq_out[:, :, :1, :].clone()
                pose_seq_out[:, 0, :1, :] = 0
                for iii in range(1, pose_seq_out.size(1)):
                    pose_seq_out[:, iii, :, :] = pose_seq_out[:, iii-1, :, :] + global_vel[:, iii-1, :, :]
                results.append(pose_seq_out.view(n, t, c)) 

            elif test_phase == 'gpt':
                
                quants = self.vqvae.module.encode(pose_seq)

                if isinstance(quants, tuple):
                    x = tuple(quants[i][0][:, :1].clone() for i in range(len(quants)))
                else:
                    x = quants[0][:, :1].clone()

                zs = self.gpt.module.sample(x, cond=music_seq)

                pose_sample = self.vqvae.module.decode(zs)
                n, t, c = pose_sample.size()
                pose_sample = pose_sample.view(n, t, c//3, 3)

                global_vel = pose_sample[:, :, :1, :].clone()
                pose_sample[:, 0, :1, :] = 0
                for iii in range(1, pose_sample.size(1)):
                    pose_sample[:, iii, :, :] = pose_sample[:, iii-1, :, :] + global_vel[:, iii-1, :, :]

                results.append(pose_sample.view(n, t, c))
            else:
                raise NotImplementedError
        
        # self.val_results.update({data['file_names'][0]: results[0]})
        outputs = {
            'output_pose': results[0],
            'file_name': data['file_names'][0]
        }

        return outputs


    

