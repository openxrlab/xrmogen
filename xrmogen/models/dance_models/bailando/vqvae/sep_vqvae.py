import numpy as np
import torch
import torch.nn as nn


from .vqvae import VQVAE


smpl_down = [0, 1, 2, 4,  5, 7, 8, 10, 11]
smpl_up = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


class SepVQVAE(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps

        self.chanel_num = hps.joint_channel
        self.vqvae_up = VQVAE(hps.up_half, len(smpl_up)*self.chanel_num)
        self.vqvae_down = VQVAE(hps.down_half, len(smpl_down)*self.chanel_num)



    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        """
        zs are list with two elements: z for up and z for down
        """
        if isinstance(zs, tuple):
            zup = zs[0]
            zdown = zs[1]
        else:
            zup = zs
            zdown = zs
        xup = self.vqvae_up.decode(zup)
        xdown = self.vqvae_down.decode(zdown)
        b, t, cup = xup.size()
        _, _, cdown = xdown.size()
        x = torch.zeros(b, t, (cup+cdown)//self.chanel_num, self.chanel_num).cuda()
        x[:, :, smpl_up] = xup.view(b, t, cup//self.chanel_num, self.chanel_num)
        x[:, :, smpl_down] = xdown.view(b, t, cdown//self.chanel_num, self.chanel_num)
        
        return x.view(b, t, -1)


    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        b, t, c = x.size()
        zup = self.vqvae_up.encode(x.view(b, t, c//self.chanel_num, self.chanel_num)[:, :, smpl_up].view(b, t, -1), start_level, end_level, bs_chunks)
        zdown = self.vqvae_down.encode(x.view(b, t, c//self.chanel_num, self.chanel_num)[:, :, smpl_down].view(b, t, -1), start_level, end_level, bs_chunks)
        return (zup, zdown)

    def sample(self, n_samples):
        # zs = [torch.randint(0, self.l_bins, size=(n_samples, *z_shape), device='cuda') for z_shape in self.z_shapes]
        xup = self.vqvae_up.sample(n_samples)
        xdown = self.vqvae_up.sample(n_samples)
        b, t, cup = xup.size()
        _, _, cdown = xdown.size()
        x = torch.zeros(b, t, (cup+cdown)//self.chanel_num, self.chanel_num).cuda()
        x[:, :, smpl_up] = xup.view(b, t, cup//self.chanel_num, self.chanel_num)
        x[:, :, smpl_down] = xdown.view(b, t, cdown//self.chanel_num, self.chanel_num)
        return x

    def forward(self, x):
        b, t, c = x.size()
        x = x.view(b, t, c//self.chanel_num, self.chanel_num)
        xup = x[:, :, smpl_up, :].view(b, t, -1)
        xdown = x[:, :, smpl_down, :].view(b, t, -1)
        # xup[:] = 0
        
        x_out_up, loss_up, metrics_up = self.vqvae_up(xup)
        x_out_down , loss_down , metrics_down  = self.vqvae_down(xdown)

        _, _, cup = x_out_up.size()
        _, _, cdown = x_out_down.size()

        xout = torch.zeros(b, t, (cup+cdown)//self.chanel_num, self.chanel_num).cuda().float()
        xout[:, :, smpl_up] = x_out_up.view(b, t, cup//self.chanel_num, self.chanel_num)
        xout[:, :, smpl_down] = x_out_down.view(b, t, cdown//self.chanel_num, self.chanel_num)

        return xout.view(b, t, -1), (loss_up + loss_down)*0.5, [metrics_up, metrics_down] 
