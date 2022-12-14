import numpy as np
import torch as t
import torch.nn as nn

from .bottleneck import Bottleneck, NoBottleneck
from .encdec import Decoder, Encoder, assert_shape


def dont_update(params):
    for param in params:
        param.requires_grad = False


def update(params):
    for param in params:
        param.requires_grad = True


def calculate_strides(strides, downs):
    return [stride**down for stride, down in zip(strides, downs)]


def _loss_fn(x_target, x_pred):
    return t.mean(t.abs(x_pred - x_target))


class VQVAER(nn.Module):

    def __init__(self, hps, input_dim=72):
        super().__init__()
        self.hps = hps

        input_shape = (hps.sample_length, input_dim)
        levels = hps.levels
        downs_t = hps.downs_t
        strides_t = hps.strides_t
        emb_width = hps.emb_width
        l_bins = hps.l_bins
        mu = hps.l_mu
        commit = hps.commit

        multipliers = hps.hvqvae_multipliers
        use_bottleneck = hps.use_bottleneck
        if use_bottleneck:
            print('We use bottleneck!')
        else:
            print('We do not use bottleneck!')
        if not hasattr(hps, 'dilation_cycle'):
            hps.dilation_cycle = None
        block_kwargs = dict(
            width=hps.width,
            depth=hps.depth,
            m_conv=hps.m_conv,
            dilation_growth_rate=hps.dilation_growth_rate,
            dilation_cycle=hps.dilation_cycle,
            reverse_decoder_dilation=hps.vqvae_reverse_decoder_dilation)

        self.sample_length = input_shape[0]
        x_shape, x_channels = input_shape[:-1], input_shape[-1]
        self.x_shape = x_shape

        self.downsamples = calculate_strides(strides_t, downs_t)
        self.hop_lengths = np.cumprod(self.downsamples)
        self.z_shapes = [(x_shape[0] // self.hop_lengths[level], )
                         for level in range(levels)]
        self.levels = levels

        if multipliers is None:
            self.multipliers = [1] * levels
        else:
            assert len(multipliers) == levels, 'Invalid number of multipliers'
            self.multipliers = multipliers

        def _block_kwargs(level):
            this_block_kwargs = dict(block_kwargs)
            this_block_kwargs['width'] *= self.multipliers[level]
            this_block_kwargs['depth'] *= self.multipliers[level]
            return this_block_kwargs

        encoder = lambda level: Encoder(  # noqa: E731
            x_channels, emb_width, level + 1, downs_t[:level + 1],
            strides_t[:level + 1], **_block_kwargs(level))
        decoder = lambda level: Decoder(  # noqa: E731
            x_channels, emb_width, level + 1, downs_t[:level + 1],
            strides_t[:level + 1], **_block_kwargs(level))
        decoder_root = lambda level: Decoder(  # noqa: E731
            hps.joint_channel, emb_width, level + 1, downs_t[:level + 1],
            strides_t[:level + 1], **_block_kwargs(level))
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.decoders_root = nn.ModuleList()
        for level in range(levels):
            self.encoders.append(encoder(level))
            self.decoders.append(decoder(level))
            self.decoders_root.append(decoder_root(level))

        if use_bottleneck:
            self.bottleneck = Bottleneck(l_bins, emb_width, mu, levels)
        else:
            self.bottleneck = NoBottleneck(levels)

        self.downs_t = downs_t
        self.strides_t = strides_t
        self.l_bins = l_bins
        self.commit = commit
        self.reg = hps.reg if hasattr(hps, 'reg') else 0
        self.acc = hps.acc if hasattr(hps, 'acc') else 0
        self.vel = hps.vel if hasattr(hps, 'vel') else 0
        if self.reg == 0:
            print('No motion regularization!')

    def preprocess(self, x):
        # x: NTC [-1,1] -> NCT [-1,1]
        assert len(x.shape) == 3
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # x: NTC [-1,1] <- NCT [-1,1]
        x = x.permute(0, 2, 1)
        return x

    def _decode(self, zs, start_level=0, end_level=None):
        # Decode
        if end_level is None:
            end_level = self.levels
        assert len(zs) == end_level - start_level
        xs_quantised = self.bottleneck.decode(
            zs, start_level=start_level, end_level=end_level)
        assert len(xs_quantised) == end_level - start_level

        # Use only lowest level
        decoder, decoder_root, x_quantised = self.decoders[
            start_level], self.decoders_root[start_level], xs_quantised[0:1]

        x_out = decoder(x_quantised, all_levels=False)
        x_vel_out = decoder_root(x_quantised, all_levels=False)
        x_out = self.postprocess(x_out)
        x_vel_out = self.postprocess(x_vel_out)

        _, _, cc = x_vel_out.size()
        x_out[:, :, :cc] = x_vel_out.clone()
        return x_out

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        z_chunks = [t.chunk(z, bs_chunks, dim=0) for z in zs]
        x_outs = []
        for i in range(bs_chunks):
            zs_i = [z_chunk[i] for z_chunk in z_chunks]
            x_out = self._decode(
                zs_i, start_level=start_level, end_level=end_level)
            x_outs.append(x_out)
        return t.cat(x_outs, dim=0)

    def _encode(self, x, start_level=0, end_level=None):
        # Encode
        if end_level is None:
            end_level = self.levels
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        zs = self.bottleneck.encode(xs)
        return zs[start_level:end_level]

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        x[:, :, :self.hps.joint_channel] = 0
        x_chunks = t.chunk(x, bs_chunks, dim=0)
        zs_list = []
        for x_i in x_chunks:
            zs_i = self._encode(
                x_i, start_level=start_level, end_level=end_level)
            zs_list.append(zs_i)
        zs = [t.cat(zs_level_list, dim=0) for zs_level_list in zip(*zs_list)]
        return zs

    def sample(self, n_samples):
        zs = [
            t.randint(
                0, self.l_bins, size=(n_samples, *z_shape), device='cuda')
            for z_shape in self.z_shapes
        ]
        return self.decode(zs)

    def forward(self, x, phase='motion vqvae'):

        if phase == 'global velocity':
            self.bottleneck.eval()
        with t.no_grad():

            metrics = {}

            x_zero = x.clone()
            x_zero[:, :, :self.hps.joint_channel] = 0

            # Encode/Decode
            x_in = self.preprocess(x_zero)
            xs = []
            for level in range(self.levels):
                encoder = self.encoders[level]
                if phase == 'global velocity':
                    encoder.eval()
                x_out = encoder(x_in)
                xs.append(x_out[-1])

            zs, xs_quantised, commit_losses, _ = \
                self.bottleneck(
                    xs)
            x_outs = []
            x_outs_vel = []

        for level in range(self.levels):
            decoder = self.decoders[level]
            if phase == 'global velocity':
                decoder.eval()
            decoder_root = self.decoders_root[level]
            x_out = decoder(xs_quantised[level:level + 1], all_levels=False)
            x_vel_out = decoder_root(
                xs_quantised[level:level + 1], all_levels=False)

            assert_shape(x_out, x_in.shape)
            x_outs.append(x_out)
            x_outs_vel.append(x_vel_out)

        recons_loss = t.zeros(()).to(x.device)
        velocity_loss = t.zeros(()).to(x.device)
        acceleration_loss = t.zeros(()).to(x.device)

        x_target = x_zero if phase == 'motion vqvae' else x.float(
        )[:, :, :self.hps.joint_channel]

        for level in reversed(range(self.levels)):
            x_out_vel = self.postprocess(x_outs_vel[level])
            x_out_zero = self.postprocess(x_outs[level])
            _, _, cc = x_out_vel.size()
            x_out = x_out_zero.clone().detach()
            x_out[:, :, :cc] = x_out_vel

            if phase == 'motion vqvae':
                this_recons_loss = _loss_fn(x_target, x_out_zero)
                this_velocity_loss = _loss_fn(
                    x_out_zero[:, 1:] - x_out_zero[:, :-1],
                    x_target[:, 1:] - x_target[:, :-1])
                this_acceleration_loss = _loss_fn(
                    x_out_zero[:, 2:] + x_out_zero[:, :-2] -
                    2 * x_out_zero[:, 1:-1],
                    x_target[:, 2:] + x_target[:, :-2] - 2 * x_target[:, 1:-1])
            else:
                this_recons_loss = _loss_fn(x_target, x_out_vel)
                this_velocity_loss = 0
                this_acceleration_loss = _loss_fn(
                    x_out_vel[:, 1:] - x_out_vel[:, :-1],
                    x_target[:, 1:] - x_target[:, :-1])

            metrics[f'recons_loss_l{level + 1}'] = this_recons_loss

            recons_loss += this_recons_loss
            velocity_loss += this_velocity_loss
            acceleration_loss += this_acceleration_loss

        if phase == 'motion vqvae':
            # this loss can not be split from the model due to commit_loss
            commit_loss = sum(commit_losses)
            loss = recons_loss + \
                commit_loss * self.commit + \
                self.vel * velocity_loss + \
                self.acc * acceleration_loss
        else:
            loss = recons_loss + self.acc * acceleration_loss

        with t.no_grad():
            l1_loss = _loss_fn(
                x_target, x_out_zero) if phase == 'motion vqvae' else _loss_fn(
                    x_target, x_out_vel)

        metrics.update(
            dict(
                recons_loss=recons_loss,
                l1_loss=l1_loss,
                velocity_loss=l1_loss,
                acceleration_loss=acceleration_loss))

        for key, val in metrics.items():
            metrics[key] = val.detach()

        return x_out, loss, metrics
