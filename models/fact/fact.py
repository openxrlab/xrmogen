"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
# logger = logging.getLogger(__name__)
class GELU(nn.Module): 
    def __init__(self): 
        super(GELU, self).__init__() 
    def forward(self, x): 
        return 0.5*x*(1+F.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # regularization
        # self.attn_drop = nn.Dropout(config.attn_pdrop)
        # self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence

        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        # att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            # nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Transformer(nn.Module):
    """  Transformer used in FACT"""

    def __init__(self, config):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # self.block_size = config.block_size
        self.apply(self._init_weights)

        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size
    

    # tensorflow init --> Glorot Uniform 
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # module.weight.data.normal_(mean=0.0, std=0.02)
            module.weight.data.uniform_(math.sqrt(6.0/sum(module.weight.size())))
            # truncated_normal_(module.weight, 0.0, 0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        # elif isinstance(module, nn.LayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.blocks(x)
        return x


class FACT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.music_emb = nn.Embedding(config.vocab_size, config.n_embd - config.n_music)
        self.music_linear_emb = nn.Linear(config.n_music, config.n_embd)
        self.music_pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.motion_linear_emb = nn.Linear(config.n_motion, config.n_embd)
        self.motion_pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))

        self.music_transformer = Transformer(config.music)
        self.motion_transformer = Transformer(config.motion)
        self.cross_model_transformer = Transformer(config.cross_model)
        
        self.proj = nn.Linear(config.n_embd, config.n_output)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            truncated_normal_(module.weight, 0.0, 0.02)
            # module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Parameter):
            # module.data.normal_(mean=0.0, std=0.02)
            truncated_normal_(module.weight, 0.0, 0.02)
        # elif isinstance(module, nn.LayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
    
    def forward(self, music, motion):
        music_feat = self.music_transformer(self.music_linear_emb(music) + self.music_pos_emb[:, :music.size(1)])
        motion_feat = self.motion_transformer(self.motion_linear_emb(motion) + self.motion_pos_emb[:, :motion.size(1)])
        output = self.cross_model_transformer(torch.cat([music_feat, motion_feat], dim=1))

        return self.proj(output)

    def generate(self, music, start_motion):
        """input: when generate, since music length can be different so batch size must be 1"""
        _, T, _ = music.shape
        motion = start_motion

        # pad last 120 frames of music to keep generated motion same len as original music 
        music = torch.cat([music, music[:, -1:].repeat(1, 120, 1)], dim=1)

        for tt in range(T - 240):
            this_music = music[:, tt:tt+240, :]
            this_motion = motion[:, tt:tt+120, :]
            this_output = self.forward(this_music, this_motion)
            motion = torch.cat([motion, this_output[:, :1, ]], 1)
        return motion

