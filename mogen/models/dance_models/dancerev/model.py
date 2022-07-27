# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" Define the Seq2Seq Generation Network """
from os import device_encoding
import this
import numpy as np
import torch
import torch.nn as nn

from .layers import MultiHeadAttention, PositionwiseFeedForward
from ...builder import DANCE_MODELS


BOS_POSE_AIST = np.array([
    0.01340632513165474, 1.6259130239486694, -0.09833218157291412, 0.0707249641418457, 1.5451008081436157, -0.12474726885557175, -0.04773886129260063, 1.536355972290039, -0.11427298933267593, 0.015812935307621956, 1.7525817155838013, -0.12864114344120026, 0.13902147114276886, 1.1639258861541748, -0.0879698246717453, -0.10036090016365051, 1.1553057432174683, -0.08047012239694595, 0.006522613577544689, 1.8904004096984863, -0.10235153883695602, 0.07891514897346497, 0.7553867101669312, -0.20340093970298767, -0.037818294018507004, 0.7545002698898315, -0.1963980495929718, 0.00045378319919109344, 1.9454832077026367, -0.09329807013273239, 0.11616306006908417, 0.668250560760498, -0.0974099189043045, -0.05322670564055443, 0.6652328968048096, -0.07871627062559128, -0.014527007937431335, 2.159270763397217, -0.08067376166582108, 0.0712718814611435, 2.0614874362945557, -0.08859370648860931, -0.08343493938446045, 2.0597264766693115, -0.09117652475833893, -0.002253010869026184, 2.244560718536377, -0.024742677807807922, 0.19795098900794983, 2.098480463027954, -0.09858542680740356, -0.20080527663230896, 2.0911219120025635, -0.0731159895658493, 0.30632710456848145, 1.8656978607177734, -0.09286995232105255, -0.3086402714252472, 1.8520605564117432, -0.06464222073554993, 0.25927090644836426, 1.632638931274414, 0.02665536105632782, -0.2640104591846466, 1.6051883697509766, 0.0331537127494812, 0.2306937426328659, 1.5523173809051514, 0.051218822598457336, -0.24223697185516357, 1.5211939811706543, 0.05606864392757416
])

BOS_POSE_AIST_ROT = np.concatenate([
    np.array([0.05072649,  1.87570345, -0.24885127]),  # root
    np.concatenate([np.eye(3).reshape(-1)] * 24)
])

def get_non_pad_mask(seq):
    assert seq.dim() == 3
    non_pad_mask = torch.abs(seq).sum(2).ne(0).type(torch.float)
    return non_pad_mask.unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """
    len_q = seq_q.size(1)
    padding_mask = torch.abs(seq_k).sum(2).eq(0)  # sum the vector of last dim and then judge
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq, sliding_windown_size):
    """ For masking out the subsequent info. """
    batch_size, seq_len, _ = seq.size()
    mask = torch.ones((seq_len, seq_len), device=seq.device, dtype=torch.uint8)

    mask = torch.triu(mask, diagonal=-sliding_windown_size)
    mask = torch.tril(mask, diagonal=sliding_windown_size)
    mask = 1 - mask
    # print(mask)
    return mask.bool()


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None, non_pad_mask=None):

        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        # enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        # enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, max_seq_len=1800, input_size=20, d_word_vec=10,
            n_layers=6, n_head=8, d_k=64, d_v=64,
            d_model=10, d_inner=256, dropout=0.1):

        super().__init__()

        self.d_model = d_model
        n_position = max_seq_len + 1

        self.src_emb = nn.Linear(input_size, d_word_vec)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, mask=None, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=mask)
    
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    def __init__(self, input_size=274, d_word_vec=150, hidden_size=200,
                 dropout=0.1, encoder_d_model=200, rotmat=False):
        super().__init__()

        self.input_size = input_size
        self.d_word_vec = d_word_vec
        self.hidden_size = hidden_size
      
        self.tgt_emb = nn.Linear(input_size, d_word_vec)
        self.dropout = nn.Dropout(dropout)
        self.encoder_d_model = encoder_d_model

        self.lstm1 = nn.LSTMCell(d_word_vec, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm3 = nn.LSTMCell(hidden_size, hidden_size)
        self.rotmat = rotmat

    def init_state(self, bsz, device):
        c0 = torch.randn(bsz, self.hidden_size).to(device)
        c1 = torch.randn(bsz, self.hidden_size).to(device)
        c2 = torch.randn(bsz, self.hidden_size).to(device)
        h0 = torch.randn(bsz, self.hidden_size).to(device)
        h1 = torch.randn(bsz, self.hidden_size).to(device)
        h2 = torch.randn(bsz, self.hidden_size).to(device)

        vec_h = [h0, h1, h2]
        vec_c = [c0, c1, c2]

        if self.rotmat:
            bos = BOS_POSE_AIST_ROT
            bos = np.tile(bos, (bsz, 1))
        else:
            bos = BOS_POSE_AIST 
            bos = np.tile(bos, (bsz, 1))
            root = bos[:, :3] 
            bos = bos - np.tile(root, (1, 24)) 
            bos[:, :3] = root
        out_frame = torch.from_numpy(bos).float().to(device)
        out_seq = torch.FloatTensor(bsz, 1).to(device)

        return (vec_h, vec_c), out_frame, out_seq

    def forward(self, in_frame, vec_h, vec_c): 

        in_frame = self.tgt_emb(in_frame)
        in_frame = self.dropout(in_frame)

        vec_h0, vec_c0 = self.lstm1(in_frame, (vec_h[0], vec_c[0]))
        vec_h1, vec_c1 = self.lstm2(vec_h[0], (vec_h[1], vec_c[1]))
        vec_h2, vec_c2 = self.lstm3(vec_h[1], (vec_h[2], vec_c[2]))

        vec_h_new = [vec_h0, vec_h1, vec_h2]
        vec_c_new = [vec_c0, vec_c1, vec_c2]
        return vec_h2, vec_h_new, vec_c_new

@DANCE_MODELS.register_module()
class DanceRevolution(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model_args = args

        encoder = Encoder(max_seq_len=args.max_seq_len,
            input_size=args.d_frame_vec,
            d_word_vec=args.frame_emb_size,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            d_model=args.d_model,
            d_inner=args.d_inner,
            dropout=args.dropout)

        decoder = Decoder(input_size=args.d_pose_vec,
            d_word_vec=args.pose_emb_size,
            hidden_size=args.d_inner,
            encoder_d_model=args.d_model,
            dropout=args.dropout,
            rotmat=args.rotmat
            )


        condition_step=args.condition_step
        sliding_windown_size=args.sliding_windown_size
        lambda_v=args.lambda_v


        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(decoder.hidden_size + encoder.d_model, decoder.input_size)

        self.condition_step = condition_step
        self.sliding_windown_size = sliding_windown_size
        self.lambda_v = lambda_v
        device = torch.device('cuda' if args.cuda else 'cpu')
        self.device = device
        

    def init_decoder_hidden(self, bsz):
        return self.decoder.init_state(bsz, self.device)

    # dynamic auto-condition + self-attention mask
    def forward(self, src_seq, tgt_seq, epoch_i):
        bsz, seq_len, _ = tgt_seq.size()
        src_pos = (torch.arange(seq_len).long() + 1)[None].expand(bsz, -1).detach().to(self.device)
        hidden, dec_output, out_seq = self.init_decoder_hidden(tgt_seq.size(0))
        # forward
       
        vec_h, vec_c = hidden

        enc_mask = get_subsequent_mask(src_seq, self.sliding_windown_size)
        enc_outputs, *_ = self.encoder(src_seq, src_pos, mask=enc_mask)

        groundtruth_mask = torch.ones(seq_len, self.condition_step)
        prediction_mask = torch.zeros(seq_len, int(epoch_i * self.lambda_v))
        mask = torch.cat([prediction_mask, groundtruth_mask], 1).view(-1)[:seq_len]  # for random

        preds = []
        for i in range(seq_len):
            dec_input = tgt_seq[:, i] if mask[i] == 1 else dec_output.detach()  # dec_output
            dec_output, vec_h, vec_c = self.decoder(dec_input, vec_h, vec_c)
            dec_output = torch.cat([dec_output, enc_outputs[:, i]], 1)
            dec_output = self.linear(dec_output)
            preds.append(dec_output)

        outputs = [z.unsqueeze(1) for z in preds]
        outputs = torch.cat(outputs, dim=1)
        return outputs
    
    def generate(self, src_seq,):
        """ Generate dance pose in one batch """
        with torch.no_grad():
            # Use the pre-defined begin of pose (BOP) to generate whole sequence
            bsz, src_seq_len, _ = src_seq.size()
            src_pos = (torch.arange(src_seq_len).long() + 1)[None].expand(bsz, -1).detach().to(self.device)
            # bsz, tgt_seq_len, dim = tgt_seq.size()
            tgt_seq_len = 1
            generated_frames_num = src_seq_len - tgt_seq_len

            hidden, dec_output, out_seq = self.init_decoder_hidden(bsz)
            vec_h, vec_c = hidden

            enc_mask = get_subsequent_mask(src_seq, self.model_args.sliding_windown_size)
            enc_outputs, *_ = self.encoder(src_seq, src_pos, enc_mask)

            preds = []
            for i in range(tgt_seq_len):
                # dec_input = tgt_seq[:, i]
                dec_input = dec_output
                dec_output, vec_h, vec_c = self.decoder(dec_input, vec_h, vec_c)
                dec_output = torch.cat([dec_output, enc_outputs[:, i]], 1)
                dec_output = self.linear(dec_output)
                preds.append(dec_output)

            for i in range(generated_frames_num):
                dec_input = dec_output
                dec_output, vec_h, vec_c = self.decoder(dec_input, vec_h, vec_c)
                dec_output = torch.cat([dec_output, enc_outputs[:, i + tgt_seq_len]], 1)
                dec_output = self.linear(dec_output)
                preds.append(dec_output)

        outputs = [z.unsqueeze(1) for z in preds]
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def train_step(self, data, optimizer, **kwargs):
        self.encoder.train()
        self.decoder.train()
        self.linear.train()
        
        aud_seq, pose_seq  = data

        gold_seq = pose_seq[:, 1:] 
        src_aud = aud_seq[:, :-1]
        src_pos = pose_seq[:, :-1]

        optimizer.zero_grad()

        output = self.forward(src_aud, src_pos, epoch_i)

        loss = torch.nn.functional.mse_loss(output, gold_seq)
        loss.backward()

        # update parameters
        optimizer.step()

        stats = {
            'loss': loss.item()
        }

        outputs = {
            'loss': loss.item(),
            'log_vars': stats,
        }
        
        return outputs

    def val_step(self, data, optimizer, **kwargs):
        return self.test_step(data, optimizer, **kwargs)

    def test_step(self, data, optimizer, **kwargs):
        results = []
        self.eval()
        with torch.no_grad():            
            aud_seq_eval, pose_seq_eval = batch_eval
            
            pose_seq_out = model.module.generate(aud_seq_eval)  # first 20 secs
            results.append(pose_seq_out)
        outputs = {
            'output_pose': results
        }
        return outputs

        

