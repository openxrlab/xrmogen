import os
from datetime import datetime


model = dict(
    type='Bailando',
    model_config=dict(
        bailando_phase='motion vqvae',
        vqvae=dict( 
            up_half=dict(
                levels=1,
                downs_t=[3,],
                strides_t =[2,],
                emb_width=512,
                l_bins=512,
                l_mu=0.99,
                commit=0.02,
                hvqvae_multipliers=[1,],
                width=512,
                depth=3,
                m_conv=1.0,
                dilation_growth_rate=3,
                sample_length=240,
                use_bottleneck=True,
                joint_channel=3,
                vqvae_reverse_decoder_dilation=True
            ),
            down_half=dict(
                levels=1,
                downs_t=[3,],
                strides_t =[2,],
                emb_width =512,
                l_bins =512,
                l_mu =0.99,
                commit =0.02,
                hvqvae_multipliers =[1,],
                width=512,
                depth=3,
                m_conv =1.0,
                dilation_growth_rate =3,
                sample_length=240,
                use_bottleneck=True,
                joint_channel=3,
                vqvae_reverse_decoder_dilation=True
            ),
            use_bottleneck=True,
            joint_channel=3,
        ),

        gpt=dict(
            block_size=29,
            base=dict(
                embd_pdrop=0.1,
                resid_pdrop=0.1,
                attn_pdrop=0.1,
                vocab_size_up=512,
                vocab_size_down=512,
                block_size=29,
                n_layer=6,
                n_head=12,
                n_embd=768 ,
                n_music=438,
                n_music_emb=768
            ),
            head=dict(
                embd_pdrop=0.1,
                resid_pdrop=0.1,
                attn_pdrop=0.1,
                vocab_size=512,
                block_size=29,
                n_layer=6,
                n_head=12,
                n_embd=768,
                vocab_size_up=512,
                vocab_size_down=512 
            ),
            n_music=438,
            n_music_emb=768
        )
    )
)
  