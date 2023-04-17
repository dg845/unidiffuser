# Make small random test models for UniDiffuser

import random

import numpy as np
import torch
from torch import nn
from transformers import (
    CLIPFeatureExtractor,
    CLIPProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModel,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from sample_test_v1 import AutoencoderKL
from libs.uvit_multi_post_ln_v1 import UViT


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_test_models():
    # Make a small random autoencoder.
    set_seed(0)
    ddconfig = dict(
        double_z=True,
        z_channels=4,
        resolution=32,
        in_channels=3,
        out_ch=3,
        ch=32,
        ch_mult=[1, 2],
        num_res_blocks=1,
        attn_resolutions=[],
        dropout=0.0
    )
    vae = AutoencoderKL(
        ddconfig=ddconfig,
        embed_dim=4,  # Not actually used in the code
        scale_factor=0.18215,
    )
    torch.save(vae.state_dict(), 'models/autoencoder_kl.pth')

    # Make a small random U-ViT noise prediction model.
    set_seed(0)
    unet = UViT(
        img_size=16,
        in_chans=4,
        patch_size=2,
        embed_dim=16,
        depth=2,
        num_heads=2,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,
        pos_drop_rate=0,
        drop_rate=0,
        attn_drop_rate=0,
        norm_layer=nn.LayerNorm,
        mlp_time_embed=False,
        use_checkpoint=False,
        text_dim=32,
        num_text_tokens=77,
        clip_img_dim=32,
    )
    torch.save(unet.state_dict(), 'models/uvit_v1.pth')

    # Make a small random CLIPTextModel
    torch.manual_seed(0)
    clip_text_encoder_config = CLIPTextConfig(
        bos_token_id=0,
        eos_token_id=2,
        hidden_size=32,
        intermediate_size=37,
        layer_norm_eps=1e-05,
        num_attention_heads=4,
        num_hidden_layers=5,
        pad_token_id=1,
        vocab_size=1000,
    )
    clip_text_encoder = CLIPTextModel(clip_text_encoder_config)
    clip_text_tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
    clip_text_encoder_dir = 'models/clip_text_encoder'
    clip_text_encoder.save_pretrained(clip_text_encoder_dir)
    clip_text_tokenizer.save_pretrained(clip_text_encoder_dir)

    # Make a small random CLIPVisionModel and accompanying CLIPProcessor
    torch.manual_seed(0)
    clip_image_encoder_config = CLIPVisionConfig(
        image_size=32,
        patch_size=2,
        num_channels=3,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
    )
    clip_image_encoder = CLIPVisionModel(clip_image_encoder_config)
    clip_feature_extractor = CLIPFeatureExtractor(crop_size=32, size=32)
    clip_processor = CLIPProcessor(clip_feature_extractor, clip_text_tokenizer)
    clip_image_encoder_dir = 'models/clip_image_encoder'
    clip_image_encoder.save_pretrained(clip_image_encoder_dir)
    clip_processor.save_pretrained(clip_image_encoder_dir)
    
    # Make a small random GPT2 clip caption model.
    set_seed(0)
    # From https://huggingface.co/hf-internal-testing/tiny-random-GPT2Model/blob/main/config.json
    clip_caption_config = GPT2Config(
        n_embd=32,
        n_layer=5,
        n_head=4,
        n_inner=37,
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_drop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
    )
    clip_caption_model = GPT2LMHeadModel(clip_caption_config)
    clip_caption_tokenizer = GPT2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-GPT2Model")
    # Save transformers model and tokenizer to output director
    clip_caption_model_dir = 'models/clip_caption_model'
    clip_caption_model.save_pretrained(clip_caption_model_dir)
    clip_caption_tokenizer.save_pretrained(clip_caption_model_dir)



def main():
    make_test_models()
    

if __name__ == '__main__':
    main()