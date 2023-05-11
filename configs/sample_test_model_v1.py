import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.pred = 'noise_pred'
    config.z_shape = (4, 16, 16)
    config.clip_img_dim = 32
    config.clip_text_dim = 32
    config.text_dim = 32  # reduce dimension
    config.data_type = 1

    config.autoencoder = d(
        pretrained_path='models/test/autoencoder_kl.pth',
    )

    config.caption_decoder = d(
        tokenizer_pretrained_path="models/clip_caption_model",
        ckpt_pretrained_path="models/test/caption_decoder.pth",
        gpt_hidden_dim=config.get_ref('clip_text_dim'),
        hidden_dim=config.get_ref('text_dim')
    )

    config.nnet = d(
        name='uvit_multi_post_ln_v1',
        img_size=16,
        in_chans=4,
        patch_size=2,
        embed_dim=16,
        depth=2,
        num_heads=2,
        mlp_ratio=4,
        qkv_bias=False,
        pos_drop_rate=0.,
        drop_rate=0.,
        attn_drop_rate=0.,
        mlp_time_embed=False,
        text_dim=config.get_ref('text_dim'),
        num_text_tokens=77,
        clip_img_dim=config.get_ref('clip_img_dim'),
        use_checkpoint=True
    )

    config.sample = d(
        sample_steps=2,
        scale=5.,  # w (imagen) = 1 + s (unidiffuser)
        t2i_cfg_mode='true_uncond',
        device="cpu",
        log_level="info", 
    )

    return config
