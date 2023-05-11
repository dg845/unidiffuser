import ml_collections
import torch
import random
import utils
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from absl import logging
import einops
import libs.autoencoder
import libs.clip
from torchvision.utils import save_image, make_grid
import torchvision.transforms as standard_transforms
import numpy as np
import clip
from PIL import Image
import time

# New imports

from typing import Optional, Union, List, Tuple
from collections import OrderedDict

from torch import nn
from transformers import (
    CLIPFeatureExtractor,
    CLIPProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModel,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from libs.autoencoder import FrozenAutoencoderKL, Encoder, Decoder
from libs.clip import AbstractEncoder
from libs.caption_decoder import generate2, generate_beam


# ----Define Testing Versions of Classes----
class AutoencoderKL(nn.Module):
    def __init__(self, ddconfig, embed_dim, scale_factor=0.18215):
        super().__init__()
        print(f'Create autoencoder with scale_factor={scale_factor}')
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor
        # m, u = self.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
        # assert len(m) == 0 and len(u) == 0
        # self.eval()
        # self.requires_grad_(False)

    def encode_moments(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return moments

    def sample(self, moments):
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(mean)
        z = self.scale_factor * z
        return z

    def encode(self, x):
        moments = self.encode_moments(x)
        z = self.sample(moments)
        return z

    def decode(self, z):
        z = (1. / self.scale_factor) * z
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, inputs, fn):
        if fn == 'encode_moments':
            return self.encode_moments(inputs)
        elif fn == 'encode':
            return self.encode(inputs)
        elif fn == 'decode':
            return self.decode(inputs)
        else:
            raise NotImplementedError
        

class TestAutoencoderKL(nn.Module):
    def __init__(self, ddconfig, embed_dim, pretrained_path, scale_factor=0.18215):
        super().__init__()
        print(f'Create autoencoder with scale_factor={scale_factor}')
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor
        m, u = self.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
        assert len(m) == 0 and len(u) == 0
        self.eval()
        self.requires_grad_(False)

    def encode_moments(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return moments
    
    def sample(self, moments):
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(mean)
        z = self.scale_factor * z
        return z
    
    def get_moment_params(self, moments):
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean, logvar

    def encode(self, x):
        moments = self.encode_moments(x)
        # z = self.sample(moments)
        # Instead of sampling from the diagonal gaussian, return its mode (mean)
        mean, logvar = self.get_moment_params(moments)
        return mean

    def decode(self, z):
        z = (1. / self.scale_factor) * z
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, inputs, fn):
        if fn == 'encode_moments':
            return self.encode_moments(inputs)
        elif fn == 'encode':
            return self.encode(inputs)
        elif fn == 'decode':
            return self.decode(inputs)
        else:
            raise NotImplementedError
    
    def freeze(self):
        self.eval()
        self.requires_grad_(False)


class CLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(
            self,
            tokenizer_pretrained_path="hf-internal-testing/tiny-random-clip",
            ckpt_pretrained_path="models/test_clip_text_model.pth",
            device="cuda",
            max_length=77
        ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_pretrained_path)
        self.transformer = CLIPTextModel.from_pretrained(ckpt_pretrained_path)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class TestClipCaptionModel(nn.Module):
    """
    Test version of ClipCaptionModel.
    """
    def __init__(
            self,
            prefix_length: int,
            tokenizer_pretrained_path,
            ckpt_pretrained_path,
            gpt_hidden_dim,
            hidden_dim=None,
    ):
        super(TestClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        eos = '<|EOS|>'
        special_tokens_dict = {'eos_token': eos}
        base_tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_pretrained_path)
        base_tokenizer.add_special_tokens(special_tokens_dict)
        self.gpt = GPT2LMHeadModel.from_pretrained(ckpt_pretrained_path, eos_token_id=base_tokenizer.eos_token_id)
        self.gpt.resize_token_embeddings(len(base_tokenizer))
        # Save GPT2LMHeadModel/GPT2Tokenizer with added special EOS token.
        # base_tokenizer.save_pretrained('models/new_gpt_tokenizer')
        # self.gpt.save_pretrained('models/new_gpt_model')

        self.hidden_dim = hidden_dim
        self.encode_prefix = nn.Linear(gpt_hidden_dim, hidden_dim) if hidden_dim is not None else nn.Identity()
        self.decode_prefix = nn.Linear(hidden_dim, gpt_hidden_dim) if hidden_dim is not None else nn.Identity()

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        """
        : param tokens: (Tensor) [N x max_seq_len] eg. [4 X 33]
        : param prefix: (Tensor) [N x prefix_length x 768] eg. [4 x 77 x 768]
        : param mask: (Tensor) [N x (prefix_length + max_seq_len) x 768] eg. [4 x 110 x768]

        : attribute embedding_text: (Tensor) [N x max_seq_len x 768] eg. [4 x 33 x 768]
        : attribute embedding_cat: (Tensor) [N x (prefix_length + max_seq_len) x 768] eg. [4 x 110 x 768]
        """
        embedding_text = self.gpt.transformer.wte(tokens)
        hidden = self.encode_prefix(prefix)
        prefix = self.decode_prefix(hidden)
        embedding_cat = torch.cat((prefix, embedding_text), dim=1)

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        if self.hidden_dim is not None:
            return out, hidden
        else:
            return out

    def encode_decode_prefix(self, prefix):
        return self.decode_prefix(self.encode_prefix(prefix))


class TestCaptionDecoder(nn.Module):
    def __init__(
        self,
        device,
        tokenizer_pretrained_path,
        ckpt_pretrained_path,
        gpt_hidden_dim,
        hidden_dim=-1,
    ):
        super(TestCaptionDecoder, self).__init__()
        if hidden_dim < 0:
            hidden_dim = None
        # tokenizer initialize
        eos = '<|EOS|>'
        special_tokens_dict = {'eos_token': eos}
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_pretrained_path)
        self.tokenizer.add_special_tokens(special_tokens_dict)

        # model initialize
        feature_length = 77
        # modelFile = "assets/caption_decoder/coco_v2_latest.pt"
        self.caption_model = TestClipCaptionModel(
            feature_length,
            tokenizer_pretrained_path,
            ckpt_pretrained_path,
            gpt_hidden_dim=gpt_hidden_dim,
            hidden_dim=hidden_dim,
        )
        # torch.save(self.caption_model.state_dict(), 'models/caption_decoder.pth')
        # print("Load Model...")
        # No overwrite checkpoint for now?
        # ckpt = torch.load(ckpt_pretrained_path, map_location='cpu')
        # state_dict = OrderedDict()
        # for k, v in ckpt.items():
        #     new_k = k[7:]
        #     state_dict[new_k] = v
        # mk, uk = self.caption_model.load_state_dict(state_dict, strict=False)
        # assert len(mk) == 0
        # assert all([name.startswith('clip') for name in uk])
        self.caption_model.eval()
        self.caption_model.to(device)
        self.caption_model.requires_grad_(False)
        self.device = device

    def encode_prefix(self, features):
        return self.caption_model.encode_prefix(features)

    def generate_captions(self, features):  # the low dimension representation of clip feature
        """
        generate captions given features
        : param features : (tensor([B x L x D]))
        : return generated_text: (list([L]))
        """

        # generate config
        use_beam_search = True

        features = torch.split(features, 1, dim=0)
        generated_captions = []
        with torch.no_grad():
            for feature in features:
                feature = self.caption_model.decode_prefix(feature.to(self.device))  # back to the clip feature
                if use_beam_search:
                    generated_captions.append(generate_beam(self.caption_model, self.tokenizer, embed=feature)[0])
                else:
                    generated_captions.append(generate2(self.caption_model, self.tokenizer, embed=feature))
        return generated_captions
    
    def forward(self, features):
        features = self.caption_model.decode_prefix(features.to(self.device))
        return self.caption_model.gpt(inputs_embeds=features)


class TestCaptionDecoder2(object):
    def __init__(
        self,
        device,
        tokenizer_pretrained_path,
        ckpt_pretrained_path,
        gpt_hidden_dim,
        hidden_dim=-1
    ):
        if hidden_dim < 0:
            hidden_dim = None
        # tokenizer initialize
        eos = '<|EOS|>'
        special_tokens_dict = {'eos_token': eos}
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_pretrained_path)
        self.tokenizer.add_special_tokens(special_tokens_dict)

        # model initialize
        feature_length = 77
        # modelFile = "assets/caption_decoder/coco_v2_latest.pt"
        self.caption_model = TestClipCaptionModel(
            feature_length,
            "models/clip_caption_model",
            "models/clip_caption_model",
            gpt_hidden_dim=gpt_hidden_dim,
            hidden_dim=hidden_dim,
        )
        # print("Load Model...")
        ckpt = torch.load(ckpt_pretrained_path, map_location='cpu')
        mk, uk = self.caption_model.load_state_dict(ckpt, strict=False)
        assert len(mk) == 0
        assert len(uk) == 0
        # assert all([name.startswith('clip') for name in uk])
        self.caption_model.eval()
        self.caption_model.to(device)
        self.caption_model.requires_grad_(False)
        self.device = device

    def encode_prefix(self, features):
        return self.caption_model.encode_prefix(features)

    def generate_captions(self, features):  # the low dimension representation of clip feature
        """
        generate captions given features
        : param features : (tensor([B x L x D]))
        : return generated_text: (list([L]))
        """

        # generate config
        use_beam_search = True

        features = torch.split(features, 1, dim=0)
        generated_captions = []
        with torch.no_grad():
            for feature in features:
                feature = self.caption_model.decode_prefix(feature.to(self.device))  # back to the clip feature
                if use_beam_search:
                    generated_captions.append(generate_beam(self.caption_model, self.tokenizer, embed=feature)[0])
                else:
                    generated_captions.append(generate2(self.caption_model, self.tokenizer, embed=feature))
        return generated_captions


# ----Define Testing Utility Functions----


def get_test_autoencoder(pretrained_path, scale_factor=0.18215):
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
    vae_scale_factor = 2 ** (len(ddconfig['ch_mult']) - 1)
    return TestAutoencoderKL(ddconfig, 4, pretrained_path, scale_factor), vae_scale_factor

# Modified from diffusers.utils.randn_tensor
def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
    will always be created on CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logging.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

# Sample from the autoencoder latent space directly instead of sampling the autoencoder moment.
def prepare_latents(
    config,
    clip_text_model,
    clip_img_model,
    clip_img_model_preprocess,
    autoencoder,
    vae_scale_factor,
    device,
):
    resolution = config.z_shape[-1] * vae_scale_factor
    # Fix device to CPU for reproducibility.
    latent_device = "cpu"
    latent_torch_device = torch.device(latent_device)
    generator = torch.Generator(device=latent_torch_device).manual_seed(config.seed)

    contexts = randn_tensor((config.n_samples, 77, config.clip_text_dim), generator=generator, device=latent_torch_device)
    img_contexts = randn_tensor((config.n_samples, config.z_shape[0], config.z_shape[1], config.z_shape[2]), generator=generator, device=latent_torch_device)
    clip_imgs = randn_tensor((config.n_samples, 1, config.clip_img_dim), generator=generator, device=latent_torch_device)

    if config.mode in ['t2i', 't2i2t']:
        prompts = [ config.prompt ] * config.n_samples
        contexts = clip_text_model.encode(prompts)
    elif config.mode in ['i2t', 'i2t2i']:
        from PIL import Image
        img_contexts = []
        clip_imgs = []

        def get_img_feature(image):
            image = np.array(image).astype(np.uint8)
            image = utils.center_crop(resolution, resolution, image)
            # clip_img_feature = clip_img_model.encode_image(clip_img_model_preprocess(Image.fromarray(image)).unsqueeze(0).to(device))
            # Make the proper call to huggingface transformers CLIPVisionModel
            clip_inputs = clip_img_model_preprocess(images=image, return_tensors="pt")
            clip_img_feature = clip_img_model(**clip_inputs).pooler_output

            image = (image / 127.5 - 1.0).astype(np.float32)
            image = einops.rearrange(image, 'h w c -> 1 c h w')
            image = torch.tensor(image, device=device)
            # Get moments then get the mode of the moment (diagonal Gaussian) distribution
            moments = autoencoder.encode(image)

            return clip_img_feature, moments

        image = Image.open(config.img).convert('RGB')
        clip_img, img_context = get_img_feature(image)

        img_contexts.append(img_context)
        clip_imgs.append(clip_img)
        img_contexts = img_contexts * config.n_samples
        clip_imgs = clip_imgs * config.n_samples

        img_contexts = torch.concat(img_contexts, dim=0)
        clip_imgs = torch.stack(clip_imgs, dim=0)
    
    contexts = contexts.to(device)
    img_contexts = img_contexts.to(device)
    clip_imgs = clip_imgs.to(device)
    return contexts, img_contexts, clip_imgs


# ----END----


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()

# Modified to use vae_scale_factor in place of hardcoded value of 8 when calculating resolution.
def prepare_contexts(config, clip_text_model, clip_img_model, clip_img_model_preprocess, autoencoder, vae_scale_factor):
    resolution = config.z_shape[-1] * vae_scale_factor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    contexts = torch.randn(config.n_samples, 77, config.clip_text_dim).to(device)
    img_contexts = torch.randn(config.n_samples, 2 * config.z_shape[0], config.z_shape[1], config.z_shape[2])
    clip_imgs = torch.randn(config.n_samples, 1, config.clip_img_dim)

    if config.mode in ['t2i', 't2i2t']:
        prompts = [ config.prompt ] * config.n_samples
        contexts = clip_text_model.encode(prompts)

    elif config.mode in ['i2t', 'i2t2i']:
        from PIL import Image
        img_contexts = []
        clip_imgs = []

        def get_img_feature(image):
            image = np.array(image).astype(np.uint8)
            image = utils.center_crop(resolution, resolution, image)
            # clip_img_feature = clip_img_model.encode_image(clip_img_model_preprocess(Image.fromarray(image)).unsqueeze(0).to(device))
            # Make the proper call to huggingface transformers CLIPVisionModel
            clip_inputs = clip_img_model_preprocess(images=image, return_tensors="pt")
            clip_img_feature = clip_img_model(**clip_inputs).pooler_output

            image = (image / 127.5 - 1.0).astype(np.float32)
            image = einops.rearrange(image, 'h w c -> 1 c h w')
            image = torch.tensor(image, device=device)
            moments = autoencoder.encode_moments(image)

            return clip_img_feature, moments

        image = Image.open(config.img).convert('RGB')
        clip_img, img_context = get_img_feature(image)

        img_contexts.append(img_context)
        clip_imgs.append(clip_img)
        img_contexts = img_contexts * config.n_samples
        clip_imgs = clip_imgs * config.n_samples

        img_contexts = torch.concat(img_contexts, dim=0)
        clip_imgs = torch.stack(clip_imgs, dim=0)

    return contexts, img_contexts, clip_imgs


def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.)
    v.clamp_(0., 1.)
    return v


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Fix device to cpu for now for reproducibility
    device = config.sample.device
    torch_device = torch.device(device)
    set_seed(config.seed)

    # Instantiate generator
    generator = torch.Generator(device=torch_device).manual_seed(config.seed)

    config = ml_collections.FrozenConfigDict(config)
    # utils.set_logger(log_level='info')
    # utils.set_logger(log_level='debug', fname="./logs/test.txt")
    utils.set_logger(log_level='debug')

    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)
    
    # logging.debug(f"betas: {_betas}")
    # logging.debug(f"N: {N}")
    logging.debug(f"PyTorch default dtype: {torch.get_default_dtype()}")

    nnet = utils.get_nnet(**config.nnet)
    logging.info(f'load nnet from {config.nnet_path}')
    nnet.load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.to(device)
    nnet.eval()

    use_caption_decoder = config.text_dim < config.clip_text_dim or config.mode != 't2i'
    if use_caption_decoder:
        # from libs.caption_decoder import CaptionDecoder
        # caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)

        # Import test versions
        caption_decoder = TestCaptionDecoder2(device=device, **config.caption_decoder)
        # print(caption_decoder)
    else:
        caption_decoder = None

    # clip_text_model = libs.clip.FrozenCLIPEmbedder(device=device)
    # Load small CLIP text model for training
    clip_text_model = CLIPEmbedder(
        tokenizer_pretrained_path="models/clip_text_encoder",
        ckpt_pretrained_path="models/clip_text_encoder",
        device=device,
    )
    clip_text_model.eval()
    clip_text_model.to(device)

    # autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    # Load test autoencoder
    autoencoder, vae_scale_factor = get_test_autoencoder(**config.autoencoder)
    autoencoder.to(device)
    # print(f"VAE scale factor: {vae_scale_factor}")

    # clip_img_model, clip_img_model_preprocess = clip.load("ViT-B/32", device=device, jit=False)
    # Load test CLIP image model from huggingface transformers 4.23.1
    # Note that this version has CLIPProcessor instead of CLIPImageProcessor
    clip_img_model = CLIPVisionModel.from_pretrained("models/clip_image_encoder")
    clip_img_model_preprocess = CLIPProcessor.from_pretrained("models/clip_image_encoder")

    empty_context = clip_text_model.encode([''])[0]

    def split(x):
        C, H, W = config.z_shape
        z_dim = C * H * W
        z, clip_img = x.split([z_dim, config.clip_img_dim], dim=1)
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
        clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=config.clip_img_dim)
        return z, clip_img


    def combine(z, clip_img):
        z = einops.rearrange(z, 'B C H W -> B (C H W)')
        clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
        return torch.concat([z, clip_img], dim=-1)


    def t2i_nnet(x, timesteps, text):  # text is the low dimension version of the text clip embedding
        """
        1. calculate the conditional model output
        2. calculate unconditional model output
            config.sample.t2i_cfg_mode == 'empty_token': using the original cfg with the empty string
            config.sample.t2i_cfg_mode == 'true_uncond: using the unconditional model learned by our method
        3. return linear combination of conditional output and unconditional output
        """
        z, clip_img = split(x)

        t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)

        z_out, clip_img_out, text_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=t_text,
                                             data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
        x_out = combine(z_out, clip_img_out)

        if config.sample.scale == 0.:
            return x_out

        if config.sample.t2i_cfg_mode == 'empty_token':
            _empty_context = einops.repeat(empty_context, 'L D -> B L D', B=x.size(0))
            if use_caption_decoder:
                _empty_context = caption_decoder.encode_prefix(_empty_context)
            z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z, clip_img, text=_empty_context, t_img=timesteps, t_text=t_text,
                                                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        elif config.sample.t2i_cfg_mode == 'true_uncond':
            # text_N = torch.randn_like(text)  # 3 other possible choices
            text_N = randn_tensor(text.shape, generator=generator, device=torch_device)
            z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z, clip_img, text=text_N, t_img=timesteps, t_text=torch.ones_like(timesteps) * N,
                                                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        else:
            raise NotImplementedError

        return x_out + config.sample.scale * (x_out - x_out_uncond)


    def i_nnet(x, timesteps):
        z, clip_img = split(x)
        # text = torch.randn(x.size(0), 77, config.text_dim, device=device)
        text = randn_tensor((x.size(0), 77, config.text_dim), generator=generator, device=torch_device)
        t_text = torch.ones_like(timesteps) * N
        z_out, clip_img_out, text_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=t_text,
                                             data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
        x_out = combine(z_out, clip_img_out)
        return x_out

    def t_nnet(x, timesteps):
        # z = torch.randn(x.size(0), *config.z_shape, device=device)
        # clip_img = torch.randn(x.size(0), 1, config.clip_img_dim, device=device)
        z = randn_tensor((x.size(0), *config.z_shape), generator=generator, device=torch_device)
        clip_img = randn_tensor((x.size(0), 1, config.clip_img_dim), generator=generator, device=torch_device)
        z_out, clip_img_out, text_out = nnet(z, clip_img, text=x, t_img=torch.ones_like(timesteps) * N, t_text=timesteps,
                                             data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + config.data_type)
        return text_out

    def i2t_nnet(x, timesteps, z, clip_img):
        """
        1. calculate the conditional model output
        2. calculate unconditional model output
        3. return linear combination of conditional output and unconditional output
        """
        t_img = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)

        z_out, clip_img_out, text_out = nnet(z, clip_img, text=x, t_img=t_img, t_text=timesteps,
                                             data_type=torch.zeros_like(t_img, device=device, dtype=torch.int) + config.data_type)

        if config.sample.scale == 0.:
            return text_out

        # z_N = torch.randn_like(z)  # 3 other possible choices
        # clip_img_N = torch.randn_like(clip_img)
        z_N = randn_tensor(z.shape, generator=generator, device=torch_device)
        clip_img_N = randn_tensor(clip_img.shape, generator=generator, device=torch_device)
        z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z_N, clip_img_N, text=x, t_img=torch.ones_like(timesteps) * N, t_text=timesteps,
                                                                  data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + config.data_type)

        return text_out + config.sample.scale * (text_out - text_out_uncond)

    def split_joint(x):
        C, H, W = config.z_shape
        z_dim = C * H * W
        z, clip_img, text = x.split([z_dim, config.clip_img_dim, 77 * config.text_dim], dim=1)
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
        clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=config.clip_img_dim)
        text = einops.rearrange(text, 'B (L D) -> B L D', L=77, D=config.text_dim)
        return z, clip_img, text

    def combine_joint(z, clip_img, text):
        z = einops.rearrange(z, 'B C H W -> B (C H W)')
        clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
        text = einops.rearrange(text, 'B L D -> B (L D)')
        return torch.concat([z, clip_img, text], dim=-1)

    def joint_nnet(x, timesteps):
        z, clip_img, text = split_joint(x)
        z_out, clip_img_out, text_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=timesteps,
                                             data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + config.data_type)
        logging.debug(f"Conditional VAE out: {z_out}")
        logging.debug(f"Conditional VAE out shape: {z_out.shape}")
        logging.debug(f"Conditional CLIP out: {clip_img_out}")
        logging.debug(f"Conditional CLIP out shape: {clip_img_out.shape}")
        logging.debug(f"Conditional text out: {text_out}")
        logging.debug(f"Conditional text out shape: {text_out.shape}")
        x_out = combine_joint(z_out, clip_img_out, text_out)

        if config.sample.scale == 0.:
            return x_out

        # z_noise = torch.randn(x.size(0), *config.z_shape, device=device)
        # clip_img_noise = torch.randn(x.size(0), 1, config.clip_img_dim, device=device)
        # text_noise = torch.randn(x.size(0), 77, config.text_dim, device=device)
        z_noise = randn_tensor((x.size(0), *config.z_shape), generator=generator, device=torch_device)
        clip_img_noise = randn_tensor((x.size(0), 1, config.clip_img_dim), generator=generator, device=torch_device)
        text_noise = randn_tensor((x.size(0), 77, config.text_dim), generator=generator, device=torch_device)

        _, _, text_out_uncond = nnet(z_noise, clip_img_noise, text=text, t_img=torch.ones_like(timesteps) * N, t_text=timesteps,
                                     data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + config.data_type)
        logging.debug(f"Unconditional text out: {text_out_uncond}")
        logging.debug(f"Unconditional text out shape: {text_out_uncond.shape}")
        z_out_uncond, clip_img_out_uncond, _ = nnet(z, clip_img, text=text_noise, t_img=timesteps, t_text=torch.ones_like(timesteps) * N,
                                                    data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + config.data_type)
        logging.debug(f"Unconditional VAE out: {z_out_uncond}")
        logging.debug(f"Unconditional VAE out shape: {z_out_uncond.shape}")
        logging.debug(f"Unconditional CLIP out: {clip_img_out_uncond}")
        logging.debug(f"Unconditional CLIP out shape: {clip_img_out_uncond.shape}")

        x_out_uncond = combine_joint(z_out_uncond, clip_img_out_uncond, text_out_uncond)

        return x_out + config.sample.scale * (x_out - x_out_uncond)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)


    logging.info(config.sample)
    logging.info(f'N={N}')

    # contexts, img_contexts, clip_imgs = prepare_contexts(
    #     config,
    #     clip_text_model,
    #     clip_img_model,
    #     clip_img_model_preprocess,
    #     autoencoder,
    #     vae_scale_factor,
    # )
    contexts, img_contexts, clip_imgs = prepare_latents(
        config,
        clip_text_model,
        clip_img_model,
        clip_img_model_preprocess,
        autoencoder,
        vae_scale_factor,
        device,
    )
    logging.debug(f"Text latents: {contexts}")
    logging.debug(f"Text latents shape: {contexts.shape}")

    contexts = contexts  # the clip embedding of conditioned texts
    contexts_low_dim = contexts if not use_caption_decoder else caption_decoder.encode_prefix(contexts)  # the low dimensional version of the contexts, which is the input to the nnet

    logging.debug(f"Low dim text latents: {contexts_low_dim}")
    logging.debug(f"Low dim text latents shape: {contexts_low_dim.shape}")

    img_contexts = img_contexts  # img_contexts is the autoencoder moment
    # z_img = autoencoder.sample(img_contexts, generator)
    z_img = img_contexts  # sample autoencoder latents directly, no need to call sample()
    clip_imgs = clip_imgs  # the clip embedding of conditioned image

    logging.debug(f"VAE latents: {z_img}")
    logging.debug(f"VAE latents shape: {z_img.shape}")
    logging.debug(f"CLIP latents: {clip_imgs}")
    logging.debug(f"CLIP latents shape: {clip_imgs.shape}")

    if config.mode in ['t2i', 't2i2t']:
        _n_samples = contexts_low_dim.size(0)
    elif config.mode in ['i2t', 'i2t2i']:
        _n_samples = img_contexts.size(0)
    else:
        _n_samples = config.n_samples


    def sample_fn(mode, **kwargs):

        # _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        # _clip_img_init = torch.randn(_n_samples, 1, config.clip_img_dim, device=device)
        # _text_init = torch.randn(_n_samples, 77, config.text_dim, device=device)
        _z_init = randn_tensor((_n_samples, *config.z_shape), generator=generator, device=torch_device)
        _clip_img_init = randn_tensor((_n_samples, 1, config.clip_img_dim), generator=generator, device=torch_device)
        _text_init = randn_tensor((_n_samples, 77, config.text_dim), generator=generator, device=torch_device)
        if mode == 'joint':
            _x_init = combine_joint(_z_init, _clip_img_init, _text_init)
        elif mode in ['t2i', 'i']:
            _x_init = combine(_z_init, _clip_img_init)
        elif mode in ['i2t', 't']:
            _x_init = _text_init

        logging.debug(f"Latents: {_x_init}")
        logging.debug(f"Latents shape: {_x_init.shape}")

        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

        def model_fn(x, t_continuous):
            t = t_continuous * N
            if mode == 'joint':
                return joint_nnet(x, t)
            elif mode == 't2i':
                return t2i_nnet(x, t, **kwargs)
            elif mode == 'i2t':
                return i2t_nnet(x, t, **kwargs)
            elif mode == 'i':
                return i_nnet(x, t)
            elif mode == 't':
                return t_nnet(x, t)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        with torch.no_grad():
            with torch.autocast(device_type=device):
                start_time = time.time()
                x = dpm_solver.sample(_x_init, steps=config.sample.sample_steps, eps=1. / N, T=1.)
                end_time = time.time()
                print(f'\ngenerate {_n_samples} samples with {config.sample.sample_steps} steps takes {end_time - start_time:.2f}s')
        
        logging.debug(f"Full UNet sample: {x}")
        logging.debug(f"Full UNet sample shape: {x.shape}")

        os.makedirs(config.output_path, exist_ok=True)
        if mode == 'joint':
            _z, _clip_img, _text = split_joint(x)
            return _z, _clip_img, _text
        elif mode in ['t2i', 'i']:
            _z, _clip_img = split(x)
            return _z, _clip_img
        elif mode in ['i2t', 't']:
            return x
    
    def test_sample_fn(mode, **kwargs):
        if mode == 'joint':
            _x_init = combine_joint(z_img, clip_imgs, contexts_low_dim)
        elif mode in ['t2i', 'i']:
            _x_init = combine(z_img, clip_imgs)
        elif mode in ['i2t', 't']:
            _x_init = contexts_low_dim

        logging.debug(f"Latents: {_x_init}")
        logging.debug(f"Latents shape: {_x_init.shape}")

        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

        def model_fn(x, t_continuous):
            t = t_continuous * N
            if mode == 'joint':
                noise_pred = joint_nnet(x, t)
                logging.debug(f"Noise pred for time {t}: {noise_pred}")
                logging.debug(f"Noise pred for time {t} shape: {noise_pred.shape}")
                return noise_pred
                # return joint_nnet(x, t)
            elif mode == 't2i':
                return t2i_nnet(x, t, **kwargs)
            elif mode == 'i2t':
                return i2t_nnet(x, t, **kwargs)
            elif mode == 'i':
                return i_nnet(x, t)
            elif mode == 't':
                return t_nnet(x, t)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        with torch.no_grad():
            # Remove autocast to run in full precision for testing on CPU
            start_time = time.time()
            x = dpm_solver.sample(_x_init, steps=config.sample.sample_steps, eps=1. / N, T=1.)
            end_time = time.time()
            print(f'\ngenerate {_n_samples} samples with {config.sample.sample_steps} steps takes {end_time - start_time:.2f}s')
        
        logging.debug(f"Full UNet sample: {x}")
        logging.debug(f"Full UNet sample shape: {x.shape}")

        os.makedirs(config.output_path, exist_ok=True)
        if mode == 'joint':
            _z, _clip_img, _text = split_joint(x)
            return _z, _clip_img, _text
        elif mode in ['t2i', 'i']:
            _z, _clip_img = split(x)
            return _z, _clip_img
        elif mode in ['i2t', 't']:
            return x

    def watermarking(save_path):
        img_pre = Image.open(save_path)
        img_pos = utils.add_water(img_pre)
        img_pos.save(save_path)

    if config.mode in ['joint']:
        _z, _clip_img, _text = test_sample_fn(config.mode)

        logging.debug(f"Text output: {_text}")
        logging.debug(f"Text output shape: {_text.shape}")
        logging.debug(f"VAE output: {_z}")
        logging.debug(f"VAE output shape: {_z.shape}")
        logging.debug(f"CLIP output: {_clip_img}")
        logging.debug(f"CLIP output shape: {_clip_img.shape}")

        samples = unpreprocess(decode(_z))

        logging.debug(f"VAE decoded sample: {samples}")
        logging.debug(f"VAE decoded sample shape: {samples.shape}")

        prompts = caption_decoder.generate_captions(_text)

        logging.debug(f"Generated text: {prompts}")

        os.makedirs(os.path.join(config.output_path, config.mode), exist_ok=True)
        with open(os.path.join(config.output_path, config.mode, 'prompts.txt'), 'w') as f:
            print('\n'.join(prompts), file=f)
        for idx, sample in enumerate(samples):
            # Convert to equivalent of diffusers numpy output
            # basically just put image on CPU, float32, nparray, B C H W => B H W C
            numpy_sample = sample.cpu().permute(1, 2, 0).float().numpy()
            # print(f"Sample {idx} array:")
            # print(numpy_sample)
            numpy_sample_slice = numpy_sample[-3:, -3:, -1].flatten()
            print(f"Sample {idx} slice:")
            print(numpy_sample_slice)
            np.savetxt("joint_image_slice.txt", numpy_sample_slice, fmt='%1.4f')

            save_path = os.path.join(config.output_path, config.mode, f'{idx}.png')
            save_image(sample, save_path)
            # Disable watermarking for testing purposes
            # watermarking(save_path)

    elif config.mode in ['t2i', 'i', 'i2t2i']:
        if config.mode == 't2i':
            # _z, _clip_img = sample_fn(config.mode, text=contexts_low_dim)  # conditioned on the text embedding
            _z, _clip_img = test_sample_fn(config.mode, text=contexts_low_dim)
        elif config.mode == 'i':
            # _z, _clip_img = sample_fn(config.mode)
            _z, _clip_img = test_sample_fn(config.mode)
        elif config.mode == 'i2t2i':
            _text = sample_fn('i2t', z=z_img, clip_img=clip_imgs)  # conditioned on the image embedding
            _z, _clip_img = sample_fn('t2i', text=_text)
        samples = unpreprocess(decode(_z))
        os.makedirs(os.path.join(config.output_path, config.mode), exist_ok=True)
        for idx, sample in enumerate(samples):
            # Convert to equivalent of diffusers numpy output
            # basically just put image on CPU, float32, nparray, B C H W => B H W C
            numpy_sample = sample.cpu().permute(1, 2, 0).float().numpy()
            # print(f"Sample {idx} array:")
            # print(numpy_sample)
            numpy_sample_slice = numpy_sample[-3:, -3:, -1].flatten()
            print(f"Sample {idx} slice:")
            print(numpy_sample_slice)
            np.savetxt("t2i_image_slice.txt", numpy_sample_slice, fmt='%1.4f')
            
            save_path = os.path.join(config.output_path, config.mode, f'{idx}.png')
            save_image(sample, save_path)
            # Disable watermarking for testing purposes
            # watermarking(save_path)
        # save a grid of generated images
        samples_pos = []
        for idx, sample in enumerate(samples):
            sample_pil = standard_transforms.ToPILImage()(sample)
            # Disable watermarking for testing purposes
            # sample_pil = utils.add_water(sample_pil)
            sample = standard_transforms.ToTensor()(sample_pil)
            samples_pos.append(sample)
        samples = make_grid(samples_pos, config.nrow)
        save_path = os.path.join(config.output_path, config.mode, f'grid.png')
        save_image(samples, save_path)


    elif config.mode in ['i2t', 't', 't2i2t']:
        if config.mode == 'i2t':
            # _text = sample_fn(config.mode, z=z_img, clip_img=clip_imgs)  # conditioned on the image embedding
            _text = test_sample_fn(config.mode, z=z_img, clip_img=clip_imgs)
        elif config.mode == 't':
            # _text = sample_fn(config.mode)
            _text = test_sample_fn(config.mode)
        elif config.mode == 't2i2t':
            _z, _clip_img = sample_fn('t2i', text=contexts_low_dim)
            _text = sample_fn('i2t', z=_z, clip_img=_clip_img)
        samples = caption_decoder.generate_captions(_text)
        logging.info(samples)
        os.makedirs(os.path.join(config.output_path, config.mode), exist_ok=True)
        with open(os.path.join(config.output_path, config.mode, f'{config.mode}.txt'), 'w') as f:
            print('\n'.join(samples), file=f)

    print(f'\nGPU memory usage: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB')
    print(f'\nresults are saved in {os.path.join(config.output_path, config.mode)} :)')


from absl import flags
from absl import app
from ml_collections import config_flags
import os


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", "configs/sample_test_model_v1.py", "Configuration.", lock_config=False)
flags.DEFINE_string("nnet_path", "models/test/uvit_v1.pth", "The nnet to evaluate.")
flags.DEFINE_string("output_path", "out", "dir to write results to")
flags.DEFINE_string("prompt", "an elephant under the sea", "the prompt for text-to-image generation and text variation")
flags.DEFINE_string("img", "assets/space.jpg", "the image path for image-to-text generation and image variation")
flags.DEFINE_integer("n_samples", 1, "the number of samples to generate")
flags.DEFINE_integer("nrow", 4, "number of images displayed in each row of the grid")
flags.DEFINE_string("mode", None,
                    "type of generation, one of t2i / i2t / joint / i / t / i2t2i/ t2i2t\n"
                    "t2i: text to image\n"
                    "i2t: image to text\n"
                    "joint: joint generation of text and image\n"
                    "i: only generate image\n"
                    "t: only generate text\n"
                    "i2t2i: image variation, first image to text, then text to image\n"
                    "t2i2t: text variation, first text to image, the image to text\n"
                    )


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    config.prompt = FLAGS.prompt
    config.nrow = min(FLAGS.nrow, FLAGS.n_samples)
    config.img = FLAGS.img
    config.n_samples = FLAGS.n_samples
    config.mode = FLAGS.mode
    evaluate(config)


if __name__ == "__main__":
    app.run(main)
