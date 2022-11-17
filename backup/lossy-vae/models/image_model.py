import math
import logging
import pickle
from pathlib import Path
from PIL import Image
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as tnf
from torch.hub import load_state_dict_from_url
import torchvision.transforms.functional as tvf

from timm.models.layers.mlp import Mlp
from timm.utils import AverageMeter

from .layers import patch_downsample, patch_upsample, conv_k1s1, conv_k3s1
from .entropy_coding import DiscretizedGaussian, gaussian_log_prob_mass
from .utils import crop_divisible_by, pad_divisible_by


MAX_LMB = 8192


def sinusoidal_embedding(values: torch.Tensor, dim=256, max_period=128):
    assert values.dim() == 1 and (dim % 2) == 0
    exponents = torch.linspace(0, 1, steps=(dim // 2))
    freqs = torch.pow(max_period, -1.0 * exponents).to(device=values.device)
    args = values.view(-1, 1) * freqs.view(1, dim//2)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


def mse_loss(fake, real):
    assert fake.shape == real.shape
    return tnf.mse_loss(fake, real, reduction='none').mean(dim=(1,2,3))


class MyConvNeXtBlockAdaLN(nn.Module):
    def __init__(self, dim, embed_dim, out_dim=None, kernel_size=7, mlp_ratio=2,
                 residual=True, ls_init_value=1e-6):
        super().__init__()
        # depthwise conv
        pad = (kernel_size - 1) // 2
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim)
        # layer norm
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm.affine = False # for FLOPs computing
        # AdaLN
        self.embedding_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(embed_dim, 2*dim),
            nn.Unflatten(1, unflattened_size=(1, 1, 2*dim))
        )
        # MLP
        hidden = int(mlp_ratio * dim)
        out_dim = out_dim or dim
        self.mlp = Mlp(dim, hidden_features=hidden, out_features=out_dim, act_layer=nn.GELU)
        # layer scaling
        if ls_init_value >= 0:
            self.gamma = nn.Parameter(torch.full(size=(1, out_dim, 1, 1), fill_value=1e-6))
        else:
            self.gamma = None

        self.residual = residual
        self.requires_embedding = True

    def forward(self, x, emb):
        shortcut = x
        # depthwise conv
        x = self.conv_dw(x)
        # layer norm
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        # AdaLN
        embedding = self.embedding_layer(emb)
        shift, scale = torch.chunk(embedding, chunks=2, dim=-1)
        x = x * (1 + scale) + shift
        # MLP
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        # scaling
        if self.gamma is not None:
            x = x.mul(self.gamma)
        if self.residual:
            x = x + shortcut
        return x


class MyConvNeXtAdaLNPatchDown(MyConvNeXtBlockAdaLN):
    def __init__(self, in_ch, out_ch, down_rate=2, **kwargs):
        super().__init__(in_ch, **kwargs)
        self.downsapmle = patch_downsample(in_ch, out_ch, rate=down_rate)

    def forward(self, x, emb):
        x = super().forward(x, emb)
        out = self.downsapmle(x)
        return out


class VRLatentBlock3Pos(nn.Module):
    def __init__(self, width, zdim, embed_dim, enc_width=None, kernel_size=7, mlp_ratio=2):
        super().__init__()
        self.in_channels  = width
        self.out_channels = width

        enc_width = enc_width or width
        concat_ch = (width * 2) if (enc_width is None) else (width + enc_width)
        self.resnet_front = MyConvNeXtBlockAdaLN(width, embed_dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.resnet_end   = MyConvNeXtBlockAdaLN(width, embed_dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.posterior0 = MyConvNeXtBlockAdaLN(enc_width, embed_dim, kernel_size=kernel_size)
        self.posterior1 = MyConvNeXtBlockAdaLN(width,     embed_dim, kernel_size=kernel_size)
        self.posterior2 = MyConvNeXtBlockAdaLN(width,     embed_dim, kernel_size=kernel_size)
        self.post_merge = conv_k1s1(concat_ch, width)
        self.posterior  = conv_k3s1(width, zdim)
        self.z_proj     = conv_k1s1(zdim, width)
        self.prior      = conv_k1s1(width, zdim*2)

        self.discrete_gaussian = DiscretizedGaussian()
        self.is_latent_block = True

    def transform_prior(self, feature, lmb_embedding):
        """ prior p(z_i | z_<i)

        Args:
            feature (torch.Tensor): feature map
        """
        feature = self.resnet_front(feature, lmb_embedding)
        pm, plogv = self.prior(feature).chunk(2, dim=1)
        plogv = tnf.softplus(plogv + 2.3) - 2.3 # make logscale > -2.3
        pv = torch.exp(plogv)
        return feature, pm, pv

    def transform_posterior(self, feature, enc_feature, lmb_embedding):
        """ posterior q(z_i | z_<i, x)

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        assert feature.shape[2:4] == enc_feature.shape[2:4]
        enc_feature = self.posterior0(enc_feature, lmb_embedding)
        feature = self.posterior1(feature, lmb_embedding)
        merged = torch.cat([feature, enc_feature], dim=1)
        merged = self.post_merge(merged)
        merged = self.posterior2(merged, lmb_embedding)
        qm = self.posterior(merged)
        return qm

    def fuse_feature_and_z(self, feature, z, lmb_embedding, log_lmb):
        # add the new information carried by z to the feature
        feature = feature + self.z_proj(z)
        return feature

    def forward(self, feature, lmb_embedding, enc_feature=None, mode='trainval',
                log_lmb=None, get_latent=False, latent=None, t=1.0, strings=None):
        """ a complicated forward function

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        feature, pm, pv = self.transform_prior(feature, lmb_embedding)

        additional = dict()
        if mode == 'trainval': # training or validation
            qm = self.transform_posterior(feature, enc_feature, lmb_embedding)
            if self.training: # if training, use additive uniform noise
                z = qm + torch.empty_like(qm).uniform_(-0.5, 0.5)
                log_prob = gaussian_log_prob_mass(pm, pv, x=z, bin_size=1.0, prob_clamp=1e-6)
                kl = -1.0 * log_prob
            else: # if evaluation, use residual quantization
                z, probs = self.discrete_gaussian(qm, scales=pv, means=pm)
                kl = -1.0 * torch.log(probs)
            additional['kl'] = kl
        elif mode == 'sampling':
            if latent is None: # if z is not provided, sample it from the prior
                z = pm + pv * torch.randn_like(pm) * t + torch.empty_like(pm).uniform_(-0.5, 0.5) * t
            else: # if `z` is provided, directly use it.
                assert pm.shape == latent.shape
                z = latent
        elif mode == 'compress': # encode z into bits
            qm = self.transform_posterior(feature, enc_feature, lmb_embedding)
            indexes = self.discrete_gaussian.build_indexes(pv)
            strings = self.discrete_gaussian.compress(qm, indexes, means=pm)
            z = self.discrete_gaussian.quantize(qm, mode='dequantize', means=pm)
            additional['strings'] = strings
        elif mode == 'decompress': # decode z from bits
            assert strings is not None
            indexes = self.discrete_gaussian.build_indexes(pv)
            z = self.discrete_gaussian.decompress(strings, indexes, means=pm)
        else:
            raise ValueError(f'Unknown mode={mode}')

        feature = self.fuse_feature_and_z(feature, z, lmb_embedding, log_lmb)
        feature = self.resnet_end(feature, lmb_embedding)
        if get_latent:
            additional['z'] = z.detach()
        return feature, additional

    def update(self):
        self.discrete_gaussian.update()


class FeatureExtractor(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.enc_blocks = nn.ModuleList(blocks)

    def forward(self, x, emb=None):
        feature = x
        enc_features = OrderedDict()
        for i, block in enumerate(self.enc_blocks):
            if getattr(block, 'requires_embedding', False):
                feature = block(feature, emb)
            else:
                feature = block(feature)
            enc_features[int(feature.shape[2])] = feature
        return enc_features


class VariableRateLossyVAE(nn.Module):
    log2_e = math.log2(math.e)

    def __init__(self, config: dict):
        super().__init__()
        # feature extractor (bottom-up path)
        self.encoder = FeatureExtractor(config.pop('enc_blocks'))
        # latent variable blocks (top-down path)
        self.dec_blocks = nn.ModuleList(config.pop('dec_blocks'))
        width = self.dec_blocks[0].in_channels
        self.bias = nn.Parameter(torch.zeros(1, width, 1, 1))
        self.num_latents = len([b for b in self.dec_blocks if getattr(b, 'is_latent_block', False)])
        # loss function, for computing reconstruction loss
        self.distortion_name = 'mse'
        self.distortion_func = mse_loss

        self._set_lmb_embedding(config)

        self.max_stride = config['max_stride']

        self.register_buffer('_dummy', torch.zeros(1), persistent=False)
        self._dummy: torch.Tensor

        self.compressing = False
        # self._stats_log = dict()
        self._logging_images = config.get('log_images', None)
        self._logging_smpl_k = [1, 2]
        self._flops_mode = False

    def _set_lmb_embedding(self, config):
        assert len(config['log_lmb_range']) == 2
        self.log_lmb_range = (float(config['log_lmb_range'][0]), float(config['log_lmb_range'][1]))
        self.lmb_embed_dim = config['lmb_embed_dim']
        self.lmb_embedding = nn.Sequential(
            nn.Linear(self.lmb_embed_dim[0], self.lmb_embed_dim[1]),
            nn.GELU(),
            nn.Linear(self.lmb_embed_dim[1], self.lmb_embed_dim[1]),
        )
        self._default_log_lmb = (self.log_lmb_range[0] + self.log_lmb_range[1]) / 2
        # experiment
        self._sin_period = config['sin_period']
        self.LOG_LMB_SCALE = self._sin_period / math.log(MAX_LMB)

    def sample_log_lmb(self, n):
        low, high = self.log_lmb_range
        low, high = math.exp(low), math.exp(high) # lmb space
        p = 3.0
        low, high = math.pow(low, 1/p), math.pow(high, 1/p) # transformed space
        transformed_lmb = low + (high-low) * torch.rand(n, device=self._dummy.device)
        log_lmb = torch.log(transformed_lmb.pow(p))
        return log_lmb

    def expand_to_tensor(self, log_lmb, n):
        assert isinstance(log_lmb, (torch.Tensor, float, int)), f'type(log_lmb)={type(log_lmb)}'
        if isinstance(log_lmb, torch.Tensor) and (log_lmb.numel() == 1):
            log_lmb = log_lmb.item()
        if isinstance(log_lmb, (float, int)):
            log_lmb = torch.full(size=(n,), fill_value=float(log_lmb), device=self._dummy.device)
        assert log_lmb.shape == (n,), f'log_lmb={log_lmb}'
        return log_lmb

    def _get_lmb_embedding(self, log_lmb, n):
        log_lmb = self.expand_to_tensor(log_lmb, n=n)
        scaled = log_lmb * self.LOG_LMB_SCALE
        embedding = sinusoidal_embedding(scaled, dim=self.lmb_embed_dim[0], max_period=self._sin_period)
        embedding = self.lmb_embedding(embedding)
        return embedding

    def get_bias(self, bhw_repeat=(1,1,1)):
        nB, nH, nW = bhw_repeat
        feature = self.bias.expand(nB, -1, nH, nW)
        # feature = torch.zeros(nB, self.initial_width, nH, nW, device=self._dummy.device)
        return feature

    def forward_end2end(self, im: torch.Tensor, log_lmb: torch.Tensor, get_latents=False):
        # ================ get lambda embedding ================
        emb = self._get_lmb_embedding(log_lmb, n=im.shape[0])
        # ================ Forward pass ================
        enc_features = self.encoder(im, emb)
        all_block_stats = []
        nB, _, nH, nW = enc_features[min(enc_features.keys())].shape
        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        for i, block in enumerate(self.dec_blocks):
            if getattr(block, 'is_latent_block', False):
                key = int(feature.shape[2])
                f_enc = enc_features[key]
                feature, stats = block(feature, emb, enc_feature=f_enc, mode='trainval',
                                       log_lmb=log_lmb, get_latent=get_latents)
                all_block_stats.append(stats)
            elif getattr(block, 'requires_embedding', False):
                feature = block(feature, emb)
            else:
                feature = block(feature)
        return feature, all_block_stats

    def forward(self, batch, log_lmb=None, return_rec=False):
        if isinstance(batch, (tuple, list)):
            im, label = batch
        else:
            im = batch
        im = im.to(self._dummy.device)
        nB, imC, imH, imW = im.shape # batch, channel, height, width

        # ================ Forward pass ================
        if (log_lmb is None): # training
            log_lmb = self.sample_log_lmb(n=im.shape[0])
        assert isinstance(log_lmb, torch.Tensor) and log_lmb.shape == (nB,)
        x_hat, stats_all = self.forward_end2end(im, log_lmb)

        # ================ Compute Loss ================
        # rate
        kl_divergences = [stat['kl'].sum(dim=(1, 2, 3)) for stat in stats_all]
        ndims = float(imC * imH * imW)
        kl = sum(kl_divergences) / ndims # nats per dimension
        # distortion
        distortion = self.distortion_func(x_hat, im)
        # rate + distortion
        loss = kl + torch.exp(log_lmb) * distortion
        loss = loss.mean(0)

        stats = OrderedDict()
        stats['loss'] = loss

        # ================ Logging ================
        with torch.no_grad():
            # for training print
            stats['bppix'] = kl.mean(0).item() * self.log2_e * imC
            stats[self.distortion_name] = distortion.mean(0).item()
            im_hat = x_hat.detach()
            im_mse = tnf.mse_loss(im_hat, im, reduction='mean')
            psnr = -10 * math.log10(im_mse.item())
            stats['psnr'] = psnr

        if return_rec:
            stats['im_hat'] = im_hat
        return stats

    def compress_mode(self, mode=True):
        if mode:
            for block in self.dec_blocks:
                if hasattr(block, 'update'):
                    block.update()
        self.compressing = mode

    @torch.no_grad()
    def compress(self, im, log_lmb=None):
        if log_lmb is None: # use default log-lambda
            log_lmb = self._default_log_lmb
        log_lmb = self.expand_to_tensor(log_lmb, n=im.shape[0])
        lmb_embedding = self._get_lmb_embedding(log_lmb, n=im.shape[0])
        enc_features = self.encoder(im, lmb_embedding)
        nB, _, nH, nW = enc_features[min(enc_features.keys())].shape
        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        strings_all = []
        for i, block in enumerate(self.dec_blocks):
            if getattr(block, 'is_latent_block', False):
                f_enc = enc_features[feature.shape[2]]
                feature, stats = block(feature, lmb_embedding, enc_feature=f_enc, mode='compress',
                                       log_lmb=log_lmb)
                strings_all.append(stats['strings'])
            elif getattr(block, 'requires_embedding', False):
                feature = block(feature, lmb_embedding)
            else:
                feature = block(feature)
        strings_all.append((nB, nH, nW)) # smallest feature shape
        strings_all.append(log_lmb) # log lambda
        return strings_all

    @torch.no_grad()
    def decompress(self, compressed_object):
        log_lmb = compressed_object[-1] # log lambda
        nB, nH, nW = compressed_object[-2] # smallest feature shape
        log_lmb = self.expand_to_tensor(log_lmb, n=nB)
        lmb_embedding = self._get_lmb_embedding(log_lmb, n=nB)

        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        str_i = 0
        for bi, block in enumerate(self.dec_blocks):
            if getattr(block, 'is_latent_block', False):
                strs_batch = compressed_object[str_i]
                feature, _ = block(feature, lmb_embedding, mode='decompress',
                                   log_lmb=log_lmb, strings=strs_batch)
                str_i += 1
            elif getattr(block, 'requires_embedding', False):
                feature = block(feature, lmb_embedding)
            else:
                feature = block(feature)
        assert str_i == len(compressed_object) - 2, f'str_i={str_i}, len={len(compressed_object)}'
        im_hat = feature
        return im_hat

    @torch.no_grad()
    def compress_file(self, img_path, output_path):
        # read image
        img = Image.open(img_path)
        img_padded = pad_divisible_by(img, div=self.max_stride)
        device = next(self.parameters()).device
        im = tvf.to_tensor(img_padded).unsqueeze_(0).to(device=device)
        # compress by model
        compressed_obj = self.compress(im)
        compressed_obj.append((img.height, img.width))
        # save bits to file
        with open(output_path, 'wb') as f:
            pickle.dump(compressed_obj, file=f)

    @torch.no_grad()
    def decompress_file(self, bits_path):
        # read from file
        with open(bits_path, 'rb') as f:
            compressed_obj = pickle.load(file=f)
        img_h, img_w = compressed_obj.pop()
        # decompress by model
        im_hat = self.decompress(compressed_obj)
        return im_hat[:, :, :img_h, :img_w]

    @torch.no_grad()
    def self_evaluate(self, im, log_lmb: float):
        image_stats = defaultdict(float)
        x_hat, stats_all = self.forward_end2end(im, log_lmb=self.expand_to_tensor(log_lmb,n=1))
        # compute bpp
        _, imC, imH, imW = im.shape
        kl = sum([stat['kl'].sum(dim=(1, 2, 3)) for stat in stats_all]).mean(0) / (imC*imH*imW)
        bpp_estimated = kl.item() * self.log2_e * imC
        # compute psnr
        x_target = im
        distortion = self.distortion_func(x_hat, x_target).item()
        mse = tnf.mse_loss(im, x_hat, reduction='mean').item()
        psnr = float(-10 * math.log10(mse))
        image_stats['loss'] = float(kl.item() + math.exp(log_lmb) * distortion)
        image_stats['bpp']  = bpp_estimated
        image_stats['psnr'] = psnr

        return image_stats


def qres_vr(lmb_range=[128,1024]):
    cfg = dict()

    # variable rate
    cfg['log_lmb_range'] = (math.log(lmb_range[0]), math.log(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64
    _emb_dim = cfg['lmb_embed_dim'][1]

    ch = 128
    dec_nums = [1, 2, 3, 3]
    enc_dims = [192, ch*3, ch*4, ch*4, ch*4]
    dec_dims = [ch*4, ch*4, ch*3, ch*2, ch*1]
    z_dims = [32, 32, 64, 8]

    im_channels = 3
    cfg['enc_blocks'] = [
        patch_downsample(im_channels, enc_dims[0], rate=4),
        *[MyConvNeXtBlockAdaLN(enc_dims[0], _emb_dim, kernel_size=7) for _ in range(6)], # 16x16
        MyConvNeXtAdaLNPatchDown(enc_dims[0], enc_dims[1], embed_dim=_emb_dim),
        *[MyConvNeXtBlockAdaLN(enc_dims[1], _emb_dim, kernel_size=7) for _ in range(6)], # 8x8
        MyConvNeXtAdaLNPatchDown(enc_dims[1], enc_dims[2], embed_dim=_emb_dim),
        *[MyConvNeXtBlockAdaLN(enc_dims[2], _emb_dim, kernel_size=5) for _ in range(6)], # 4x4
        MyConvNeXtAdaLNPatchDown(enc_dims[2], enc_dims[3], embed_dim=_emb_dim),
        *[MyConvNeXtBlockAdaLN(enc_dims[3], _emb_dim, kernel_size=3) for _ in range(4)], # 2x2
        MyConvNeXtAdaLNPatchDown(enc_dims[3], enc_dims[3], embed_dim=_emb_dim),
        *[MyConvNeXtBlockAdaLN(enc_dims[3], _emb_dim, kernel_size=1) for _ in range(4)], # 1x1
    ]
    cfg['dec_blocks'] = [
        # 1x1
        *[VRLatentBlock3Pos(dec_dims[0], z_dims[0], _emb_dim, enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=4) for _ in range(dec_nums[0])],
        MyConvNeXtBlockAdaLN(dec_dims[0], _emb_dim, kernel_size=1, mlp_ratio=4),
        patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        MyConvNeXtBlockAdaLN(dec_dims[1], _emb_dim, kernel_size=3, mlp_ratio=3),
        *[VRLatentBlock3Pos(dec_dims[1], z_dims[1], _emb_dim, enc_width=enc_dims[-2], kernel_size=3, mlp_ratio=3) for _ in range(dec_nums[1])],
        MyConvNeXtBlockAdaLN(dec_dims[1], _emb_dim, kernel_size=3, mlp_ratio=3),
        patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        MyConvNeXtBlockAdaLN(dec_dims[2], _emb_dim, kernel_size=5, mlp_ratio=2),
        *[VRLatentBlock3Pos(dec_dims[2], z_dims[2], _emb_dim, enc_width=enc_dims[-3], kernel_size=5, mlp_ratio=2) for _ in range(dec_nums[2])],
        MyConvNeXtBlockAdaLN(dec_dims[2], _emb_dim, kernel_size=5, mlp_ratio=2),
        patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        MyConvNeXtBlockAdaLN(dec_dims[3], _emb_dim, kernel_size=7, mlp_ratio=1.75),
        *[VRLatentBlock3Pos(dec_dims[3], z_dims[3], _emb_dim, enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(dec_nums[3])],
        MyConvNeXtBlockAdaLN(dec_dims[3], _emb_dim, kernel_size=7, mlp_ratio=1.75),
        patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[MyConvNeXtBlockAdaLN(dec_dims[4], _emb_dim, kernel_size=7, mlp_ratio=1.5) for _ in range(8)],
        patch_upsample(dec_dims[4], im_channels, rate=4)
    ]

    cfg['max_stride'] = 64

    # cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    model = VariableRateLossyVAE(cfg)
    # if pretrained:
    #     url = 'https://huggingface.co/duanzh0/my-model-weights/resolve/main/vr_version2.pt'
    #     msd = load_state_dict_from_url(url)['model']
    #     model.load_state_dict(msd)
    return model