import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as tnf

from lvae.models.registry import register_model
from lvae.models.vr.model import FeatureExtractor, mse_loss, sinusoidal_embedding, \
    MyConvNeXtBlockAdaLN, MyConvNeXtAdaLNPatchDown, VRLatentBlock3Pos
import lvae.models.common as common


MAX_LMB = 8192


class MotionLossyVAE(nn.Module):
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

        self.im_shift = float(config['im_shift'])
        self.im_scale = float(config['im_scale'])
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
            im_hat = x_hat.detach().clamp_(0, 1.0)
            im_mse = tnf.mse_loss(im_hat, im, reduction='mean')
            psnr = -10 * math.log10(im_mse.item())
            stats['psnr'] = psnr

        if return_rec:
            stats['im_hat'] = im_hat
        return stats


@register_model
def mv_codec(lmb_range=[16,1024]):
    cfg = dict()

    # variable rate
    cfg['log_lmb_range'] = (math.log(lmb_range[0]), math.log(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64
    _emb_dim = cfg['lmb_embed_dim'][1]

    dec_nums = [1, 2, 3, 3]
    enc_dims = [64, 64, 64, 64, 64]
    dec_dims = [64, 64, 64, 64, 64]
    z_dims = [16, 16, 32, 8]

    mv_channels = 2
    cfg['enc_blocks'] = [
        common.patch_downsample(mv_channels, enc_dims[0], rate=4),
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
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        MyConvNeXtBlockAdaLN(dec_dims[1], _emb_dim, kernel_size=3, mlp_ratio=3),
        *[VRLatentBlock3Pos(dec_dims[1], z_dims[1], _emb_dim, enc_width=enc_dims[-2], kernel_size=3, mlp_ratio=3) for _ in range(dec_nums[1])],
        MyConvNeXtBlockAdaLN(dec_dims[1], _emb_dim, kernel_size=3, mlp_ratio=3),
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        MyConvNeXtBlockAdaLN(dec_dims[2], _emb_dim, kernel_size=5, mlp_ratio=2),
        *[VRLatentBlock3Pos(dec_dims[2], z_dims[2], _emb_dim, enc_width=enc_dims[-3], kernel_size=5, mlp_ratio=2) for _ in range(dec_nums[2])],
        MyConvNeXtBlockAdaLN(dec_dims[2], _emb_dim, kernel_size=5, mlp_ratio=2),
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        MyConvNeXtBlockAdaLN(dec_dims[3], _emb_dim, kernel_size=7, mlp_ratio=1.75),
        *[VRLatentBlock3Pos(dec_dims[3], z_dims[3], _emb_dim, enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(dec_nums[3])],
        MyConvNeXtBlockAdaLN(dec_dims[3], _emb_dim, kernel_size=7, mlp_ratio=1.75),
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[MyConvNeXtBlockAdaLN(dec_dims[4], _emb_dim, kernel_size=7, mlp_ratio=1.5) for _ in range(8)],
        common.patch_upsample(dec_dims[4], mv_channels, rate=4)
    ]

    cfg['max_stride'] = 64

    model = MotionLossyVAE(cfg)

    return model