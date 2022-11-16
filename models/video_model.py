import math
import logging
from pathlib import Path
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as tnf

from timm.models.layers.mlp import Mlp
from timm.utils import AverageMeter

from .layers import patch_downsample, patch_upsample, conv_k1s1, conv_k3s1, ME_Spynet
from .entropy_coding import DiscretizedGaussian, gaussian_log_prob_mass
from .utils import crop_divisible_by, pad_divisible_by

from .image_model import qres_vr


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
        plogv = F.softplus(plogv + 2.3) - 2.3 # make logscale > -2.3
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


class MyInterModel(nn.Module):
    log2_e = math.log2(math.e)

    def __init__(self, config: dict):
        super().__init__()

        self.optic_flow = ME_Spynet()

        # feature extractor (bottom-up path)
        self.mv_encoder = FeatureExtractor(config.pop('mv_enc_blocks'))
        # latent variable blocks (top-down path)
        self.mv_dec_blocks = nn.ModuleList(config.pop('mv_dec_blocks'))
        width = self.mv_dec_blocks[0].in_channels
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
        self._logging_images = config.get('log_images', None)
        self._logging_smpl_k = [1, 2]
        self._flops_mode = False

        # im_channels = 3
        # # ================================ feature extractor ================================
        # ch = 96
        # enc_dims = (32, 64, ch*1, ch*2, ch*4, ch*4, ch*4)
        # enc_nums     = (1, 2, 2, 2, 2, 2, 2)
        # kernel_sizes = (7, 7, 7, 7, 7, 5, 3)
        # enc_blocks = [common.conv_k3s1(im_channels, 32),]
        # for i, (dim, ks, num) in enumerate(zip(enc_dims, kernel_sizes, enc_nums,)):
        #     for _ in range(num):
        #         enc_blocks.append(common.MyConvNeXtBlock(dim, kernel_size=ks))
        #     if i < len(enc_dims) - 1:
        #         new_dim = enc_dims[i+1]
        #         # enc_blocks.append(common.MyConvNeXtPatchDown(dim, new_dim, kernel_size=ks))
        #         enc_blocks.append(myconvnext_down(dim, new_dim, kernel_size=ks))
        # self.feature_extractor = common.BottomUpEncoder(enc_blocks, dict_key='stride')

        # self.strides_that_have_bits = set([4, 8, 16, 32, 64])
        # # ================================ flow models ================================
        # global_strides = (1, 2, 4, 8, 16, 32, 64)
        # flow_dims = (None, None, 48, 72, 96, 128, 128)
        # flow_zdims = (None, None, 2, 4, 4, 4, 4)
        # kernel_sizes = (7, 7, 7, 7, 7, 5, 3)
        # self.flow_blocks = nn.ModuleDict()
        # for s, indim, dim, zdim, ks in zip(global_strides, enc_dims, flow_dims, flow_zdims, kernel_sizes):
        #     if s not in self.strides_that_have_bits:
        #         continue
        #     corr_dim, strided = (96, True) if (s == 4) else (128, False)
        #     module = qrvm.CorrelationFlowCodingBlock(indim, dim=dim, zdim=zdim, ks=ks,
        #                     corr_dim=corr_dim, strided_corr=strided)
        #     self.flow_blocks[f'stride{s}'] = module
        # # ================================ p-frame models ================================
        # dec_dims = enc_dims
        # self.bias = nn.Parameter(torch.zeros(1, dec_dims[-1], 1, 1))
        # self.dec_blocks = nn.ModuleDict()
        # for s, dim, ks in zip(global_strides, dec_dims, kernel_sizes):
        #     if s in self.strides_that_have_bits:
        #         module = qrvm.SpyCodingFrameBlock(dim, zdim=8, kernel_size=ks)
        #     else:
        #         module = qrvm.ResConditional(dim, kernel_size=ks)
        #     self.dec_blocks[f'stride{s}'] = module
        # # ================================ upsample layers ================================
        # self.upsamples = nn.ModuleDict()
        # for i, (s, dim) in enumerate(zip(global_strides, dec_dims)):
        #     if s == 1: # same as image resolution; no need to upsample
        #         conv = common.conv_k3s1(dim, im_channels)
        #     else:
        #         conv = common.patch_upsample(dim, dec_dims[i-1], rate=2)
        #     self.upsamples[f'stride{s}'] = conv

        # self.global_strides = global_strides
        # self.max_stride = max(global_strides)

        # self.distortion_lmb = float(distortion_lmb)

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
        return feature

    def forward_end2end(self, x_cur: torch.Tensor, dpb, log_lmb: torch.Tensor, get_latents=False):
        # ================ get lambda embedding ================
        emb = self._get_lmb_embedding(log_lmb, n=x_cur.shape[0])
        
        x_ref = dpb["x_ref"]
        mv = self.optic_flow(x_cur, x_ref)
        enc_mv_features = self.mv_encoder(mv, emb)
        mv_block_stats = []
        nB, _, nH, nW = enc_mv_features[min(enc_mv_features.keys())].shape
        mv_feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        for i, block in enumerate(self.mv_dec_blocks):
            if getattr(block, 'is_latent_block', False):
                key = int(mv_feature.shape[2])
                f_enc = enc_mv_features[key]
                mv_feature, stats = block(mv_feature, emb, enc_feature=f_enc, mode='trainval',
                                       log_lmb=log_lmb, get_latent=get_latents)
                mv_block_stats.append(stats)
            elif getattr(block, 'requires_embedding', False):
                mv_feature = block(mv_feature, emb)
            else:
                mv_feature = block(mv_feature)

        print(mv_feature.shape)

        return mv_feature, mv_block_stats



    def forward(self, im, im_prev):
        # ================ Forward pass ================
        flow_stats, frame_stats, x_hat, flow_hat = self.forward_end2end(im, im_prev)

        # ================ Compute loss ================
        num_pix = float(im.shape[2] * im.shape[3])
        # Rate
        flow_kls  = [stat['kl'] for stat in flow_stats  if ('kl' in stat)]
        frame_kls = [stat['kl'] for stat in frame_stats if ('kl' in stat)]
        # from total nats to bits-per-pixel
        flow_bpp  = self.log2_e * sum([kl.sum(dim=(1,2,3)) for kl in flow_kls]).mean(0) / num_pix
        frame_bpp = self.log2_e * sum([kl.sum(dim=(1,2,3)) for kl in frame_kls]).mean(0) / num_pix
        # Distortion
        mse = tnf.mse_loss(x_hat, im, reduction='mean')
        # Rate + lmb * Distortion
        loss = (flow_bpp + frame_bpp) + self.distortion_lmb * mse

        stats = OrderedDict()
        stats['loss'] = loss
        # ================ Logging ================
        with torch.no_grad():
            stats['bpp'] = (flow_bpp + frame_bpp).item()
            stats['psnr'] = -10 * math.log10(mse.item())
            stats['mv-bpp'] = flow_bpp.item()
            stats['fr-bpp'] = frame_bpp.item()
            warp_mse = tnf.mse_loss(bilinear_warp(im_prev, flow_hat), im)
            stats['warp-psnr'] = -10 * math.log10(warp_mse.item())

        context = {'x_hat': x_hat}
        return stats, context


class LossyVideoCodec(nn.Module):
    def __init__(self):
        super().__init__()

        # ================================ i-frame model ================================
        self.i_model = qres_vr()
        # # fix i-model parameters
        for p in self.i_model.parameters():
            p.requires_grad_(False)
        # load pre-trained parameters
        wpath = f'checkpoints/qres-vr/checkpoint_best_loss.pth.tar'
        self.i_model.load_state_dict(torch.load(wpath)['state_dict'])

        # ================================ p-frame model ================================
        self.p_model = MyInterModel()

        # self.testing_gop = 32 # group of pictures at test-time
        self.register_buffer('_dummy', torch.zeros(1), persistent=False)
        self._dummy: torch.Tensor

    def train(self, mode=True):
        super().train(mode)
        self.i_model.eval() # the i-frame model is always in eval mode
        return self

    def forward(self, frames):
        assert isinstance(frames, list)
        assert all([(im.shape == frames[0].shape) for im in frames])
        frames = [f.to(device=self._dummy.device) for f in frames]

        # initialize statistics for training and logging
        stats = OrderedDict()
        stats['loss'] = 0.0
        stats['bpp']  = None
        stats['psnr'] = None

        # ---------------- i-frame ----------------
        # assert not self.i_model.training
        # with torch.no_grad():
        #     _stats_i = self.i_model(frames[0], return_rec=True)
        #     prev_frame = _stats_i['im_hat']
        #     # logging
        #     stats['i-bpp']  = _stats_i['bppix']
        #     stats['i-psnr'] = _stats_i['psnr']
        stats['i-bpp'] = 0
        stats['i-psnr'] = 0

        # ---------------- p-frames ----------------
        p_stats_keys = ['loss', 'mv-bpp', 'warp-psnr', 'p-bpp', 'p-psnr']
        for key in p_stats_keys:
            stats[key] = 0.0
        p_frames = frames[1:]
        for i, frame in enumerate(p_frames):
            # conditional coding of current frame
            _stats_p, context_p = self.p_model(frame, prev_frame)

            # logging
            stats['loss'] = stats['loss'] + _stats_p['loss']
            stats['mv-bpp']    += float(_stats_p['mv-bpp'])
            stats['warp-psnr'] += float(_stats_p['warp-psnr'])
            stats['p-bpp']     += float(_stats_p['fr-bpp'])
            stats['p-psnr']    += float(_stats_p['psnr'])
            # if (log_dir is not None): # save results
            #     log_dir = Path(log_dir)
            #     save_images(log_dir, f'prev_xhat_cur-{i}.png',
            #         [prev_frame, context_p['x_hat'], frame]
            #     )
            prev_frame = context_p['x_hat']

        # all frames statictics
        stats['bpp'] = (stats['i-bpp'] + stats['mv-bpp'] + stats['p-bpp']) / len(frames)
        stats['psnr'] = (stats['i-psnr'] + stats['p-psnr']) / len(frames)
        # average over p-frames only
        for key in p_stats_keys:
            stats[key] = stats[key] / len(p_frames)

        return stats

    # @torch.no_grad()
    # def forward_eval(self, frames):
    #     return self.forward(frames)

    # @torch.no_grad()
    # def self_evaluate(self, dataset, max_frames, log_dir=None):
    #     results = video_fast_evaluate(self, dataset, max_frames)
    #     return results


def dhvc_vr(lmb_range=[64, 512]):
    model = LossyVideoCodec()
    return model