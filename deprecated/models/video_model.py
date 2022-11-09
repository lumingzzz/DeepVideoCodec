import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .intra_model import vr_ch128n12_no4
from ..layers import ME_Spynet

class MyInterModel(nn.Module):
    log2_e = math.log2(math.e)

    def __init__(self, lmb):
        super().__init__()

        self.optic_flow = ME_Spynet()

    def forward_end2end(self, im, im_prev, get_intermediate=False):
        mv = self.optic_flow(im, im_prev)

        


        curr_features = self.feature_extractor(im)
        prev_features = self.feature_extractor(im_prev)
        flow = None
        nB, _, nH, nW = prev_features[self.max_stride].shape
        feature = self.get_bias(nhw_repeat=(nB, nH, nW))
        flow_stats = []
        frame_stats = []
        for s in sorted(self.global_strides, reverse=True): # from large stride to small stride
            # select features
            f_curr = curr_features[s]
            f_prev = prev_features[s]
            if flow is None: # the lowest resolution level
                nB, _, nH, nW = f_prev.shape
                flow = torch.zeros(nB, 2, nH, nW, device=im.device)
                f_warped = f_prev
            else: # bilinear upsampling of the flow
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear') * 2.0
                # warp t-1 feature by the flow
                f_warped = bilinear_warp(f_prev, flow)
            # TODO: use neighbourhood attention to implement flow estimation
            blk_key = f'stride{s}' # block key
            if s in self.strides_that_have_bits:
                flow, stats = self.flow_blocks[blk_key](flow, f_warped, f_curr)
                # warp t-1 feature again
                f_warped = bilinear_warp(f_prev, flow)
            else:
                stats = dict()
            if get_intermediate:
                stats['flow'] = flow
            flow_stats.append(stats)
            # p-frame prediction
            if s in self.strides_that_have_bits:
                feature, stats = self.dec_blocks[blk_key](feature, f_warped, f_curr)
            else:
                feature = self.dec_blocks[blk_key]([feature, f_warped])
                stats = dict()
            if get_intermediate:
                stats['feature'] = feature
            frame_stats.append(stats)
            feature = self.upsamples[blk_key](feature)
        x_hat = feature
        return flow_stats, frame_stats, x_hat, flow

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
        mse = F.mse_loss(x_hat, im, reduction='mean')
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
            warp_mse = F.mse_loss(bilinear_warp(im_prev, flow_hat), im)
            stats['warp-psnr'] = -10 * math.log10(warp_mse.item())

        context = {'x_hat': x_hat}
        return stats, context


class VideoModel(nn.Module):
    def __init__(self, lmb=2048):
        super().__init__()

        # ================================ i-frame model ================================
        i_lambda = lmb // 8
        self.i_model = vr_ch128n12_no4(lmb=i_lambda)
        # # fix i-model parameters
        for p in self.i_model.parameters():
            p.requires_grad_(False)

        # ================================ p-frame model ================================
        self.p_model = MyInterModel(lmb)

        self.register_buffer('_dummy', torch.zeros(1), persistent=False)
        self._dummy: torch.Tensor

    def train(self, mode=True):
        super().train(mode)
        self.i_model.eval() # the i-frame model is always in eval mode
        return self

    def forward(self, frames, log_dir=None):
        assert isinstance(frames, list)
        assert all([(im.shape == frames[0].shape) for im in frames])
        frames = [f.to(device=self._dummy.device) for f in frames]

        # initialize statistics for training and logging
        stats = OrderedDict()
        stats['loss'] = 0.0
        stats['bpp']  = None
        stats['psnr'] = None

        # ---------------- i-frame ----------------
        assert not self.i_model.training
        with torch.no_grad():
            _stats_i = self.i_model(frames[0], return_rec=True)
            prev_frame = _stats_i['im_hat']
            # logging
            stats['i-bpp']  = _stats_i['bppix']
            stats['i-psnr'] = _stats_i['psnr']

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


@register_model
def dhvc(lmb=2048):
    model = VideoModel(lmb=lmb)
    return model