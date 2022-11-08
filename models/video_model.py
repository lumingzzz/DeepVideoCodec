from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .intra_model import vr_ch128n12_no4


class VideoModel(nn.Module):
    def __init__(self, lmb=2048):
        super().__init__()

        # ================================ i-frame model ================================
        i_lambda = lmb // 8
        self.i_model = vr_ch128n12_no4(lmb=i_lambda)
        # # fix i-model parameters
        for p in self.i_model.parameters():
            p.requires_grad_(False)
        # load pre-trained parameters
        wpath = f'weights/qres34m/lmb{i_lambda}/last_ema.pt'
        self.i_model.load_state_dict(torch.load(wpath)['model'])

        # ================================ p-frame model ================================
        self.p_model = MyVideoModel(lmb)

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