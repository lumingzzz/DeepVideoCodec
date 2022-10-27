# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

import torch
import torch.nn as nn

from pytorch_msssim import MS_SSIM

from compressai.entropy_models import EntropyBottleneck

from .modules import get_enc_dec_models, get_hyper_enc_dec_models, \
    ContextualEncoder, ContextualDecoder, ReconGeneration, \
        FeatureExtractor, MultiScaleContextFusion, LowerBound
from ..layers import ME_Spynet, CheckerboardMaskedConv2d, flow_warp, bilineardownsacling, \
    subpel_conv1x1, conv3x3
from ..entropy_models import laplace_log_prob_mass, DiscretizedLaplace


class Base(nn.Module):
    def __init__(self, lmbda=1024):
        super().__init__()
        # y_distribution='laplace'
        channel_mv = 64
        channel_N = 64
        channel_M = 96
        # z_channel=64
        # mv_z_channel=64

        self.channel_mv = channel_mv
        self.channel_N = channel_N
        self.channel_M = channel_M
        self.lmbda = lmbda

        self.optic_flow = ME_Spynet()

        self.mv_encoder, self.mv_decoder = get_enc_dec_models(2, 2, channel_mv)
        self.mv_hyper_encoder, self.mv_hyper_decoder = \
            get_hyper_enc_dec_models(channel_mv, channel_N)

        self.y_mv_prior_fusion = nn.Sequential(
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 2, 3, stride=1, padding=1)
        )

        self.entropy_parameters_mv = nn.Sequential(
            nn.Conv2d(channel_mv * 12 // 3, channel_mv * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel_mv * 10 // 3, channel_mv * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel_mv * 8 // 3, channel_mv * 6 // 3, 1),
        )

        self.contextual_encoder = ContextualEncoder(channel_N=channel_N, channel_M=channel_M)
        self.contextual_decoder = ContextualDecoder(channel_N=channel_N, channel_M=channel_M)

        self.contextual_hyper_encoder = nn.Sequential(
            nn.Conv2d(channel_M, channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
        )

        self.contextual_hyper_decoder = nn.Sequential(
            conv3x3(channel_N, channel_M),
            nn.LeakyReLU(),
            subpel_conv1x1(channel_M, channel_M, 2),
            nn.LeakyReLU(),
            conv3x3(channel_M, channel_M * 3 // 2),
            nn.LeakyReLU(),
            subpel_conv1x1(channel_M * 3 // 2, channel_M * 3 // 2, 2),
            nn.LeakyReLU(),
            conv3x3(channel_M * 3 // 2, channel_M * 2),
        )

        self.temporal_prior_encoder = nn.Sequential(
            nn.Conv2d(channel_N, channel_M * 3 // 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channel_M * 3 // 2, channel_M * 2, 3, stride=2, padding=1),
        )

        self.y_prior_fusion = nn.Sequential(
            nn.Conv2d(channel_M * 5, channel_M * 4, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 4, channel_M * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 3, channel_M * 2, 3, stride=1, padding=1)
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(channel_M * 12 // 3, channel_M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel_M * 10 // 3, channel_M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel_M * 8 // 3, channel_M * 6 // 3, 1),
        )

        self.feature_adaptor_I = nn.Conv2d(3, channel_N, 3, stride=1, padding=1)
        self.feature_adaptor_P = nn.Conv2d(channel_N, channel_N, 1)
        self.feature_extractor = FeatureExtractor()
        self.context_fusion_net = MultiScaleContextFusion()

        self.recon_generation_net = ReconGeneration()

        self.context_prediction_mv = CheckerboardMaskedConv2d(
            channel_mv, 2 * channel_mv, kernel_size=5, padding=2, stride=1
        )

        self.context_prediction = CheckerboardMaskedConv2d(
            channel_M, 2 * channel_M, kernel_size=5, padding=2, stride=1
        )

        self.entropy_bottleneck_mv = EntropyBottleneck(channel_N)
        self.entropy_bottleneck = EntropyBottleneck(channel_N)

        self.laplace_conditional_mv = GaussianConditional(None)
        self.laplace_conditional = GaussianConditional(None)
        
        self.mse = nn.MSELoss()
        self.ssim = MS_SSIM(data_range=1.0, size_average=False)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.xavier_normal_(m.weight, math.sqrt(2))
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)

    def multi_scale_feature_extractor(self, dpb):
        if dpb["ref_feature"] is None:
            feature = self.feature_adaptor_I(dpb["ref_frame"])
        else:
            feature = self.feature_adaptor_P(dpb["ref_feature"])
        return self.feature_extractor(feature)

    def motion_compensation(self, dpb, mv):
        warpframe = flow_warp(dpb["ref_frame"], mv)
        mv2 = bilineardownsacling(mv) / 2
        mv3 = bilineardownsacling(mv2) / 2
        ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(dpb)
        context1 = flow_warp(ref_feature1, mv)
        context2 = flow_warp(ref_feature2, mv2)
        context3 = flow_warp(ref_feature3, mv3)
        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)
        return context1, context2, context3, warpframe

    @staticmethod
    def probs_to_bits(probs):
        bits = -1.0 * torch.log(probs + 1e-5) / math.log(2.0)
        bits = LowerBound.apply(bits, 0)
        return bits

    def get_y_laplace_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return self.probs_to_bits(probs)

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def forward(self, x, dpb):
        ref_frame = dpb["ref_frame"]
        est_mv = self.optic_flow(x, ref_frame)
        y_mv = self.mv_encoder(est_mv)
        z_mv = self.mv_hyper_encoder(y_mv)
        z_mv_hat, z_mv_likelihoods = self.entropy_bottleneck_mv(z_mv)
        params_mv = self.mv_hyper_decoder(z_mv_hat)
        ref_y_mv = dpb["ref_y_mv"]
        if ref_y_mv is None:
            ref_y_mv = torch.zeros_like(y_mv)
        params_mv = torch.cat((params_mv, ref_y_mv), dim=1)
        params_mv = self.y_mv_prior_fusion(params_mv)

        y_mv_hat = self.laplace_conditional_mv.quantize(
            y_mv, "noise" if self.training else "dequantize"
        )
        # set non_anchor to 0
        y_mv_half = y_mv_hat.clone()
        y_mv_half[:, :, 0::2, 0::2] = 0
        y_mv_half[:, :, 1::2, 1::2] = 0
        # set anchor's ctx to 0, otherwise there will be a bias
        ctx_params_mv = self.context_prediction_mv(y_mv_half)
        ctx_params_mv[:, :, 0::2, 1::2] = 0
        ctx_params_mv[:, :, 1::2, 0::2] = 0

        gaussian_params_mv = self.entropy_parameters_mv(
            torch.cat((params_mv, ctx_params_mv), dim=1)
        )
        means_hat_mv, scales_hat_mv = gaussian_params_mv.chunk(2, 1)
        if self.training:
            z_sample = qm + torch.empty_like(qm).uniform_(-0.5, 0.5)
            log_prob = ec.gaussian_log_prob_mass(pm, pv, x=z_sample, bin_size=1.0, prob_clamp=1e-6)
            kl = -1.0 * log_prob
            
            laplace_log_prob_mass()
            _, mv_likelihoods = self.laplace_conditional_mv(y_mv_hat, scales_hat_mv, means=means_hat_mv)
        else:
            
        mv_hat = self.mv_decoder(y_mv_hat)
        context1, context2, context3, warp_frame = self.motion_compensation(dpb, mv_hat)
        
        y = self.contextual_encoder(x, context1, context2, context3)
        z = self.contextual_hyper_encoder(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params_hierarchical = self.contextual_hyper_decoder(z_hat)
        params_temporal = self.temporal_prior_encoder(context3)
        
        ref_y = dpb["ref_y"]
        if ref_y is None:
            ref_y = torch.zeros_like(y)
        params = torch.cat((params_temporal, params_hierarchical, ref_y), dim=1)
        params = self.y_prior_fusion(params)
        
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        # set non_anchor to 0
        y_half = y_hat.clone()
        y_half[:, :, 0::2, 0::2] = 0
        y_half[:, :, 1::2, 1::2] = 0
        # set anchor's ctx to 0, otherwise there will be a bias
        ctx_params = self.context_prediction(y_half)
        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0

        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        means_hat, scales_hat = gaussian_params.chunk(2, 1)
        recon_image_feature = self.contextual_decoder(y_hat, context2, context3)
        feature, recon_image = self.recon_generation_net(recon_image_feature, context1)

        B, _, H, W = x.size()
        pixel_num = H * W
        me_mse = self.mse(x, warp_frame)
        mse = self.mse(x, recon_image)
        # ssim = self.ssim(x, recon_image)

        bits_y = self.get_y_laplace_bits(y_hat, scales_hat)
        bits_mv_y = self.get_y_laplace_bits(y_mv_hat, scales_hat_mv)
        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_mv_y = torch.sum(bits_mv_y, dim=(1, 2, 3)) / pixel_num
        
        bpp_z = torch.sum(torch.log(z_likelihoods) / (-math.log(2)), dim=(1, 2, 3)) / pixel_num
        bpp_mv_z = torch.sum(torch.log(z_mv_likelihoods) / (-math.log(2)), dim=(1, 2, 3)) / pixel_num

        bpp = (bpp_y + bpp_z + bpp_mv_y + bpp_mv_z).mean(0)
        bit = torch.sum(bpp) * pixel_num
        bit_y = torch.sum(bpp_y) * pixel_num
        bit_z = torch.sum(bpp_z) * pixel_num
        bit_mv_y = torch.sum(bpp_mv_y) * pixel_num
        bit_mv_z = torch.sum(bpp_mv_z) * pixel_num

        loss = (bpp + self.lmbda * mse).mean(0)

        return {"bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "me_mse": me_mse,
                "mse": mse,
                # "ssim": ssim,
                "dpb": {
                    "ref_frame": recon_image,
                    "ref_feature": feature,
                    "ref_y": y_hat,
                    "ref_mv_y": y_mv_hat,
                },
                "bit": bit,
                "bit_y": bit_y,
                "bit_z": bit_z,
                "bit_mv_y": bit_mv_y,
                "bit_mv_z": bit_mv_z,
                "loss": loss,
                }
