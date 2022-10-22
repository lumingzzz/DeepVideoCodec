# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# import time

import torch
import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck

from .modules import ME_Spynet, get_enc_dec_models, get_hyper_enc_dec_models

# from .video_net import ME_Spynet, flow_warp, ResBlock, bilineardownsacling, LowerBound, UNet, \
#     get_enc_dec_models, get_hyper_enc_dec_models
# from ..layers.layers import conv3x3, subpel_conv1x1, subpel_conv3x3
# from ..utils.stream_helper import get_downsampled_shape, encode_p, decode_p, filesize, \
#     get_rounded_q, get_state_dict


class Bench(nn.Module):
    def __init__(self):
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

        self.optic_flow = ME_Spynet()

        self.mv_encoder, self.mv_decoder = get_enc_dec_models(2, 2, channel_mv)
        self.mv_hyper_prior_encoder, self.mv_hyper_prior_decoder = \
            get_hyper_enc_dec_models(channel_mv, channel_N)

        self.mv_y_prior_fusion = nn.Sequential(
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, stride=1, padding=1)
        )

        self.mv_y_spatial_prior = nn.Sequential(
            nn.Conv2d(channel_mv * 4, channel_mv * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 2, 3, padding=1)
        )

        self.feature_adaptor_I = nn.Conv2d(3, channel_N, 3, stride=1, padding=1)
        self.feature_adaptor_P = nn.Conv2d(channel_N, channel_N, 1)
        # self.feature_extractor = FeatureExtractor()
        # self.context_fusion_net = MultiScaleContextFusion()

        # self.contextual_encoder = ContextualEncoder(channel_N=channel_N, channel_M=channel_M)

        self.contextual_hyper_prior_encoder = nn.Sequential(
            nn.Conv2d(channel_M, channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
        )

        # self.contextual_hyper_prior_decoder = nn.Sequential(
        #     conv3x3(channel_N, channel_M),
        #     nn.LeakyReLU(),
        #     subpel_conv1x1(channel_M, channel_M, 2),
        #     nn.LeakyReLU(),
        #     conv3x3(channel_M, channel_M * 3 // 2),
        #     nn.LeakyReLU(),
        #     subpel_conv1x1(channel_M * 3 // 2, channel_M * 3 // 2, 2),
        #     nn.LeakyReLU(),
        #     conv3x3(channel_M * 3 // 2, channel_M * 2),
        # )

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
            nn.Conv2d(channel_M * 3, channel_M * 3, 3, stride=1, padding=1)
        )

        self.y_spatial_prior = nn.Sequential(
            nn.Conv2d(channel_M * 4, channel_M * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 3, channel_M * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 3, channel_M * 2, 3, padding=1)
        )

        self.entropy_bottleneck = EntropyBottleneck(channel_mv)
        self.entropy_bottleneck = EntropyBottleneck(channel_M)

        # self.contextual_decoder = ContextualDecoder(channel_N=channel_N, channel_M=channel_M)
        # self.recon_generation_net = ReconGeneration()

        # self.mv_y_q_basic = nn.Parameter(torch.ones((1, channel_mv, 1, 1)))
        # self.mv_y_q_scale = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        # self.y_q_basic = nn.Parameter(torch.ones((1, channel_M, 1, 1)))
        # self.y_q_scale = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        # self.anchor_num = int(anchor_num)

        # self._initialize_weights()

    def multi_scale_feature_extractor(self, dpb):
        if dpb["ref_feature"] is None:
            feature = self.feature_adaptor_I(dpb["ref_frame"])
        else:
            feature = self.feature_adaptor_P(dpb["ref_feature"])
        return self.feature_extractor(feature)

    # def motion_compensation(self, dpb, mv):
    #     warpframe = flow_warp(dpb["ref_frame"], mv)
    #     mv2 = bilineardownsacling(mv) / 2
    #     mv3 = bilineardownsacling(mv2) / 2
    #     ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(dpb)
    #     context1 = flow_warp(ref_feature1, mv)
    #     context2 = flow_warp(ref_feature2, mv2)
    #     context3 = flow_warp(ref_feature3, mv3)
    #     context1, context2, context3 = self.context_fusion_net(context1, context2, context3)
    #     return context1, context2, context3, warpframe

    def forward(self, x, dpb):
        ref_frame = dpb["ref_frame"]
        print(x.shape, ref_frame.shape)
        est_mv = self.optic_flow(x, ref_frame)
        mv_y = self.mv_encoder(est_mv)
        mv_z = self.mv_hyper_prior_encoder(mv_y)


        
        mv_z_hat = self.quant(mv_z)
        mv_params = self.mv_hyper_prior_decoder(mv_z_hat)
        ref_mv_y = dpb["ref_mv_y"]
        if ref_mv_y is None:
            ref_mv_y = torch.zeros_like(mv_y)
        mv_params = torch.cat((mv_params, ref_mv_y), dim=1)
        mv_q_step, mv_scales, mv_means = self.mv_y_prior_fusion(mv_params).chunk(3, 1)
        mv_y_res, mv_y_q, mv_y_hat, mv_scales_hat = self.forward_dual_prior(
            mv_y, mv_means, mv_scales, mv_q_step, self.mv_y_spatial_prior)

        mv_hat = self.mv_decoder(mv_y_hat)
        context1, context2, context3, warp_frame = self.motion_compensation(dpb, mv_hat)

        y = self.contextual_encoder(x, context1, context2, context3)
        z = self.contextual_hyper_prior_encoder(y)
        z_hat = self.quant(z)
        hierarchical_params = self.contextual_hyper_prior_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(context3)

        ref_y = dpb["ref_y"]
        if ref_y is None:
            ref_y = torch.zeros_like(y)
        params = torch.cat((temporal_params, hierarchical_params, ref_y), dim=1)
        q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
        y_res, y_q, y_hat, scales_hat = self.forward_dual_prior(
            y, means, scales, q_step, self.y_spatial_prior)

        recon_image_feature = self.contextual_decoder(y_hat, context2, context3)
        feature, recon_image = self.recon_generation_net(recon_image_feature, context1)

        B, _, H, W = x.size()
        pixel_num = H * W
        mse = self.mse(x, recon_image)
        ssim = self.ssim(x, recon_image)
        me_mse = self.mse(x, warp_frame)
        mse = torch.sum(mse, dim=(1, 2, 3)) / pixel_num
        me_mse = torch.sum(me_mse, dim=(1, 2, 3)) / pixel_num

        if self.training:
            y_for_bit = self.add_noise(y_res)
            mv_y_for_bit = self.add_noise(mv_y_res)
            z_for_bit = self.add_noise(z)
            mv_z_for_bit = self.add_noise(mv_z)
        else:
            y_for_bit = y_q
            mv_y_for_bit = mv_y_q
            z_for_bit = z_hat
            mv_z_for_bit = mv_z_hat
        bits_y = self.get_y_laplace_bits(y_for_bit, scales_hat)
        bits_mv_y = self.get_y_laplace_bits(mv_y_for_bit, mv_scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        bits_mv_z = self.get_z_bits(mv_z_for_bit, self.bit_estimator_z_mv)

        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bpp_mv_y = torch.sum(bits_mv_y, dim=(1, 2, 3)) / pixel_num
        bpp_mv_z = torch.sum(bits_mv_z, dim=(1, 2, 3)) / pixel_num

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z
        bit = torch.sum(bpp) * pixel_num
        bit_y = torch.sum(bpp_y) * pixel_num
        bit_z = torch.sum(bpp_z) * pixel_num
        bit_mv_y = torch.sum(bpp_mv_y) * pixel_num
        bit_mv_z = torch.sum(bpp_mv_z) * pixel_num

        return {"bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "me_mse": me_mse,
                "mse": mse,
                "ssim": ssim,
                "dpb": {
                    "ref_frame": recon_image,
                    "ref_feature": feature,
                    "ref_y": y_hat,
                    "ref_mv_y": mv_y_hat,
                },
                "bit": bit,
                "bit_y": bit_y,
                "bit_z": bit_z,
                "bit_mv_y": bit_mv_y,
                "bit_mv_z": bit_mv_z,
                }
