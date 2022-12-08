import math
from typing import List

import torch
import torch.nn as nn

from compressai.entropy_models import GaussianConditional

from .base_model import CompressionModel, get_scale_table
from .utils import quantize_ste, update_registered_buffers
from .layers import ME_Spynet, ResBlock, UNet, subpel_conv3x3, \
        get_enc_dec_models, get_hyper_enc_dec_models, flow_warp, bilineardownsacling


class FeatureExtractor(nn.Module):
    def __init__(self, channel=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        self.res_block1 = ResBlock(channel)
        self.conv2 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
        self.res_block2 = ResBlock(channel)
        self.conv3 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
        self.res_block3 = ResBlock(channel)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)
        return layer1, layer2, layer3


class MultiScaleContextFusion(nn.Module):
    def __init__(self, channel_in=64, channel_out=64):
        super().__init__()
        self.conv3_up = subpel_conv3x3(channel_in, channel_out, 2)
        self.res_block3_up = ResBlock(channel_out)
        self.conv3_out = nn.Conv2d(channel_out, channel_out, 3, padding=1)
        self.res_block3_out = ResBlock(channel_out)
        self.conv2_up = subpel_conv3x3(channel_out * 2, channel_out, 2)
        self.res_block2_up = ResBlock(channel_out)
        self.conv2_out = nn.Conv2d(channel_out * 2, channel_out, 3, padding=1)
        self.res_block2_out = ResBlock(channel_out)
        self.conv1_out = nn.Conv2d(channel_out * 2, channel_out, 3, padding=1)
        self.res_block1_out = ResBlock(channel_out)

    def forward(self, context1, context2, context3):
        context3_up = self.conv3_up(context3)
        context3_up = self.res_block3_up(context3_up)
        context3_out = self.conv3_out(context3)
        context3_out = self.res_block3_out(context3_out)
        context2_up = self.conv2_up(torch.cat((context3_up, context2), dim=1))
        context2_up = self.res_block2_up(context2_up)
        context2_out = self.conv2_out(torch.cat((context3_up, context2), dim=1))
        context2_out = self.res_block2_out(context2_out)
        context1_out = self.conv1_out(torch.cat((context2_up, context1), dim=1))
        context1_out = self.res_block1_out(context1_out)
        context1 = context1 + context1_out
        context2 = context2 + context2_out
        context3 = context3 + context3_out
        return context1, context2, context3


class ContextualEncoder(nn.Module):
    def __init__(self, N=64, M=96):
        super().__init__()
        self.conv1 = nn.Conv2d(N + 3, N, 3, stride=2, padding=1)
        self.res1 = ResBlock(N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.conv2 = nn.Conv2d(N * 2, N, 3, stride=2, padding=1)
        self.res2 = ResBlock(N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.conv3 = nn.Conv2d(N * 2, N, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(N, M, 3, stride=2, padding=1)

    def forward(self, x, context1, context2, context3):
        feature = self.conv1(torch.cat([x, context1], dim=1))
        feature = self.res1(torch.cat([feature, context2], dim=1))
        feature = self.conv2(feature)
        feature = self.res2(torch.cat([feature, context3], dim=1))
        feature = self.conv3(feature)
        feature = self.conv4(feature)
        return feature


class ContextualDecoder(nn.Module):
    def __init__(self, N=64, M=96):
        super().__init__()
        self.up1 = subpel_conv3x3(M, N, 2)
        self.up2 = subpel_conv3x3(N, N, 2)
        self.res1 = ResBlock(N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.up3 = subpel_conv3x3(N * 2, N, 2)
        self.res2 = ResBlock(N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.up4 = subpel_conv3x3(N * 2, 32, 2)

    def forward(self, x, context2, context3):
        feature = self.up1(x)
        feature = self.up2(feature)
        feature = self.res1(torch.cat([feature, context3], dim=1))
        feature = self.up3(feature)
        feature = self.res2(torch.cat([feature, context2], dim=1))
        feature = self.up4(feature)
        return feature


class ReconGeneration(nn.Module):
    def __init__(self, ctx_channel=64, res_channel=32, channel=64):
        super().__init__()
        self.first_conv = nn.Conv2d(ctx_channel + res_channel, channel, 3, stride=1, padding=1)
        self.unet_1 = UNet(channel)
        self.unet_2 = UNet(channel)
        self.recon_conv = nn.Conv2d(channel, 3, 3, stride=1, padding=1)

    def forward(self, ctx, res):
        feature = self.first_conv(torch.cat((ctx, res), dim=1))
        feature = self.unet_1(feature)
        feature = self.unet_2(feature)
        recon = self.recon_conv(feature)
        return feature, recon


class MotionContextModel(CompressionModel):
    def __init__(self, ch_mv: int = 64):
        super().__init__(entropy_bottleneck_channels=ch_mv)
        self.hyper_encoder, self.hyper_decoder = \
            get_hyper_enc_dec_models(ch_mv, ch_mv)

        self.y_prior_fusion = nn.Sequential(
            nn.Conv2d(ch_mv * 3, ch_mv * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ch_mv * 3, ch_mv * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ch_mv * 3, ch_mv * 2, 3, stride=1, padding=1)
        )

        self.y_spatial_prior = nn.Sequential(
            nn.Conv2d(ch_mv * 3, ch_mv * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ch_mv * 3, ch_mv * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ch_mv * 3, ch_mv * 2, 3, stride=1, padding=1)
        )

        self.gaussian_conditional = GaussianConditional(None)

    @staticmethod
    def get_mask(height, width, device):
        micro_mask = torch.tensor(((1, 0), (0, 1)), dtype=torch.float32, device=device)
        mask_0 = micro_mask.repeat(height // 2, width // 2)
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_1 = torch.ones_like(mask_0) - mask_0
        return mask_0, mask_1

    def process_with_mask(self, y, means, scales, mask):
        means_hat = means * mask
        scales_hat = scales * mask

        y_hat = quantize_ste((y - means_hat) * mask) + means_hat
        return y_hat, means_hat, scales_hat

    def forward_dual_prior(self, y, means, scales):
        '''
        y_0 means split in channel, the first half
        y_1 means split in channel, the second half
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        '''
        device = y.device
        _, _, H, W = y.size()
        mask_0, mask_1 = self.get_mask(H, W, device)

        y_0, y_1 = y.chunk(2, 1)
        means_0, means_1 = means.chunk(2, 1)
        scales_0, scales_1 = scales.chunk(2, 1)

        y_hat_0_0, means_hat_0_0, scales_hat_0_0 = \
            self.process_with_mask(y_0, means_0, scales_0, mask_0)
        y_hat_1_1, means_hat_1_1, scales_hat_1_1 = \
            self.process_with_mask(y_1, means_1, scales_1, mask_1)

        params = torch.cat((y_hat_0_0, y_hat_1_1, means, scales), dim=1)
        means_0, scales_0, means_1, scales_1 = self.y_spatial_prior(params).chunk(4, 1)

        y_hat_0_1, means_hat_0_1, scales_hat_0_1 = \
            self.process_with_mask(y_0, means_0, scales_0, mask_1)
        y_hat_1_0, means_hat_1_0, scales_hat_1_0 = \
            self.process_with_mask(y_1, means_1, scales_1, mask_0)

        y_hat_0 = y_hat_0_0 + y_hat_0_1
        means_hat_0 = means_hat_0_0 + means_hat_0_1 
        scales_hat_0 = scales_hat_0_0 + scales_hat_0_1 

        y_hat_1 = y_hat_1_1 + y_hat_1_0
        means_hat_1 = means_hat_1_1 + means_hat_1_0 
        scales_hat_1 = scales_hat_1_1 + scales_hat_1_0

        y_hat = torch.cat((y_hat_0, y_hat_1), dim=1)
        means_hat = torch.cat((means_hat_0, means_hat_1), dim=1)
        scales_hat = torch.cat((scales_hat_0, scales_hat_1), dim=1)
        return y_hat, means_hat, scales_hat

    def forward(self, y, y_ref):
        z = self.hyper_encoder(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset

        params = self.hyper_decoder(z_hat)
        if y_ref is None:
            y_ref = torch.zeros_like(y)
        means_hat, scales_hat = self.y_prior_fusion(torch.cat((params, y_ref), dim=1)).chunk(2, 1)
        y_hat, means_hat, scales_hat = self.forward_dual_prior(y, means_hat, scales_hat)

        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means_hat)
        return y_hat, {"y": y_likelihoods, "z": z_likelihoods}

    def compress(self, y, y_ref):
        z = self.hyper_encoder(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        
        params = self.hyper_decoder(z_hat)
        if y_ref is None:
            y_ref = torch.zeros_like(y)
        means_hat, scales_hat = self.y_prior_fusion(torch.cat((params, y_ref), dim=1)).chunk(2, 1)
        y_hat, means_hat, scales_hat = self.forward_dual_prior(y, means_hat, scales_hat)
        
        params = self.y_mv_prior_fusion(torch.cat((params, y_ref), dim=1))
        
        ctx_params = torch.zeros_like(params)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        _, means_hat = gaussian_params.chunk(2, 1)
        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat)
        
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
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat)
        
        y_anchor, y_non_anchor = Demultiplexer(y)
        scales_hat_anchor, scales_hat_non_anchor = Demultiplexer(scales_hat)
        means_hat_anchor, means_hat_non_anchor = Demultiplexer(means_hat)

        indexes_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
        indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)

        anchor_strings = self.gaussian_conditional.compress(y_anchor, indexes_anchor, means=means_hat_anchor)
        non_anchor_strings = self.gaussian_conditional.compress(y_non_anchor, indexes_non_anchor, means=means_hat_non_anchor)

        return y_hat, {"strings": [anchor_strings, non_anchor_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, y_ref):
        assert isinstance(strings, list) and len(strings) == 3
        z_hat = self.entropy_bottleneck.decompress(strings[2], shape)
        N, _, H, W = z_hat.shape
        params = self.hyper_decoder(z_hat)
        
        if y_ref is None:
            y_ref = torch.zeros([N, params.size(1)//2, H * 4, W * 4]).to(z_hat.device)
        params = self.y_mv_prior_fusion(torch.cat((params, y_ref), dim=1))
        
        ctx_params = torch.zeros_like(params)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        scales_hat_anchor, _ = Demultiplexer(scales_hat)
        means_hat_anchor, _ = Demultiplexer(means_hat)
        
        indexes_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
        y_anchor = self.gaussian_conditional.decompress(strings[0], indexes_anchor, means=means_hat_anchor)     # [1, 384, 8, 8]
        y_anchor = Multiplexer(y_anchor, torch.zeros_like(y_anchor))    # [1, 192, 16, 16]
        
        ctx_params = self.context_prediction(y_anchor)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )

        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, scales_hat_non_anchor = Demultiplexer(scales_hat)
        _, means_hat_non_anchor = Demultiplexer(means_hat)
        
        indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
        y_non_anchor = self.gaussian_conditional.decompress(strings[1], indexes_non_anchor, means=means_hat_non_anchor)  # [1, 384, 8, 8]
        y_non_anchor = Multiplexer(torch.zeros_like(y_non_anchor), y_non_anchor)    # [1, 192, 16, 16]
        
        # gather
        y_hat = y_anchor + y_non_anchor

        return y_hat


class FrameContextModel(CompressionModel):
    def __init__(self, N: int = 64, M: int = 64):
        super().__init__(entropy_bottleneck_channels=N)
        self.hyper_encoder, self.hyper_decoder = \
            get_hyper_enc_dec_models(M, N)

        self.temporal_prior_encoder = nn.Sequential(
            nn.Conv2d(N, M * 3 // 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(M * 3 // 2, M * 2, 3, stride=2, padding=1),
        )

        self.y_prior_fusion = nn.Sequential(
            nn.Conv2d(M * 5, M * 4, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(M * 4, M * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(M * 3, M * 2, 3, stride=1, padding=1)
        )

        self.y_spatial_prior = nn.Sequential(
            nn.Conv2d(M * 3, M * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(M * 3, M * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(M * 3, M * 2, 3, padding=1)
        )

        self.gaussian_conditional = GaussianConditional(None)

    @staticmethod
    def get_mask(height, width, device):
        micro_mask = torch.tensor(((1, 0), (0, 1)), dtype=torch.float32, device=device)
        mask_0 = micro_mask.repeat(height // 2, width // 2)
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_1 = torch.ones_like(mask_0) - mask_0
        return mask_0, mask_1

    def process_with_mask(self, y, means, scales, mask):
        means_hat = means * mask
        scales_hat = scales * mask

        y_hat = quantize_ste((y - means_hat) * mask) + means_hat
        return y_hat, means_hat, scales_hat

    def forward_dual_prior(self, y, means, scales):
        '''
        y_0 means split in channel, the first half
        y_1 means split in channel, the second half
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        '''
        device = y.device
        _, _, H, W = y.size()
        mask_0, mask_1 = self.get_mask(H, W, device)

        y_0, y_1 = y.chunk(2, 1)
        means_0, means_1 = means.chunk(2, 1)
        scales_0, scales_1 = scales.chunk(2, 1)

        y_hat_0_0, means_hat_0_0, scales_hat_0_0 = \
            self.process_with_mask(y_0, means_0, scales_0, mask_0)
        y_hat_1_1, means_hat_1_1, scales_hat_1_1 = \
            self.process_with_mask(y_1, means_1, scales_1, mask_1)

        params = torch.cat((y_hat_0_0, y_hat_1_1, means, scales), dim=1)
        means_0, scales_0, means_1, scales_1 = self.y_spatial_prior(params).chunk(4, 1)

        y_hat_0_1, means_hat_0_1, scales_hat_0_1 = \
            self.process_with_mask(y_0, means_0, scales_0, mask_1)
        y_hat_1_0, means_hat_1_0, scales_hat_1_0 = \
            self.process_with_mask(y_1, means_1, scales_1, mask_0)

        y_hat_0 = y_hat_0_0 + y_hat_0_1
        means_hat_0 = means_hat_0_0 + means_hat_0_1 
        scales_hat_0 = scales_hat_0_0 + scales_hat_0_1 

        y_hat_1 = y_hat_1_1 + y_hat_1_0
        means_hat_1 = means_hat_1_1 + means_hat_1_0 
        scales_hat_1 = scales_hat_1_1 + scales_hat_1_0

        y_hat = torch.cat((y_hat_0, y_hat_1), dim=1)
        means_hat = torch.cat((means_hat_0, means_hat_1), dim=1)
        scales_hat = torch.cat((scales_hat_0, scales_hat_1), dim=1)
        return y_hat, means_hat, scales_hat

    def forward(self, y, y_ref, context):
        z = self.hyper_encoder(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset

        params = self.hyper_decoder(z_hat)
        if y_ref is None:
            y_ref = torch.zeros_like(y)
        temporal_params = self.temporal_prior_encoder(context)
        means_hat, scales_hat = self.y_prior_fusion(torch.cat((temporal_params, params, y_ref), dim=1)).chunk(2, 1)
        y_hat, means_hat, scales_hat = self.forward_dual_prior(y, means_hat, scales_hat)

        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means_hat)
        return y_hat, {"y": y_likelihoods, "z": z_likelihoods}

    # def compress(self, y, y_ref, context):
    #     z = self.hyper_encoder(y)

    #     z_strings = self.entropy_bottleneck.compress(z)
    #     z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        
    #     params = self.hyper_decoder(z_hat)
        
    #     if y_ref is None:
    #         y_ref = torch.zeros_like(y)
    #     temporal_params = self.temporal_prior_encoder(context)
    #     params = self.y_prior_fusion(torch.cat((temporal_params, params, y_ref), dim=1))
        
    #     ctx_params = torch.zeros_like(params)
    #     gaussian_params = self.entropy_parameters(
    #         torch.cat((params, ctx_params), dim=1)
    #     )
    #     _, means_hat = gaussian_params.chunk(2, 1)
    #     y_hat = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat)
        
    #     # set non_anchor to 0
    #     y_half = y_hat.clone()
    #     y_half[:, :, 0::2, 0::2] = 0
    #     y_half[:, :, 1::2, 1::2] = 0

    #     # set anchor's ctx to 0, otherwise there will be a bias
    #     ctx_params = self.context_prediction(y_half)
    #     ctx_params[:, :, 0::2, 1::2] = 0
    #     ctx_params[:, :, 1::2, 0::2] = 0

    #     gaussian_params = self.entropy_parameters(
    #         torch.cat((params, ctx_params), dim=1)
    #     )
    #     scales_hat, means_hat = gaussian_params.chunk(2, 1)
    #     y_hat = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat)
        
    #     y_anchor, y_non_anchor = Demultiplexer(y)
    #     scales_hat_anchor, scales_hat_non_anchor = Demultiplexer(scales_hat)
    #     means_hat_anchor, means_hat_non_anchor = Demultiplexer(means_hat)

    #     indexes_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
    #     indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)

    #     anchor_strings = self.gaussian_conditional.compress(y_anchor, indexes_anchor, means=means_hat_anchor)
    #     non_anchor_strings = self.gaussian_conditional.compress(y_non_anchor, indexes_non_anchor, means=means_hat_non_anchor)

    #     return y_hat, {"strings": [anchor_strings, non_anchor_strings, z_strings], "shape": z.size()[-2:]}

    # def decompress(self, strings, shape, y_ref, context):
    #     assert isinstance(strings, list) and len(strings) == 3
    #     z_hat = self.entropy_bottleneck.decompress(strings[2], shape)
    #     N, _, H, W = z_hat.shape
    #     params = self.hyper_decoder(z_hat)
        
    #     if y_ref is None:
    #         y_ref = torch.zeros([N, params.size(1)//2, H * 4, W * 4]).to(z_hat.device)
    #     temporal_params = self.temporal_prior_encoder(context)
    #     params = self.y_prior_fusion(torch.cat((temporal_params, params, y_ref), dim=1))
        
    #     ctx_params = torch.zeros_like(params)
    #     gaussian_params = self.entropy_parameters(
    #         torch.cat((params, ctx_params), dim=1)
    #     )
    #     scales_hat, means_hat = gaussian_params.chunk(2, 1)
    #     scales_hat_anchor, _ = Demultiplexer(scales_hat)
    #     means_hat_anchor, _ = Demultiplexer(means_hat)
        
    #     indexes_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
    #     y_anchor = self.gaussian_conditional.decompress(strings[0], indexes_anchor, means=means_hat_anchor)     # [1, 384, 8, 8]
    #     y_anchor = Multiplexer(y_anchor, torch.zeros_like(y_anchor))    # [1, 192, 16, 16]
        
    #     ctx_params = self.context_prediction(y_anchor)
    #     gaussian_params = self.entropy_parameters(
    #         torch.cat((params, ctx_params), dim=1)
    #     )

    #     scales_hat, means_hat = gaussian_params.chunk(2, 1)
    #     _, scales_hat_non_anchor = Demultiplexer(scales_hat)
    #     _, means_hat_non_anchor = Demultiplexer(means_hat)
        
    #     indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
    #     y_non_anchor = self.gaussian_conditional.decompress(strings[1], indexes_non_anchor, means=means_hat_non_anchor)  # [1, 384, 8, 8]
    #     y_non_anchor = Multiplexer(torch.zeros_like(y_non_anchor), y_non_anchor)    # [1, 192, 16, 16]
        
    #     # gather
    #     y_hat = y_anchor + y_non_anchor

    #     return y_hat


class DMC(nn.Module):
    def __init__(self, ch_mv=64, N=64, M=96):
        super().__init__()

        self.optic_flow = ME_Spynet()

        self.motion_encoder, self.motion_decoder = get_enc_dec_models(2, 2, ch_mv)
        self.motion_context_model = MotionContextModel(ch_mv)

        self.feature_adaptor_I = nn.Conv2d(3, N, 3, stride=1, padding=1)
        self.feature_adaptor_P = nn.Conv2d(N, N, 1)
        self.feature_extractor = FeatureExtractor()
        self.context_fusion_net = MultiScaleContextFusion()

        self.contextual_encoder = ContextualEncoder(N, M)
        self.contextual_decoder = ContextualDecoder(N, M)
        self.frame_context_model = FrameContextModel(N, M)
        self.recon_generation_net = ReconGeneration()

        self._initialize_weights()

    def multi_scale_feature_extractor(self, dpb):
        if dpb["feature_ref"] is None:
            feature = self.feature_adaptor_I(dpb["x_ref"])
        else:
            feature = self.feature_adaptor_P(dpb["feature_ref"])
        return self.feature_extractor(feature)

    def motion_compensation(self, mv, dpb):
        warpframe = flow_warp(dpb["x_ref"], mv)
        mv2 = bilineardownsacling(mv) / 2
        mv3 = bilineardownsacling(mv2) / 2
        ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(dpb)
        context1 = flow_warp(ref_feature1, mv)
        context2 = flow_warp(ref_feature2, mv2)
        context3 = flow_warp(ref_feature3, mv3)
        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)
        return context1, context2, context3, warpframe

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.xavier_normal_(m.weight, math.sqrt(2))
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)

    def forward(self, frames, motion_pretrain=False, frame_pretrain=False):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")
        
        reconstructions = []
        frames_likelihoods = []

        # i frame
        x_rec = frames[0]
        # x_hat, likelihoods = self.forward_keyframe(frames[0])
        # reconstructions.append(x_hat)
        # frames_likelihoods.append(likelihoods)
        # x_ref = x_hat.detach()  # stop gradient flow (cf: google2020 paper)
        
        dpb = {
                "y_mv_ref": None,
                "y_ref": None,
                "feature_ref": None,
                "x_ref": x_rec,
                }

        # p frame
        for i in range(1, len(frames)):
            x = frames[i]
            x_rec, likelihoods, context = self.forward_inter(x, dpb, motion_pretrain, frame_pretrain)
            reconstructions.append(x_rec)
            frames_likelihoods.append(likelihoods)
            
            if len(frames) >= 3:
                dpb = {
                        "y_mv_ref": context["y_mv_ref"],
                        "y_ref": context["y_ref"],
                        "feature_ref": context["feature_ref"],
                        "x_ref": x_rec,
                    }

        return {
            "x_hat": reconstructions,
            "likelihoods": frames_likelihoods,
        }

    def forward_inter(self, x_cur, dpb, motion_pretrain=False, frame_pretrain=False):
        x_ref = dpb["x_ref"]
        mv = self.optic_flow(x_cur, x_ref)
        y_mv = self.motion_encoder(mv)
        y_mv_hat, mv_likelihoods = self.motion_context_model(y_mv, dpb["y_mv_ref"])

        mv_hat = self.motion_decoder(y_mv_hat)
        context1, context2, context3, x_warp = self.motion_compensation(mv_hat, dpb)

        if motion_pretrain:
            return x_warp, {"motion": mv_likelihoods}, {}

        elif frame_pretrain:
            mv_hat = mv_hat.detach()
        
        y = self.contextual_encoder(x_cur, context1, context2, context3)
        y_hat, frame_likelihoods = self.frame_context_model(y, dpb["y_ref"], context3)

        x_rec_feature = self.contextual_decoder(y_hat, context2, context3)
        feature, x_rec = self.recon_generation_net(x_rec_feature, context1)

        return x_rec, \
               {"motion": mv_likelihoods, "frame": frame_likelihoods}, \
               {"y_mv_ref": y_mv_hat, "y_ref": y_hat, "feature_ref": feature, "x_ref": x_rec}
    
    def encode_inter(self, x_cur, dpb):
        x_ref = dpb["x_ref"]
        mv = self.optic_flow(x_cur, x_ref)
        y_mv = self.motion_encoder(mv)
        y_mv_hat, mv_strings = self.motion_context_model.compress(y_mv, dpb["y_mv_ref"])

        mv_hat = self.motion_decoder(y_mv_hat)
        context1, context2, context3, _ = self.motion_compensation(mv_hat, dpb)

        y = self.contextual_encoder(x_cur, context1, context2, context3)
        _, frame_strings = self.frame_context_model.compress(y, dpb["y_ref"], context3)

        return {
            "strings": {
                "motion": mv_strings["strings"],
                "frame": frame_strings["strings"],
            },
            "shape": {"motion": mv_strings["shape"], "frame": frame_strings["shape"]},
        }

    def decode_inter(self, strings, shapes, dpb):
        key = "motion"
        y_mv_hat = self.motion_context_model.decompress(strings[key], shapes[key], dpb["y_mv_ref"])

        mv_hat = self.motion_decoder(y_mv_hat)
        context1, context2, context3, _ = self.motion_compensation(mv_hat, dpb)

        key = "frame"
        y_hat = self.frame_context_model.decompress(strings[key], shapes[key], dpb["y_ref"], context3)

        x_rec_feature = self.contextual_decoder(y_hat, context1, context2, context3)
        feature, x_rec = self.recon_generation_net(x_rec_feature, context1)
    
        return x_rec, {"x_ref": x_rec, "feature_ref": feature, "y_ref": y_hat, "y_mv_ref": y_mv_hat}

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""

        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())

        return aux_loss_list

    def load_state_dict(self, state_dict, strict=True):

        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.motion_context_model.gaussian_conditional,
            "motion_context_model.gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        
        update_registered_buffers(
            self.motion_context_model.entropy_bottleneck,
            "motion_context_model.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )

        update_registered_buffers(
            self.frame_context_model.gaussian_conditional,
            "frame_context_model.gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.frame_context_model.entropy_bottleneck,
            "frame_context_model.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )

        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()

        updated = self.motion_context_model.gaussian_conditional.update_scale_table(
            scale_table, force=force
        )
        updated |= self.motion_context_model.entropy_bottleneck.update(force=force)

        updated |= self.frame_context_model.gaussian_conditional.update_scale_table(
            scale_table, force=force
        )
        updated |= self.frame_context_model.entropy_bottleneck.update(force=force)

        return updated