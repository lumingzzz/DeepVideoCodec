import math

from typing import List

import torch
import torch.nn as nn

from compressai.entropy_models import GaussianConditional

from .base_model import CompressionModel, get_scale_table
from .utils import quantize_ste, update_registered_buffers, Demultiplexer, Multiplexer
from ..layers import ME_Spynet, ResBlock, UNet, conv1x1, subpel_conv3x3, \
        get_enc_dec_models, get_hyper_enc_dec_models, flow_warp, bilineardownsacling, \
        CheckerboardMaskedConv2d 


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
    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_N + 3, channel_N, 3, stride=2, padding=1)
        self.res1 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.conv2 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
        self.res2 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.conv3 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(channel_N, channel_M, 3, stride=2, padding=1)

    def forward(self, x, context1, context2, context3):
        feature = self.conv1(torch.cat([x, context1], dim=1))
        feature = self.res1(torch.cat([feature, context2], dim=1))
        feature = self.conv2(feature)
        feature = self.res2(torch.cat([feature, context3], dim=1))
        feature = self.conv3(feature)
        feature = self.conv4(feature)
        return feature


class ContextualDecoder(nn.Module):
    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.up1 = subpel_conv3x3(channel_M, channel_N, 2)
        self.up2 = subpel_conv3x3(channel_N, channel_N, 2)
        self.res1 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.up3 = subpel_conv3x3(channel_N * 2, channel_N, 2)
        self.res2 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.up4 = subpel_conv3x3(channel_N * 2, 32, 2)

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


class DualSpatialPriorModel(CompressionModel):
    def __init__(self, planes: int = 64, latent_planes: int = 64):
        super().__init__(entropy_bottleneck_channels=latent_planes)
        self.hyper_encoder, self.hyper_decoder = \
            get_hyper_enc_dec_models(planes, latent_planes)
        self.gaussian_conditional = GaussianConditional(None)
        
        self.context_prediction = CheckerboardMaskedConv2d(
            planes, 2 * planes, kernel_size=5, padding=2, stride=1
        )

        self.y_mv_prior_fusion = nn.Sequential(
            nn.Conv2d(planes * 3, planes * 3, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(planes * 3, planes * 3, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(planes * 3, planes * 2, 3, stride=1, padding=1)
        )
        
        self.entropy_parameters = nn.Sequential(
                conv1x1(planes*12//3, planes*10//3, 1),
                nn.GELU(),
                conv1x1(planes*10//3, planes*8//3, 1),
                nn.GELU(),
                conv1x1(planes*8//3, planes*6//3, 1),
        ) 

    def forward(self, y, y_ref):
        z = self.hyper_encoder(y)
        # z_hat, z_likelihoods = self.entropy_bottleneck(z)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset

        params = self.hyper_decoder(z_hat)
        if y_ref is None:
            y_ref = torch.zeros_like(y)
        params = self.y_mv_prior_fusion(torch.cat((params, y_ref), dim=1))

        # y_hat = self.gaussian_conditional.quantize(
        #     y, "noise" if self.training else "dequantize"
        # )
        
        ctx_params = torch.zeros_like(params)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat = quantize_ste(y - means_hat) + means_hat
        
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
        
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means_hat)
        y_hat = quantize_ste(y - means_hat) + means_hat
        return y_hat, {"y": y_likelihoods, "z": z_likelihoods}

    def compress(self, y, y_ref):
        z = self.hyper_encoder(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        
        params = self.hyper_decoder(z_hat)
        
        if y_ref is None:
            y_ref = torch.zeros_like(y)
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


class DMC(nn.Module):
    def __init__(self):
        super().__init__()

        channel_mv = 64
        channel_N = 64
        channel_M = 96

        self.channel_mv = channel_mv
        self.channel_N = channel_N
        self.channel_M = channel_M

        self.optic_flow = ME_Spynet()

        self.mv_encoder, self.mv_decoder = get_enc_dec_models(2, 2, channel_mv)
        self.mv_context_model = DualSpatialPriorModel(planes=channel_mv, latent_planes=channel_N)

        self.feature_adaptor_I = nn.Conv2d(3, channel_N, 3, stride=1, padding=1)
        self.feature_adaptor_P = nn.Conv2d(channel_N, channel_N, 1)
        self.feature_extractor = FeatureExtractor()
        self.context_fusion_net = MultiScaleContextFusion()

        self.contextual_encoder = ContextualEncoder(channel_N=channel_N, channel_M=channel_M)
        self.contextual_decoder = ContextualDecoder(channel_N=channel_N, channel_M=channel_M)
        self.ctx_context_model = DualSpatialPriorModel(planes=channel_M, latent_planes=channel_N)
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

    def forward(self, frames):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")
        
        reconstructions = []
        frames_likelihoods = []

        # reconstructions.append(frames[0])
        # frames_likelihoods.append({})
        x_rec = frames[0].detach()
        
        dpb = {
                    "x_ref": x_rec,
                    "feature_ref": None,
                    "y_ref": None,
                    "y_ref_mv": None,
                }

        for i in range(1, len(frames)):
            x = frames[i]
            x_rec, likelihoods, info = self.forward_inter(x, dpb)
            reconstructions.append(x_rec)
            frames_likelihoods.append(likelihoods)
            
            if len(frames) >= 3:
                dpb = {
                        "x_ref": x_rec,
                        "feature_ref": info["feature_ref"],
                        "y_ref": info["y_ref"],
                        "y_ref_mv": info["y_ref_mv"],
                    }

        return {
            "x_hat": reconstructions,
            "likelihoods": frames_likelihoods,
        }

    def forward_inter(self, x_cur, dpb):
        x_ref = dpb["x_ref"]

        motion = self.optic_flow(x_cur, x_ref)
        y_motion = self.mv_encoder(motion)
        y_motion_hat, motion_likelihoods = self.mv_context_model(y_motion, dpb["y_ref_mv"])
        
        x_motion_hat = self.mv_decoder(y_motion_hat)
        context1, context2, context3, x_warp = self.motion_compensation(x_motion_hat, dpb)
        
        # y = self.contextual_encoder(x_cur, context1, context2, context3)
        # y_hat, ctx_likelihoods = self.ctx_context_model(y, dpb["y_ref"])

        # x_rec_feature = self.contextual_decoder(y_hat, context2, context3)
        # feature, x_rec = self.recon_generation_net(x_rec_feature, context1)

        return x_warp, {"motion": motion_likelihoods}, {}
        # return x_rec, {"motion": motion_likelihoods, "context": ctx_likelihoods}
        # return x_rec, {"motion": motion_likelihoods, "context": ctx_likelihoods}, {"x_ref": x_rec, "feature_ref": feature, "y_ref": y_hat, "y_ref_mv": y_motion_hat}
        
    def encode_inter(self, x_cur, dpb):
        x_ref = dpb["x_ref"]
        
        motion = self.optic_flow(x_cur, x_ref)
        y_motion = self.mv_encoder(motion)
        y_motion_hat, out_motion = self.mv_context_model.compress(y_motion, dpb["y_ref_mv"])

        x_motion_hat = self.mv_decoder(y_motion_hat)
        context1, context2, context3, x_warp = self.motion_compensation(x_motion_hat, dpb)
        
        y = self.contextual_encoder(x_cur, context1, context2, context3)
        y_hat, out_y = self.ctx_context_model.compress(y, dpb["y_ref"])

        return {
            "strings": {
                "motion": out_motion["strings"],
                "context": out_y["strings"],
            },
            "shape": {"motion": out_motion["shape"], "context": out_y["shape"]},
        }

    def decode_inter(self, x_ref, strings, shapes, dpb):
        key = "motion"
        y_motion_hat = self.mv_context_model.decompress(strings[key], shapes[key], dpb["y_ref_mv"])

        x_motion_hat = self.mv_decoder(y_motion_hat)
        context1, context2, context3, x_warp = self.motion_compensation(x_motion_hat, dpb)

        # context
        key = "context"
        y_hat = self.ctx_context_model.decompress(strings[key], shapes[key], dpb["y_ref"])
        
        x_rec_feature = self.contextual_decoder(y_hat, context2, context3)
        feature, x_rec = self.recon_generation_net(x_rec_feature, context1)

        # return x_rec
    
        return x_rec, {"x_ref": x_rec, "feature_ref": feature, "y_ref": y_hat, "y_ref_mv": y_motion_hat}

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""

        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())

        return aux_loss_list

    def compress(self, frames):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")

        frame_strings = []
        shape_infos = []

        x_ref, out_keyframe = self.encode_keyframe(frames[0])

        frame_strings.append(out_keyframe["strings"])
        shape_infos.append(out_keyframe["shape"])

        for i in range(1, len(frames)):
            x = frames[i]
            x_ref, out_interframe = self.encode_inter(x, x_ref)

            frame_strings.append(out_interframe["strings"])
            shape_infos.append(out_interframe["shape"])

        return frame_strings, shape_infos

    def decompress(self, strings, shapes):

        if not isinstance(strings, List) or not isinstance(shapes, List):
            raise RuntimeError(f"Invalid number of frames: {len(strings)}.")

        assert len(strings) == len(
            shapes
        ), f"Number of information should match {len(strings)} != {len(shapes)}."

        dec_frames = []

        x_ref = self.decode_keyframe(strings[0], shapes[0])
        dec_frames.append(x_ref)

        for i in range(1, len(strings)):
            string = strings[i]
            shape = shapes[i]
            x_ref = self.decode_inter(x_ref, string, shape)
            dec_frames.append(x_ref)

        return dec_frames

    def load_state_dict(self, state_dict, strict=True):

        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.mv_context_model.gaussian_conditional,
            "mv_context_model.gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        
        update_registered_buffers(
            self.mv_context_model.entropy_bottleneck,
            "mv_context_model.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )

        update_registered_buffers(
            self.ctx_context_model.gaussian_conditional,
            "ctx_context_model.gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.ctx_context_model.entropy_bottleneck,
            "ctx_context_model.entropy_bottleneck",
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

        updated = self.mv_context_model.gaussian_conditional.update_scale_table(
            scale_table, force=force
        )
        updated |= self.mv_context_model.entropy_bottleneck.update(force=force)

        updated |= self.ctx_context_model.gaussian_conditional.update_scale_table(
            scale_table, force=force
        )
        updated |= self.ctx_context_model.entropy_bottleneck.update(force=force)

        return updated
