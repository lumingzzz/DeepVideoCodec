# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

backward_grid = [{} for _ in range(9)]    # 0~7 for GPU, -1 for CPU


def conv1x1(in_ch, out_ch, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv1x1(in_ch, out_ch, r=1):
    """1x1 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )


def subpel_conv3x3(in_ch, out_ch, r=1):
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch, out_ch, leaky_relu_slope=0.01):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.conv2 = conv3x3(out_ch, out_ch)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        out = out + identity
        return out


class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU()
        self.conv2 = conv3x3(out_ch, out_ch)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        if stride != 1:
            self.downsample = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch, out_ch, upsample=2):
        super().__init__()
        self.subpel_conv = subpel_conv1x1(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU()
        self.conv = conv3x3(out_ch, out_ch)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.upsample = subpel_conv1x1(in_ch, out_ch, upsample)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.leaky_relu2(out)
        identity = self.upsample(x)
        out += identity
        return out


def torch_warp(feature, flow):
    device_id = -1 if feature.device == torch.device('cpu') else feature.device.index
    if str(flow.size()) not in backward_grid[device_id]:
        N, _, H, W = flow.size()
        tensor_hor = torch.linspace(-1.0, 1.0, W, device=feature.device, dtype=feature.dtype).view(
            1, 1, 1, W).expand(N, -1, H, -1)
        tensor_ver = torch.linspace(-1.0, 1.0, H, device=feature.device, dtype=feature.dtype).view(
            1, 1, H, 1).expand(N, -1, -1, W)
        backward_grid[device_id][str(flow.size())] = torch.cat([tensor_hor, tensor_ver], 1)

    flow = torch.cat([flow[:, 0:1, :, :] / ((feature.size(3) - 1.0) / 2.0),
                      flow[:, 1:2, :, :] / ((feature.size(2) - 1.0) / 2.0)], 1)

    grid = (backward_grid[device_id][str(flow.size())] + flow)
    return torch.nn.functional.grid_sample(input=feature,
                                           grid=grid.permute(0, 2, 3, 1),
                                           mode='bilinear',
                                           padding_mode='border',
                                           align_corners=True)


def flow_warp(im, flow):
    warp = torch_warp(im, flow)
    return warp


def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(
        inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=False)
    return outfeature


class MEBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x


class ME_Spynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.L = 4
        self.moduleBasic = torch.nn.ModuleList([MEBasic() for _ in range(self.L)])

    def forward(self, im1, im2):
        batchsize = im1.size()[0]
        im1_pre = im1
        im2_pre = im2

        im1_list = [im1_pre]
        im2_list = [im2_pre]
        for level in range(self.L - 1):
            im1_list.append(F.avg_pool2d(im1_list[level], kernel_size=2, stride=2))
            im2_list.append(F.avg_pool2d(im2_list[level], kernel_size=2, stride=2))

        shape_fine = im2_list[self.L - 1].size()
        zero_shape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        flow = torch.zeros(zero_shape, dtype=im1.dtype, device=im1.device)
        for level in range(self.L):
            flow_up = bilinearupsacling(flow) * 2.0
            img_index = self.L - 1 - level
            flow = flow_up + \
                self.moduleBasic[level](torch.cat([im1_list[img_index],
                                                   flow_warp(im2_list[img_index], flow_up),
                                                   flow_up], 1))

        return flow


# class ResBlock(nn.Module):
#     def __init__(self, inputchannel, outputchannel, kernel_size, stride=1):
#         super(ResBlock, self).__init__()
#         self.relu1 = nn.ReLU()
#         self.conv1 = nn.Conv2d(inputchannel, outputchannel,
#                                kernel_size, stride, padding=kernel_size//2)
#         torch.nn.init.xavier_uniform_(self.conv1.weight.data)
#         torch.nn.init.constant_(self.conv1.bias.data, 0.0)
#         self.relu2 = nn.ReLU()
#         self.conv2 = nn.Conv2d(outputchannel, outputchannel,
#                                kernel_size, stride, padding=kernel_size//2)
#         torch.nn.init.xavier_uniform_(self.conv2.weight.data)
#         torch.nn.init.constant_(self.conv2.bias.data, 0.0)
#         if inputchannel != outputchannel:
#             self.adapt_conv = nn.Conv2d(inputchannel, outputchannel, 1)
#             torch.nn.init.xavier_uniform_(self.adapt_conv.weight.data)
#             torch.nn.init.constant_(self.adapt_conv.bias.data, 0.0)
#         else:
#             self.adapt_conv = None

#     def forward(self, x):
#         x_1 = self.relu1(x)
#         firstlayer = self.conv1(x_1)
#         firstlayer = self.relu2(firstlayer)
#         seclayer = self.conv2(firstlayer)
#         if self.adapt_conv is None:
#             return x + seclayer
#         else:
#             return self.adapt_conv(x) + seclayer


# class MaskedConv2d(nn.Conv2d):
#     r"""Masked 2D convolution implementation, mask future "unseen" pixels.
#     Useful for building auto-regressive network components.

#     Introduced in `"Conditional Image Generation with PixelCNN Decoders"
#     <https://arxiv.org/abs/1606.05328>`_.

#     Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
#     first layer (which also masks the "current pixel"), `mask_type='B'` for the
#     following layers.
#     """

#     def __init__(self, *args, mask_type="A", **kwargs):
#         super().__init__(*args, **kwargs)

#         if mask_type not in ("A", "B"):
#             raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

#         self.register_buffer("mask", torch.ones_like(self.weight.data))
#         _, _, h, w = self.mask.size()
#         self.mask[:, :, h // 2, w // 2 + (mask_type == "B"):] = 0
#         self.mask[:, :, h // 2 + 1:] = 0

#     def forward(self, x):
#         # TODO(begaintj): weight assigment is not supported by torchscript
#         self.weight.data *= self.mask
#         return super().forward(x)


# def conv3x3(in_ch, out_ch, stride=1):
#     """3x3 convolution with padding."""
#     return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


# def subpel_conv3x3(in_ch, out_ch, r=1):
#     """3x3 sub-pixel convolution for up-sampling."""
#     return nn.Sequential(
#         nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
#     )


# def conv1x1(in_ch, out_ch, stride=1):
#     """1x1 convolution."""
#     return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


# class ResidualBlockWithStride(nn.Module):
#     """Residual block with a stride on the first convolution.

#     Args:
#         in_ch (int): number of input channels
#         out_ch (int): number of output channels
#         stride (int): stride value (default: 2)
#     """

#     def __init__(self, in_ch, out_ch, stride=2):
#         super().__init__()
#         self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
#         self.leaky_relu = nn.LeakyReLU(inplace=True)
#         self.conv2 = conv3x3(out_ch, out_ch)
#         self.gdn = GDN(out_ch)
#         if stride != 1:
#             self.downsample = conv1x1(in_ch, out_ch, stride=stride)
#         else:
#             self.downsample = None

#     def forward(self, x):
#         identity = x
#         out = self.conv1(x)
#         out = self.leaky_relu(out)
#         out = self.conv2(out)
#         out = self.gdn(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         return out


# class ResidualBlockUpsample(nn.Module):
#     """Residual block with sub-pixel upsampling on the last convolution.

#     Args:
#         in_ch (int): number of input channels
#         out_ch (int): number of output channels
#         upsample (int): upsampling factor (default: 2)
#     """

#     def __init__(self, in_ch, out_ch, upsample=2):
#         super().__init__()
#         self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
#         self.leaky_relu = nn.LeakyReLU(inplace=True)
#         self.conv = conv3x3(out_ch, out_ch)
#         self.igdn = GDN(out_ch, inverse=True)
#         self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

#     def forward(self, x):
#         identity = x
#         out = self.subpel_conv(x)
#         out = self.leaky_relu(out)
#         out = self.conv(out)
#         out = self.igdn(out)
#         identity = self.upsample(x)
#         out += identity
#         return out


# class ResidualBlock(nn.Module):
#     """Simple residual block with two 3x3 convolutions.

#     Args:
#         in_ch (int): number of input channels
#         out_ch (int): number of output channels
#     """

#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv1 = conv3x3(in_ch, out_ch)
#         self.leaky_relu = nn.LeakyReLU(inplace=True)
#         self.conv2 = conv3x3(out_ch, out_ch)

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.leaky_relu(out)
#         out = self.conv2(out)
#         out = self.leaky_relu(out)

#         out = out + identity
#         return out