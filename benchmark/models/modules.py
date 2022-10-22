import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.layers import subpel_conv1x1, conv3x3, \
    ResidualBlock, ResidualBlockWithStride, ResidualBlockUpsample

backward_grid = [{} for _ in range(9)]    # 0~7 for GPU, -1 for CPU


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


def get_enc_dec_models(input_channel, output_channel, channel):
    enc = nn.Sequential(
        ResidualBlockWithStride(input_channel, channel, stride=2),
        ResidualBlock(channel, channel),
        ResidualBlockWithStride(channel, channel, stride=2),
        ResidualBlock(channel, channel),
        ResidualBlockWithStride(channel, channel, stride=2),
        ResidualBlock(channel, channel),
        conv3x3(channel, channel, stride=2),
    )

    dec = nn.Sequential(
        ResidualBlock(channel, channel),
        ResidualBlockUpsample(channel, channel, 2),
        ResidualBlock(channel, channel),
        ResidualBlockUpsample(channel, channel, 2),
        ResidualBlock(channel, channel),
        ResidualBlockUpsample(channel, channel, 2),
        ResidualBlock(channel, channel),
        subpel_conv1x1(channel, output_channel, 2),
    )

    return enc, dec


def get_hyper_enc_dec_models(y_channel, z_channel):
    enc = nn.Sequential(
        conv3x3(y_channel, z_channel),
        nn.LeakyReLU(),
        conv3x3(z_channel, z_channel),
        nn.LeakyReLU(),
        conv3x3(z_channel, z_channel, stride=2),
        nn.LeakyReLU(),
        conv3x3(z_channel, z_channel),
        nn.LeakyReLU(),
        conv3x3(z_channel, z_channel, stride=2),
    )

    dec = nn.Sequential(
        conv3x3(z_channel, y_channel),
        nn.LeakyReLU(),
        subpel_conv1x1(y_channel, y_channel, 2),
        nn.LeakyReLU(),
        conv3x3(y_channel, y_channel * 3 // 2),
        nn.LeakyReLU(),
        subpel_conv1x1(y_channel * 3 // 2, y_channel * 3 // 2, 2),
        nn.LeakyReLU(),
        conv3x3(y_channel * 3 // 2, y_channel * 2),
    )

    return enc, dec


# class FeatureExtractor(nn.Module):
#     def __init__(self, channel=64):
#         super().__init__()
#         self.conv1 = nn.Conv2d(channel, channel, 3, stride=1, padding=1)
#         self.res_block1 = ResBlock(channel)
#         self.conv2 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
#         self.res_block2 = ResBlock(channel)
#         self.conv3 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
#         self.res_block3 = ResBlock(channel)

#     def forward(self, feature):
#         layer1 = self.conv1(feature)
#         layer1 = self.res_block1(layer1)

#         layer2 = self.conv2(layer1)
#         layer2 = self.res_block2(layer2)

#         layer3 = self.conv3(layer2)
#         layer3 = self.res_block3(layer3)

#         return layer1, layer2, layer3


# class MultiScaleContextFusion(nn.Module):
#     def __init__(self, channel_in=64, channel_out=64):
#         super().__init__()
#         self.conv3_up = subpel_conv3x3(channel_in, channel_out, 2)
#         self.res_block3_up = ResBlock(channel_out)
#         self.conv3_out = nn.Conv2d(channel_out, channel_out, 3, padding=1)
#         self.res_block3_out = ResBlock(channel_out)
#         self.conv2_up = subpel_conv3x3(channel_out * 2, channel_out, 2)
#         self.res_block2_up = ResBlock(channel_out)
#         self.conv2_out = nn.Conv2d(channel_out * 2, channel_out, 3, padding=1)
#         self.res_block2_out = ResBlock(channel_out)
#         self.conv1_out = nn.Conv2d(channel_out * 2, channel_out, 3, padding=1)
#         self.res_block1_out = ResBlock(channel_out)

#     def forward(self, context1, context2, context3):
#         context3_up = self.conv3_up(context3)
#         context3_up = self.res_block3_up(context3_up)
#         context3_out = self.conv3_out(context3)
#         context3_out = self.res_block3_out(context3_out)
#         context2_up = self.conv2_up(torch.cat((context3_up, context2), dim=1))
#         context2_up = self.res_block2_up(context2_up)
#         context2_out = self.conv2_out(torch.cat((context3_up, context2), dim=1))
#         context2_out = self.res_block2_out(context2_out)
#         context1_out = self.conv1_out(torch.cat((context2_up, context1), dim=1))
#         context1_out = self.res_block1_out(context1_out)
#         context1 = context1 + context1_out
#         context2 = context2 + context2_out
#         context3 = context3 + context3_out
#         return context1, context2, context3


# class ContextualEncoder(nn.Module):
#     def __init__(self, channel_N=64, channel_M=96):
#         super().__init__()
#         self.conv1 = nn.Conv2d(channel_N + 3, channel_N, 3, stride=2, padding=1)
#         self.res1 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
#                              start_from_relu=True, end_with_relu=True)
#         self.conv2 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
#         self.res2 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
#                              start_from_relu=True, end_with_relu=True)
#         self.conv3 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
#         self.conv4 = nn.Conv2d(channel_N, channel_M, 3, stride=2, padding=1)

#     def forward(self, x, context1, context2, context3):
#         feature = self.conv1(torch.cat([x, context1], dim=1))
#         feature = self.res1(torch.cat([feature, context2], dim=1))
#         feature = self.conv2(feature)
#         feature = self.res2(torch.cat([feature, context3], dim=1))
#         feature = self.conv3(feature)
#         feature = self.conv4(feature)
#         return feature


# class ContextualDecoder(nn.Module):
#     def __init__(self, channel_N=64, channel_M=96):
#         super().__init__()
#         self.up1 = subpel_conv3x3(channel_M, channel_N, 2)
#         self.up2 = subpel_conv3x3(channel_N, channel_N, 2)
#         self.res1 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
#                              start_from_relu=True, end_with_relu=True)
#         self.up3 = subpel_conv3x3(channel_N * 2, channel_N, 2)
#         self.res2 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
#                              start_from_relu=True, end_with_relu=True)
#         self.up4 = subpel_conv3x3(channel_N * 2, 32, 2)

#     def forward(self, x, context2, context3):
#         feature = self.up1(x)
#         feature = self.up2(feature)
#         feature = self.res1(torch.cat([feature, context3], dim=1))
#         feature = self.up3(feature)
#         feature = self.res2(torch.cat([feature, context2], dim=1))
#         feature = self.up4(feature)
#         return feature


# class ReconGeneration(nn.Module):
#     def __init__(self, ctx_channel=64, res_channel=32, channel=64):
#         super().__init__()
#         self.first_conv = nn.Conv2d(ctx_channel + res_channel, channel, 3, stride=1, padding=1)
#         self.unet_1 = UNet(channel)
#         self.unet_2 = UNet(channel)
#         self.recon_conv = nn.Conv2d(channel, 3, 3, stride=1, padding=1)

#     def forward(self, ctx, res):
#         feature = self.first_conv(torch.cat((ctx, res), dim=1))
#         feature = self.unet_1(feature)
#         feature = self.unet_2(feature)
#         recon = self.recon_conv(feature)
#         return feature, recon