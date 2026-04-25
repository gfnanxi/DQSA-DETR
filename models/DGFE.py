# Modified from https://github.com/Jongchan/attention-module
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Conv_GN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 gn=True, bias=False):
        super(Conv_GN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.gn = nn.GroupNorm(32, out_channel)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv_BN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(Conv_BN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    


class DensityAwareChannelGate(nn.Module):
    """
    密度感知的动态通道注意力模块。
    输入：视觉特征 + 对应的密度图
    输出：通道注意力权重（与输入特征同 shape）
    """
    def __init__(self, gate_channels, reduction_ratio=16):
        super(DensityAwareChannelGate, self).__init__()
        self.gate_channels = gate_channels
        hidden_dim = gate_channels // reduction_ratio

        # 融合视觉上下文 (C) 和密度上下文 (1)
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, gate_channels),
            nn.Sigmoid()  # 注意：这里用 Sigmoid，因为是 scale
        )

    def forward(self, feat, density_map):
        """
        Args:
            feat: [B, C, H, W] - 视觉特征（已做空间增强）
            density_map: [B, 1, H, W] - 密度图（来自计数头）
        Returns:
            scale: [B, C, H, W] - 通道注意力权重
        """
        B, C, H, W = feat.shape

        # 1. 视觉上下文：全局平均池化
        z_v = F.adaptive_avg_pool2d(feat, 1).view(B, C)  # [B, C]

        # 2. 密度上下文：全局平均池化（得到整图密度强度）
        z_d = F.adaptive_avg_pool2d(density_map, 1).view(B, 1)  # [B, 1]

        # 3. 融合两种上下文
        z_fused = torch.cat([z_v, z_d], dim=1)  # [B, C + 1]

        # 4. 生成通道权重
        w_c = self.mlp(z_fused)  # [B, C]

        # 5. 扩展回空间维度以进行广播
        scale = w_c.view(B, C, 1, 1).expand_as(feat)  # [B, C, H, W]
        return scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = Conv_BN(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return scale


class DGFE(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False,
                 num_feature_levels=4):
        super(CGFE, self).__init__()
        self.num_feat = num_feature_levels
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.ChannelGate = DensityAwareChannelGate(gate_channels, reduction_ratio)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

        self.to_density = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1),nn.ReLU())  # 假设输入通道是256

    def forward(self, x, memory, spatial_shapes):
        feats = []
        idx = 0
        encoder_feat = memory.transpose(1, 2)
        bs, c, hw = encoder_feat.shape

        for i in range(self.num_feat):
            h, w = spatial_shapes[i]
            feat = encoder_feat[:, :, idx:idx + h * w].view(bs, 256, h, w)

            density_map = self.to_density(feat)  # [B, 1, H, W]

            c2 = self.SpatialGate(x[i])
            feat = feat * c2
            c1 = self.ChannelGate(feat, density_map)  # 传入 feat 和 density map x[i]
            feat = feat * c1
            feat = feat.flatten(2).transpose(1, 2)
            feats.append(feat)
            idx += h * w

        x_out = torch.cat(feats, 1)
        return x_out


class MultiScaleFeature(nn.Module):
    def __init__(self, channels=256, is_5_scale=False):
        super(MultiScaleFeature, self).__init__()
        self.conv1 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        if is_5_scale:
            self.conv4 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        self.is_5_scale = is_5_scale

    def forward(self, x):
        x_out = []
        x_out.append(x)
        x = self.conv1(x)
        x_out.append(x)
        x = self.conv2(x)
        x_out.append(x)
        x = self.conv3(x)
        x_out.append(x)

        if self.is_5_scale:
            x = self.conv4(x)
            x_out.append(x)
        return x_out

