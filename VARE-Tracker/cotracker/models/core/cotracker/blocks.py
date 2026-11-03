# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import math

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable, Optional
import collections
from itertools import repeat

from timm.layers import DropPath
from torch import Tensor

from cotracker.models.core.model_utils import bilinear_sampler



# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
            self,
            in_features,  # 输入特征的维度 (必须指定)
            hidden_features=None,  # 隐藏层的维度 (可选)
            out_features=None,  # 输出特征的维度 (可选)
            act_layer=nn.GELU,  # 激活函数层 (可选, 默认为GELU)
            norm_layer=None,  # 归一化层 (可选, 默认无)
            bias=True,  # 是否使用偏置 (可选)
            drop=0.0,  # Dropout的比率 (可选)
            use_conv=False,  # 是否使用1x1卷积代替全连接层
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        # print(f'x={x.shape}')
        x = self.fc1(x)
        # print(f'x={x.shape}')
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        # print(f'x={x.shape}')
        x = self.drop2(x)
        return x



#-------------------------------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            stride=stride,
            padding_mode="zeros",
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros"
        )
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, stride=4):
        super(BasicEncoder, self).__init__()
        self.stride = stride  # 最终输出特征图相对于原图的缩放步幅
        self.norm_fn = "instance"  # 写死了归一化方法为实例归一化 (Instance Normalization)
        self.in_planes = output_dim // 2  # 这是一个状态变量，用于在构建残差块时追踪输入的通道数

        # 定义两个独立的归一化层
        self.norm1 = nn.InstanceNorm2d(self.in_planes)
        self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        # 1. 初始卷积层 (Stem)
        # 这是一个大的7x7卷积，步幅为2，它将输入图像的分辨率减半 (H, W) -> (H/2, W/2)
        self.conv1 = nn.Conv2d(input_dim, self.in_planes, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        # 2. 四个阶段的残差块
        # 每个_make_layer包含两个残差块。
        self.layer1 = self._make_layer(output_dim // 2, stride=1)
        self.layer2 = self._make_layer(output_dim // 4 * 3, stride=2)
        self.layer3 = self._make_layer(output_dim, stride=2)
        self.layer4 = self._make_layer(output_dim, stride=2)

        # 3. 特征融合与最终处理
        # conv2的输入通道数是四个残差阶段输出通道数的总和，用于融合多尺度特征
        # (128//2) + (128//4*3) + 128 + 128 = 64 + 96 + 128 + 128 = 416
        # output_dim*3 + output_dim//4 = 128*3 + 32 = 384 + 32 = 416
        self.conv2 = nn.Conv2d(output_dim * 3 + output_dim // 4, output_dim * 2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        # conv3是一个1x1卷积，用于将融合后的特征通道数调整到最终的output_dim
        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)

        # 4. 权重初始化
        # 对卷积层使用Kaiming初始化（适用于ReLU激活函数），
        # 对归一化层的权重和偏置进行初始化，使其初始时等效于一个恒等变换。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        # 第一个残差块可以有步幅，用于下采样
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        # 第二个残差块步幅总是1
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        # 更新 self.in_planes 状态，以便下一个_make_layer知道其输入通道数
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, H, W = x.shape  # 获取输入图像的原始高和宽
        # print(f'输入图像的原始高和宽:({H},{W})')
        # --- 第1步: 初始特征提取 ---

        x = self.conv1(x)

        x = self.norm1(x)

        x = self.relu1(x)

        # 此时 x 的分辨率是 (H/2, W/2)
        # print(x.shape)
        # --- 第2步: 通过四个残差阶段，得到不同尺度的特征 ---
        a = self.layer1(x)  # a 的分辨率是 (H/2, W/2)
        # print(a.shape)
        b = self.layer2(a)  # b 的分辨率是 (H/4, W/4)
        # print(b.shape)
        c = self.layer3(b)  # c 的分辨率是 (H/8, W/8)
        # print(c.shape)
        d = self.layer4(c)  # d 的分辨率是 (H/16, W/16)

        # print(d.shape)
        # --- 第3步: 多尺度特征融合 ---
        # 定义一个内部辅助函数，用于将所有特征图插值到统一的分辨率
        def _bilinear_intepolate(x):
            return F.interpolate(
                x,
                (H // self.stride, W // self.stride),  # 目标分辨率，例如 stride=4 时为 (H/4, W/4)
                mode="bilinear",
                align_corners=True,
            )

        # 将 a, b, c, d 都调整到 (H/4, W/4) 分辨率
        a = _bilinear_intepolate(a)  # 下采样
        b = _bilinear_intepolate(b)  # 保持不变
        c = _bilinear_intepolate(c)  # 上采样
        d = _bilinear_intepolate(d)  # 上采样

        # --- 第4步: 合并与最终处理 ---
        # 沿通道维度拼接所有尺度的特征
        x = self.conv2(torch.cat([a, b, c, d], dim=1))
        x = self.norm2(x)
        x = self.relu2(x)
        # 使用1x1卷积进行降维，得到最终的输出特征
        x = self.conv3(x)
        return x

#--------------------------------------------------------------------------------------
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h], indexing='ij')).transpose(1, 2).repeat(1, self.groups, 1).reshape(1,
                                                                                                                    -1,
                                                                                                                    1,
                                                                                                                    1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing='ij')
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'lp':
            return self.forward_lp(x)
        return self.forward_pl(x)


class TrackingFriendlyRoutingBlock(nn.Module):
    """
    Tracking-Friendly Routing Block (TFRB)
    核心亮点:
    1. Spatial-Aware Mask: DW3x3 捕捉空间边界 + 1x1 决策特征归属，使得 Mask 在物体边缘更锐利。
    2. Strict Metric-Preserving: 对特征的提纯严格使用 1x1 结构，杜绝特征融合时的跨像素污染，保护亚像素锐度。
    """

    def __init__(self, dim, norm="instance"):
        super().__init__()
        NormLayer = nn.InstanceNorm2d if norm == "instance" else nn.BatchNorm2d

        # 1. 空间感知的上下文路由器 (Spatial-Aware Context Router)
        self.router = nn.Sequential(
            # 分组数为 dim*2，即纯 Depthwise 操作，获取空间上下文但不融合通道
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=False),
            NormLayer(dim * 2),
            nn.GELU(),
            # 使用 1x1 Pointwise 卷积进行通道混合，并压缩为 1 个通道的空间门控
            nn.Conv2d(dim * 2, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 2. 严格度量保护的特征提纯器 (Strict Metric-Preserving Refiners)
        self.refine_x = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            NormLayer(dim),
            nn.GELU()
        )
        self.refine_guide = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            NormLayer(dim),
            nn.GELU()
        )

    def forward(self, x, guide):
        # 1. 生成空间感知门控：0 代表保留浅层物理特征，1 代表注入深层语义
        ctx = torch.cat([x, guide], dim=1)
        mask = self.router(ctx)  # 维度: [B, 1, H, W]

        # 2. 独立度量等距提纯
        r_x = self.refine_x(x)
        r_guide = self.refine_guide(guide)

        # 3. 软插值路由 (语义无缝注入)
        return r_x * (1 - mask) + r_guide * mask


# ==========================================
# [核心架构 1] 语义路由融合网络 (FPN)
# ==========================================
class SemanticRoutingFusion(nn.Module):
    """
    Semantic Routing Fusion Network
    适用于轻量化点跟踪的特征颈部 (Neck) 架构
    """

    def __init__(self, in_channels_list=[64, 96, 128, 256], out_dim=128, norm="instance"):
        super().__init__()
        NormLayer = nn.InstanceNorm2d if norm == "instance" else nn.BatchNorm2d

        # 1. 通道统一映射
        self.proj4 = nn.Conv2d(in_channels_list[3], out_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(in_channels_list[2], out_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(in_channels_list[1], out_dim, kernel_size=1)

        # 2. 宏观几何对齐
        self.up4_to_3 = DySample(in_channels=out_dim, scale=2, style='lp', groups=4)
        self.up3_to_2 = DySample(in_channels=out_dim, scale=2, style='lp', groups=4)

        # 3. 追踪级语义路由主干
        self.fusion4_to_3 = TrackingFriendlyRoutingBlock(out_dim, norm)
        self.fusion3_to_2 = TrackingFriendlyRoutingBlock(out_dim, norm)

        # 4. 微观相位校准流 (L1 -> L2)
        self.down1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels_list[0], out_dim, kernel_size=1, bias=False),
            NormLayer(out_dim),
            nn.GELU()
        )
        self.l1_gate = nn.Conv2d(out_dim, out_dim, kernel_size=1)

        # 5. 最终特征输出 (坚持度量保护叙事)
        self.out_proj = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim, bias=False),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            NormLayer(out_dim, affine=True),
            nn.GELU()
        )

    def forward(self, feats):
        l1, l2, l3, l4 = feats

        # 阶段 1：独立特征投影
        src4 = self.proj4(l4)
        src3 = self.proj3(l3)
        src2 = self.proj2(l2)
        src1 = self.down1(l1)

        # 阶段 2：主干流 (自顶向下的上下文路由)
        up4 = self.up4_to_3(src4)
        p3 = self.fusion4_to_3(x=src3, guide=up4)

        up3 = self.up3_to_2(p3)
        p2 = self.fusion3_to_2(x=src2, guide=up3)

        # 阶段 3：细节流 (利用 p2 安检门隔离 L1 背景噪声)
        gate1 = torch.sigmoid(self.l1_gate(p2))
        fused = p2 + src1 * gate1

        # 阶段 4：受保护的度量输出
        return self.out_proj(fused)


class GuideRoutingBlock(nn.Module):
    """Guide-only semantic residual routing used by the recovered MR-FPN.

    The shallow feature is kept as the geometric anchor. The high-level guide
    only contributes a masked residual, which is safer for point localization
    than replacing or re-normalizing the shallow feature.
    """

    def __init__(self, dim=128):
        super().__init__()
        self.router = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True),
            nn.Conv2d(dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.semantic_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x, guide):
        mask = self.router(guide)
        return x + mask * self.semantic_proj(guide)


class RecoveredGuideRoutingFPN(nn.Module):
    """Recovered DySample + guide-routing FPN.

    This module intentionally matches the old checkpoint key layout:
    fpn.proj{2,3,4}, fpn.down1.1, fpn.up{4,3}_to_{3,2},
    and fpn.fusion*_to_*.{router,semantic_proj}.
    """

    def __init__(self, in_channels_list=(64, 96, 128, 128), out_dim=128):
        super().__init__()
        self.proj4 = nn.Conv2d(in_channels_list[3], out_dim, kernel_size=1, bias=True)
        self.proj3 = nn.Conv2d(in_channels_list[2], out_dim, kernel_size=1, bias=True)
        self.proj2 = nn.Conv2d(in_channels_list[1], out_dim, kernel_size=1, bias=True)

        self.down1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels_list[0], out_dim, kernel_size=1, bias=True),
        )

        self.up4_to_3 = DySample(in_channels=out_dim, scale=2, style="lp", groups=4)
        self.up3_to_2 = DySample(in_channels=out_dim, scale=2, style="lp", groups=4)

        self.fusion4_to_3 = GuideRoutingBlock(out_dim)
        self.fusion3_to_2 = GuideRoutingBlock(out_dim)

    def forward(self, feats):
        l1, l2, l3, l4 = feats

        src4 = self.proj4(l4)
        src3 = self.proj3(l3)
        src2 = self.proj2(l2)
        src1 = self.down1(l1)

        up4 = self.up4_to_3(src4)
        p3 = self.fusion4_to_3(src3, up4)

        up3 = self.up3_to_2(p3)
        p2 = self.fusion3_to_2(src2, up3)

        return p2 + src1


class ConservativeGuideRoutingBlock(nn.Module):
    """Low-amplitude guide residual for descriptor-preserving fusion.

    Unlike a conventional FPN fusion block, this module treats the shallow
    descriptor as the metric anchor and injects guide information only through a
    learnable residual scale. The scale is initialized near zero so the module
    starts from an identity mapping and learns how much context is safe to add.
    """

    def __init__(self, dim=128, init_scale=0.0):
        super().__init__()
        self.router = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True),
            nn.Conv2d(dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.semantic_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), float(init_scale)))

    def forward(self, x, guide):
        mask = self.router(guide)
        return x + self.gamma * mask * self.semantic_proj(guide)


class ConservativeGuideRoutingFPN(nn.Module):
    """Conservative metric-preserving routing FPN.

    This A-compatible B variant keeps the H/4 descriptor as the primary metric
    space. High-level context and shallow-detail anchors are injected only as
    gated, learnable low-amplitude residuals to avoid over-smoothing or
    re-distributing descriptors before frozen-template correlation.
    """

    def __init__(
        self,
        in_channels_list=(64, 96, 128, 128),
        out_dim=128,
        use_dysample=False,
        init_high_scale=0.0,
        init_low_scale=0.0,
    ):
        super().__init__()
        self.use_dysample = use_dysample
        self.proj4 = nn.Conv2d(in_channels_list[3], out_dim, kernel_size=1, bias=True)
        self.proj3 = nn.Conv2d(in_channels_list[2], out_dim, kernel_size=1, bias=True)
        self.proj2 = nn.Conv2d(in_channels_list[1], out_dim, kernel_size=1, bias=True)

        self.down1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels_list[0], out_dim, kernel_size=1, bias=True),
        )

        if use_dysample:
            self.up4_to_3 = DySample(in_channels=out_dim, scale=2, style="lp", groups=4)
            self.up3_to_2 = DySample(in_channels=out_dim, scale=2, style="lp", groups=4)
        else:
            self.up4_to_3 = None
            self.up3_to_2 = None

        self.fusion4_to_3 = ConservativeGuideRoutingBlock(
            out_dim, init_scale=init_high_scale
        )
        self.fusion3_to_2 = ConservativeGuideRoutingBlock(
            out_dim, init_scale=init_high_scale
        )
        self.low_router = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim, bias=True),
            nn.Conv2d(out_dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.low_gamma = nn.Parameter(
            torch.full((1, out_dim, 1, 1), float(init_low_scale))
        )

    def _upsample(self, x, target_size, sampler):
        if sampler is not None:
            return sampler(x)
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)

    def forward(self, feats):
        l1, l2, l3, l4 = feats

        src4 = self.proj4(l4)
        src3 = self.proj3(l3)
        src2 = self.proj2(l2)
        src1 = self.down1(l1)

        up4 = self._upsample(src4, src3.shape[-2:], self.up4_to_3)
        p3 = self.fusion4_to_3(src3, up4)

        up3 = self._upsample(p3, src2.shape[-2:], self.up3_to_2)
        p2 = self.fusion3_to_2(src2, up3)

        low_mask = self.low_router(src1)
        return p2 + self.low_gamma * low_mask * src1


def conv_dw_pw(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
    """
    深度可分离卷积封装 (Depthwise Separable Convolution)
    用于在特征提取阶段实现极致轻量化。
    """
    return nn.Sequential(
        # Depthwise: 空间特征提取
        nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=False
        ),
        nn.InstanceNorm2d(in_channels, affine=True),
        # Pointwise: 通道特征混合
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )


class MotionAwarePConv(nn.Module):
    """
    创新点一：深度可分离版 MAPConv (Motion-Aware PConv)
    1. Active Branch: 使用深度卷积 (groups=C) 提取特征。
    2. Passive Branch: 十字平移 (Cross-Shift)，0 参数增强空间感知。
    """

    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv

        c_fine = self.dim_conv // 2
        c_coarse = self.dim_conv - c_fine

        # 内部改为 Depthwise 卷积，与 Block 后的 1x1 卷积组合成 DS-Conv 逻辑
        self.conv_fine = nn.Conv2d(
            c_fine, c_fine, 3, 1, 1, groups=c_fine, bias=False
        )
        self.conv_coarse = nn.Conv2d(
            c_coarse, c_coarse, 3, 1, 2, dilation=2, groups=c_coarse, bias=False
        )

    def forward(self, x):
        # 1. 核心通道分离
        x_conv, x_untouched = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)

        # 2. Active: 异构多尺度提取 (Depthwise)
        x_fine = self.conv_fine(x_conv[:, :self.dim_conv // 2, :, :])
        x_coarse = self.conv_coarse(x_conv[:, self.dim_conv // 2:, :, :])

        # 3. Passive: 十字平移 (利用边缘复制，白嫖空间感知)
        C_un = self.dim_untouched
        chunk = C_un // 5
        splits = torch.split(x_untouched, [chunk, chunk, chunk, chunk, C_un - 4 * chunk], dim=1)

        x_up = torch.cat([splits[0][:, :, 1:, :], splits[0][:, :, -1:, :]], dim=2)
        x_down = torch.cat([splits[1][:, :, :1, :], splits[1][:, :, :-1, :]], dim=2)
        x_left = torch.cat([splits[2][:, :, :, 1:], splits[2][:, :, :, -1:]], dim=3)
        x_right = torch.cat([splits[3][:, :, :, :1], splits[3][:, :, :, :-1]], dim=3)
        x_center = splits[4]

        return torch.cat((x_fine, x_coarse, x_up, x_down, x_left, x_right, x_center), dim=1)


class MotionAwareBlock(nn.Module):
    """
    轻量化模块：PConv (Depthwise) + MLP (Pointwise)
    """

    def __init__(self, dim, n_div=4, mlp_ratio=1.0):
        super().__init__()
        self.spatial_mixing = MotionAwarePConv(dim, n_div)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.InstanceNorm2d(hidden_dim, affine=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return x + self.mlp(self.spatial_mixing(x))


class AlignedFasterEncoder(nn.Module):
    """
    算子对比版 Encoder：
    1. 提取阶段 (Before Fusion): 使用深度可分离卷积追求极致轻量。
    2. 融合阶段 (Fusion Part): 保持标准 3x3 卷积，严格对齐原版融合压力。
    """

    def __init__(self, input_dim=3, output_dim=128):
        super().__init__()
        embed_dims = [64, 96, 128, 128]

        # Stem 层：首层保留标准卷积，第二层改用深度可分离卷积
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            conv_dw_pw(32, embed_dims[0], kernel_size=3, padding=1)
        )

        self.stage1 = self._make_stage(embed_dims[0], embed_dims[0], stride=1, depth=1)
        self.stage2 = self._make_stage(embed_dims[0], embed_dims[1], stride=2, depth=2)
        self.stage3 = self._make_stage(embed_dims[1], embed_dims[2], stride=2, depth=4)
        self.stage4 = self._make_stage(embed_dims[2], embed_dims[3], stride=2, depth=2)

        # 核心融合层：保持标准卷积 (Standard Cat)
        # 这里不改深度可分离，是为了证明即便在沉重的 Cat 压力下，我们的 Backbone 依然更快
        sum_dims = sum(embed_dims)  # 416 维
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(sum_dims, output_dim * 2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(output_dim * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim * 2, output_dim, kernel_size=1),
        )

    def _make_stage(self, in_dim, out_dim, stride, depth):
        layers = []
        if stride == 2 or in_dim != out_dim:
            # 下采样层：也改用深度可分离卷积以降低开销
            layers.append(conv_dw_pw(in_dim, out_dim, kernel_size=stride, stride=stride))
        else:
            out_dim = in_dim

        for _ in range(depth):
            layers.append(MotionAwareBlock(out_dim, n_div=4, mlp_ratio=1.0))

        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.stem(x)
        a = self.stage1(x)
        b = self.stage2(a)
        c = self.stage3(b)
        d = self.stage4(c)

        # 统一尺度
        a = F.interpolate(a, (H // 4, W // 4), mode="bilinear", align_corners=True)
        c = F.interpolate(c, (H // 4, W // 4), mode="bilinear", align_corners=True)
        d = F.interpolate(d, (H // 4, W // 4), mode="bilinear", align_corners=True)

        fused_features = torch.cat([a, b, c, d], dim=1)
        return self.fusion_conv(fused_features)

# =====================================================================================
def build_norm(num_channels, norm_type="in", num_groups=8):
    norm_type = norm_type.lower()
    if norm_type == "bn":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm_type == "gn":
        groups = min(num_groups, num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    return nn.InstanceNorm2d(num_channels, affine=True)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm_type="in"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            build_norm(out_channels, norm_type),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ChannelSE(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden_dim = max(dim // reduction, 8)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attn(x)


class KineticDifferentialPConv(nn.Module):
    """
    去掉 shift 后的版本：
    1. 一部分通道做 3x3 depthwise
    2. 一部分通道做 3x3 dilated depthwise
    3. 剩余通道保留原始细节
    4. 通过 1x1 混合和 SE 重标定完成融合
    """
    def __init__(self, dim, n_div=2, norm_type="in"):
        super().__init__()
        self.dim = dim
        self.dim_conv = max(dim // n_div, 1)
        self.dim_id = dim - self.dim_conv

        c_fine = max(self.dim_conv // 2, 1)
        c_coarse = self.dim_conv - c_fine
        self.c_fine = c_fine
        self.c_coarse = c_coarse

        self.conv_fine = nn.Conv2d(
            c_fine, c_fine, kernel_size=3, stride=1, padding=1,
            groups=c_fine, bias=False
        )

        self.conv_coarse = None
        if c_coarse > 0:
            self.conv_coarse = nn.Conv2d(
                c_coarse, c_coarse, kernel_size=3, stride=1, padding=2,
                dilation=2, groups=c_coarse, bias=False
            )

        self.channel_mixer = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            build_norm(dim, norm_type),
            nn.ReLU(inplace=True)
        )

        self.recalibration = ChannelSE(dim)

    def forward(self, x):
        if self.dim_id > 0:
            x_conv, x_id = torch.split(x, [self.dim_conv, self.dim_id], dim=1)
        else:
            x_conv = x
            x_id = None

        x_fine = self.conv_fine(x_conv[:, :self.c_fine, :, :])

        outs = [x_fine]

        if self.conv_coarse is not None:
            x_coarse = self.conv_coarse(x_conv[:, self.c_fine:, :, :])
            outs.append(x_coarse)

        if x_id is not None:
            outs.append(x_id)

        out = torch.cat(outs, dim=1)
        out = self.channel_mixer(out)
        out = self.recalibration(out)
        return out


class KineticFeatureBlock(nn.Module):
    def __init__(self, dim, n_div=2, mlp_ratio=1.0, norm_type="in"):
        super().__init__()
        self.spatial = KineticDifferentialPConv(dim, n_div=n_div, norm_type=norm_type)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            build_norm(hidden_dim, norm_type),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)
        )

        self.scale = nn.Parameter(1e-2 * torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return x + self.scale * self.mlp(self.spatial(x))


class GeometryGatedFusionBlock(nn.Module):
    def __init__(self, dim, norm_type="in"):
        super().__init__()
        self.router = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=False),
            build_norm(dim, norm_type),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(1e-2 * torch.ones(1, dim, 1, 1))

    def forward(self, x, guide):
        mask = self.router(torch.cat([x, guide], dim=1))
        return x + self.gamma * mask * self.proj(guide)


class FusionRefineBlock(nn.Module):
    def __init__(self, dim, norm_type="in"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            build_norm(dim, norm_type),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class GeometryAwareRoutingFPN(nn.Module):
    def __init__(self, in_dims, out_dim=128, norm_type="in"):
        super().__init__()
        self.proj2 = nn.Conv2d(in_dims[1], out_dim, kernel_size=1, bias=False)
        self.proj3 = nn.Conv2d(in_dims[2], out_dim, kernel_size=1, bias=False)
        self.proj4 = nn.Conv2d(in_dims[3], out_dim, kernel_size=1, bias=False)

        self.anchor = nn.Sequential(
            nn.Conv2d(in_dims[0], out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            build_norm(out_dim, norm_type),
            nn.ReLU(inplace=True)
        )

        self.gate3 = GeometryGatedFusionBlock(out_dim, norm_type=norm_type)
        self.gate2 = GeometryGatedFusionBlock(out_dim, norm_type=norm_type)

        self.refine3 = FusionRefineBlock(out_dim, norm_type=norm_type)
        self.refine2 = FusionRefineBlock(out_dim, norm_type=norm_type)

    def forward(self, feats):
        l1, l2, l3, l4 = feats

        s2 = self.proj2(l2)
        s3 = self.proj3(l3)
        s4 = self.proj4(l4)

        up4 = F.interpolate(s4, size=s3.shape[-2:], mode="bilinear", align_corners=True)
        f3 = self.refine3(self.gate3(s3, up4))

        up3 = F.interpolate(f3, size=s2.shape[-2:], mode="bilinear", align_corners=True)
        f2 = self.refine2(self.gate2(s2, up3))

        return f2 + self.anchor(l1)


class HighFrequencyGeometryResidual(nn.Module):
    """Preserve local geometric details for rigid point tracking.

    The residual is a cheap high-pass response: x - avg_pool(x). It keeps
    edge/corner cues that are important for stable point localization.
    """

    def __init__(self, dim, init_scale=0.05):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), init_scale))

    def forward(self, x):
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x + self.gamma * (x - low)


class RigidGeometryPreservingBlock(nn.Module):
    """Geometry-preserved motion block for rigid point tracking.

    The block has two conceptual steps:
    1. high-frequency geometry enhancement (HFG);
    2. partial fine-to-coarse motion mixing (PMM).

    The passive channels are only an implementation detail of partial mixing,
    not a separate branch in the method story.
    """

    def __init__(
        self,
        dim,
        active_ratio=0.50,
        mlp_ratio=1.0,
        norm_type="in",
        residual_init=1e-2,
    ):
        super().__init__()
        active_dim = max(2, int(dim * active_ratio))
        active_dim = min(active_dim, dim)
        passive_dim = dim - active_dim

        fine_dim = max(1, active_dim // 2)
        coarse_dim = active_dim - fine_dim

        self.active_dim = active_dim
        self.passive_dim = passive_dim
        self.fine_dim = fine_dim
        self.coarse_dim = coarse_dim

        self.geometry_enhance = HighFrequencyGeometryResidual(dim)
        self.fine_dw = nn.Conv2d(
            fine_dim, fine_dim, kernel_size=3, padding=1, groups=fine_dim, bias=False
        )
        self.coarse_dw = None
        if coarse_dim > 0:
            self.coarse_dw = nn.Conv2d(
                coarse_dim,
                coarse_dim,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=coarse_dim,
                bias=False,
            )

        hidden_dim = int(dim * mlp_ratio)
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            build_norm(hidden_dim, norm_type),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
        )
        self.scale = nn.Parameter(residual_init * torch.ones(1, dim, 1, 1))

    def forward(self, x):
        # HFG: preserve edges/corners before lightweight spatial mixing.
        x_geo = self.geometry_enhance(x)

        if self.passive_dim > 0:
            x_active, x_passive = torch.split(
                x_geo, [self.active_dim, self.passive_dim], dim=1
            )
        else:
            x_active = x_geo
            x_passive = None

        # PMM: only active channels perform fine/coarse depthwise mixing.
        x_fine = x_active[:, : self.fine_dim]
        y_motion = [self.fine_dw(x_fine)]
        if self.coarse_dim > 0:
            x_coarse = x_active[:, self.fine_dim :]
            y_motion.append(self.coarse_dw(x_coarse))

        out = [torch.cat(y_motion, dim=1)]
        if x_passive is not None:
            out.append(x_passive)

        mixed = self.channel_mixer(torch.cat(out, dim=1))
        return x + self.scale * mixed


class RigidGeometryPreservingEncoder(nn.Module):
    """Rigid-aware geometry-preserving lightweight encoder.

    It is designed for point tracking rather than classification: compact
    high-level channels, partial fine/coarse spatial mixing, and explicit
    high-frequency geometry preservation.
    """

    def __init__(
        self,
        input_dim=3,
        output_dim=128,
        stride=4,
        dims=None,
        depths=None,
        fusion_type="cat",
        norm_type="in",
    ):
        super().__init__()
        if dims is None:
            dims = [64, 96, 128, 128]
        if depths is None:
            depths = [2, 2, 3, 1]
        assert len(dims) == 4 and len(depths) == 4
        assert fusion_type in ("cat", "routing", "none")

        self.stride = stride
        self.dims = dims
        self.depths = depths
        self.fusion_type = fusion_type
        self.norm_type = norm_type

        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=2, padding=1, bias=False),
            build_norm(32, norm_type),
            nn.ReLU(inplace=True),
            ConvNormAct(32, dims[0], kernel_size=3, stride=1, padding=1, norm_type=norm_type),
        )

        self.stage1 = self._make_stage(dims[0], dims[0], depths[0], stride=1)
        self.stage2 = self._make_stage(dims[0], dims[1], depths[1], stride=2)
        self.stage3 = self._make_stage(dims[1], dims[2], depths[2], stride=2)
        self.stage4 = self._make_stage(dims[2], dims[3], depths[3], stride=2)

        if fusion_type == "cat":
            self.fusion = nn.Sequential(
                nn.Conv2d(sum(dims), output_dim * 2, kernel_size=1, bias=False),
                build_norm(output_dim * 2, norm_type),
                nn.GELU(),
                nn.Conv2d(output_dim * 2, output_dim, kernel_size=1, bias=True),
            )
        elif fusion_type == "routing":
            self.fusion = RecoveredGuideRoutingFPN(dims, out_dim=output_dim)
        else:
            self.fusion = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_stage(self, in_dim, out_dim, depth, stride=1):
        layers = []
        if stride == 2 or in_dim != out_dim:
            layers.append(
                ConvNormAct(
                    in_dim,
                    out_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    norm_type=self.norm_type,
                )
            )
        for _ in range(depth):
            layers.append(
                RigidGeometryPreservingBlock(
                    out_dim,
                    active_ratio=0.50,
                    mlp_ratio=1.0,
                    norm_type=self.norm_type,
                )
            )
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]

    def forward(self, x):
        _, _, H, W = x.shape
        feats = self.forward_features(x)
        if self.fusion_type == "none":
            return feats
        if self.fusion_type == "routing":
            return self.fusion(feats)

        target_size = (H // self.stride, W // self.stride)
        aligned = [
            F.interpolate(feat, target_size, mode="bilinear", align_corners=True)
            if feat.shape[-2:] != target_size
            else feat
            for feat in feats
        ]
        return self.fusion(torch.cat(aligned, dim=1))


class AntiAliasDownsample(nn.Module):
    """Low-alias downsampling for point-level descriptor maps."""

    def __init__(self, in_dim, out_dim, norm_type="in"):
        super().__init__()
        self.block = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            build_norm(out_dim, norm_type),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class RigidDescriptorBlock(nn.Module):
    """Local-rigid-geometry block for online point tracking.

    The block explicitly encodes center-neighbor feature differences before the
    lightweight spatial/channel mixing. For rigid scenes, these local relative
    geometry cues are more stable than isolated point appearance.
    """

    def __init__(
        self,
        dim,
        kernel_size=3,
        mlp_ratio=1.0,
        norm_type="in",
        residual_init=1e-2,
        use_geometry=True,
    ):
        super().__init__()
        padding = kernel_size // 2
        hidden_dim = int(dim * mlp_ratio)
        self.use_geometry = use_geometry

        if use_geometry:
            self.geometry_proj = nn.Sequential(
                nn.Conv2d(dim * 4, dim, kernel_size=1, bias=False),
                build_norm(dim, norm_type),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=padding,
                groups=dim,
                bias=False,
            ),
            build_norm(dim, norm_type),
            nn.GELU(),
        )
        self.channel = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            build_norm(hidden_dim, norm_type),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
        )
        self.scale = nn.Parameter(residual_init * torch.ones(1, dim, 1, 1))
        if use_geometry:
            self.geometry_scale = nn.Parameter(torch.zeros(1, dim, 1, 1))

    @staticmethod
    def _shift_up(x):
        return torch.cat([x[:, :, 1:, :], x[:, :, -1:, :]], dim=2)

    @staticmethod
    def _shift_down(x):
        return torch.cat([x[:, :, :1, :], x[:, :, :-1, :]], dim=2)

    @staticmethod
    def _shift_left(x):
        return torch.cat([x[:, :, :, 1:], x[:, :, :, -1:]], dim=3)

    @staticmethod
    def _shift_right(x):
        return torch.cat([x[:, :, :, :1], x[:, :, :, :-1]], dim=3)

    def local_geometry(self, x):
        diffs = torch.cat(
            [
                x - self._shift_up(x),
                x - self._shift_down(x),
                x - self._shift_left(x),
                x - self._shift_right(x),
            ],
            dim=1,
        )
        return self.geometry_proj(diffs)

    def forward(self, x):
        if self.use_geometry:
            x = x + self.geometry_scale * self.local_geometry(x)
        return x + self.scale * self.channel(self.spatial(x))


class RigidDescriptorEncoder(nn.Module):
    """Local Rigid Geometry Encoder.

    This encoder is designed around the role of CoTracker's front-end: producing
    stable, interpolatable descriptors for cost-volume matching. It keeps
    all-channel local spatial modeling, explicitly encodes center-neighbor local
    geometry, and allocates larger receptive fields only to low-resolution stages.
    """

    def __init__(
        self,
        input_dim=3,
        output_dim=128,
        stride=4,
        dims=None,
        depths=None,
        kernels=None,
        geometry_stages=None,
        norm_type="in",
    ):
        super().__init__()
        if dims is None:
            dims = [64, 96, 128, 128]
        if depths is None:
            depths = [1, 2, 3, 1]
        if kernels is None:
            kernels = [3, 3, 5, 5]
        if geometry_stages is None:
            geometry_stages = [False, False, True, True]
        assert len(dims) == 4 and len(depths) == 4 and len(kernels) == 4
        assert len(geometry_stages) == 4

        self.stride = stride
        self.dims = dims
        self.depths = depths
        self.kernels = kernels
        self.geometry_stages = geometry_stages
        self.norm_type = norm_type

        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=2, padding=1, bias=False),
            build_norm(32, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, dims[0], kernel_size=3, stride=1, padding=1, bias=False),
            build_norm(dims[0], norm_type),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(dims[0], depths[0], kernels[0], geometry_stages[0])
        self.down2 = AntiAliasDownsample(dims[0], dims[1], norm_type=norm_type)
        self.stage2 = self._make_stage(dims[1], depths[1], kernels[1], geometry_stages[1])
        self.down3 = AntiAliasDownsample(dims[1], dims[2], norm_type=norm_type)
        self.stage3 = self._make_stage(dims[2], depths[2], kernels[2], geometry_stages[2])
        self.down4 = AntiAliasDownsample(dims[2], dims[3], norm_type=norm_type)
        self.stage4 = self._make_stage(dims[3], depths[3], kernels[3], geometry_stages[3])

        self.conv2 = nn.Conv2d(sum(dims), output_dim * 2, kernel_size=3, padding=1)
        self.norm_fuse = build_norm(output_dim * 2, norm_type)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_stage(self, dim, depth, kernel_size, use_geometry):
        return nn.Sequential(
            *[
                RigidDescriptorBlock(
                    dim,
                    kernel_size=kernel_size,
                    mlp_ratio=1.0,
                    norm_type=self.norm_type,
                    use_geometry=use_geometry,
                )
                for _ in range(depth)
            ]
        )

    def forward_features(self, x):
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(self.down2(f1))
        f3 = self.stage3(self.down3(f2))
        f4 = self.stage4(self.down4(f3))
        return [f1, f2, f3, f4]

    def forward(self, x):
        _, _, H, W = x.shape
        feats = self.forward_features(x)

        target_size = (H // self.stride, W // self.stride)
        aligned = [
            F.interpolate(feat, target_size, mode="bilinear", align_corners=True)
            if feat.shape[-2:] != target_size
            else feat
            for feat in feats
        ]

        x = self.conv2(torch.cat(aligned, dim=1))
        x = self.norm_fuse(x)
        x = self.relu2(x)
        return self.conv3(x)


class RigidDescriptorMRFPNEncoder(RigidDescriptorEncoder):
    """LRG-Encoder backbone combined with MR-FPN fusion.

    The inherited RigidDescriptorEncoder extracts four multi-scale descriptor
    maps. This variant replaces the default concatenation fusion with the
    recovered metric-aware routing FPN used by innovation B.
    """

    def __init__(
        self,
        input_dim=3,
        output_dim=128,
        stride=4,
        dims=None,
        depths=None,
        kernels=None,
        geometry_stages=None,
        norm_type="in",
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            stride=stride,
            dims=dims,
            depths=depths,
            kernels=kernels,
            geometry_stages=geometry_stages,
            norm_type=norm_type,
        )
        self.mrfpn = RecoveredGuideRoutingFPN(self.dims, out_dim=output_dim)

    def forward(self, x):
        feats = self.forward_features(x)
        return self.mrfpn(feats)


# Paper-facing alias for the full A+B front-end.
LRGMRFPNEncoder = RigidDescriptorMRFPNEncoder


class StandaloneLRGMRFPNEncoder(nn.Module):
    """Standalone LRG backbone with MR-FPN fusion.

    This class does not inherit from RigidDescriptorEncoder. It contains an
    independent LRG feature extractor and the recovered metric-aware routing FPN
    in one module, which makes it suitable as a clean A+B training target.
    """

    def __init__(
        self,
        input_dim=3,
        output_dim=128,
        stride=4,
        dims=None,
        depths=None,
        kernels=None,
        geometry_stages=None,
        norm_type="in",
    ):
        super().__init__()
        if dims is None:
            dims = [64, 96, 128, 128]
        if depths is None:
            depths = [1, 2, 3, 1]
        if kernels is None:
            kernels = [3, 3, 5, 5]
        if geometry_stages is None:
            geometry_stages = [False, False, True, True]
        assert len(dims) == 4 and len(depths) == 4 and len(kernels) == 4
        assert len(geometry_stages) == 4

        self.stride = stride
        self.dims = dims
        self.depths = depths
        self.kernels = kernels
        self.geometry_stages = geometry_stages
        self.norm_type = norm_type

        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=2, padding=1, bias=False),
            build_norm(32, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, dims[0], kernel_size=3, stride=1, padding=1, bias=False),
            build_norm(dims[0], norm_type),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(dims[0], depths[0], kernels[0], geometry_stages[0])
        self.down2 = AntiAliasDownsample(dims[0], dims[1], norm_type=norm_type)
        self.stage2 = self._make_stage(dims[1], depths[1], kernels[1], geometry_stages[1])
        self.down3 = AntiAliasDownsample(dims[1], dims[2], norm_type=norm_type)
        self.stage3 = self._make_stage(dims[2], depths[2], kernels[2], geometry_stages[2])
        self.down4 = AntiAliasDownsample(dims[2], dims[3], norm_type=norm_type)
        self.stage4 = self._make_stage(dims[3], depths[3], kernels[3], geometry_stages[3])

        self.fpn = RecoveredGuideRoutingFPN(dims, out_dim=output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_stage(self, dim, depth, kernel_size, use_geometry):
        return nn.Sequential(
            *[
                RigidDescriptorBlock(
                    dim,
                    kernel_size=kernel_size,
                    mlp_ratio=1.0,
                    norm_type=self.norm_type,
                    use_geometry=use_geometry,
                )
                for _ in range(depth)
            ]
        )

    def forward_features(self, x):
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(self.down2(f1))
        f3 = self.stage3(self.down3(f2))
        f4 = self.stage4(self.down4(f3))
        return [f1, f2, f3, f4]

    def forward(self, x):
        return self.fpn(self.forward_features(x))



class KineticLightweightEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, dims=None, norm_type="in"):
        super().__init__()
        if dims is None:
            dims = [64, 96, 128, 128]

        self.norm_type = norm_type
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=7, stride=2, padding=3, bias=False),
            build_norm(32, norm_type),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(32, dims[0], depth=1, stride=1)
        self.stage2 = self._make_stage(dims[0], dims[1], depth=2, stride=2)
        self.stage3 = self._make_stage(dims[1], dims[2], depth=4, stride=2)
        self.stage4 = self._make_stage(dims[2], dims[3], depth=2, stride=2)

        self.fusion = GeometryAwareRoutingFPN(dims, out_dim=output_dim, norm_type=norm_type)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_stage(self, in_d, out_d, depth, stride=1):
        layers = []

        if stride == 2 or in_d != out_d:
            layers.append(
                ConvNormAct(
                    in_d, out_d,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    norm_type=self.norm_type
                )
            )
        else:
            out_d = in_d

        for _ in range(depth):
            layers.append(
                KineticFeatureBlock(
                    out_d,
                    n_div=2,
                    mlp_ratio=1.0,
                    norm_type=self.norm_type
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, H, W = x.shape

        x = self.stem(x)
        f1 = self.stage1(x)   # H/2
        f2 = self.stage2(f1)  # H/4
        f3 = self.stage3(f2)  # H/8
        f4 = self.stage4(f3)  # H/16

        return self.fusion([f1, f2, f3, f4])
# -----------------------------------------------------------------------------------------------------------------

class KineticEncoderWithCatFusion(nn.Module):
    """
    消融实验版本：
    - 特征提取：使用创新点一（KineticFeatureBlock）
    - 特征融合：使用原始 BasicEncoder 的 cat 融合
    """
    def __init__(self, input_dim=3, output_dim=128, stride=4, dims=None, norm_type="in"):
        super().__init__()
        if dims is None:
            dims = [64, 96, 128, 128]

        self.stride = stride
        self.norm_type = norm_type

        # stem 尽量对齐原始 BasicEncoder
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=7, stride=2, padding=3, bias=False),
            build_norm(32, norm_type),
            nn.ReLU(inplace=True)
        )

        # 用创新点一替换原始残差提取
        self.stage1 = self._make_stage(32, dims[0], depth=1, stride=1)
        self.stage2 = self._make_stage(dims[0], dims[1], depth=2, stride=2)
        self.stage3 = self._make_stage(dims[1], dims[2], depth=4, stride=2)
        self.stage4 = self._make_stage(dims[2], dims[3], depth=2, stride=2)

        # 融合部分完全回退到原始 cat 逻辑
        sum_dims = sum(dims)  # 64 + 96 + 128 + 256 = 544
        self.norm_fuse = build_norm(output_dim * 2, norm_type)

        self.conv2 = nn.Conv2d(sum_dims, output_dim * 2, kernel_size=3, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_stage(self, in_d, out_d, depth, stride=1):
        layers = []

        if stride == 2 or in_d != out_d:
            layers.append(
                ConvNormAct(
                    in_d, out_d,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    norm_type=self.norm_type
                )
            )
        else:
            out_d = in_d

        for _ in range(depth):
            layers.append(
                KineticFeatureBlock(
                    out_d,
                    n_div=2,
                    mlp_ratio=1.0,
                    norm_type=self.norm_type
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, H, W = x.shape

        x = self.stem(x)
        a = self.stage1(x)   # H/2
        b = self.stage2(a)   # H/4
        c = self.stage3(b)   # H/8
        d = self.stage4(c)   # H/16

        def _bilinear_interpolate(feat):
            return F.interpolate(
                feat,
                size=(H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )

        a = _bilinear_interpolate(a)
        b = _bilinear_interpolate(b)
        c = _bilinear_interpolate(c)
        d = _bilinear_interpolate(d)

        x = self.conv2(torch.cat([a, b, c, d], dim=1))
        x = self.norm_fuse(x)
        x = self.relu2(x)
        x = self.conv3(x)

        return x

# -----------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------
class EfficientCorrBlock:
    def __init__(
            self,
            fmaps,
            num_levels=4,
            radius=4,
            padding_mode="zeros",
    ):
        B, S, C, H, W = fmaps.shape
        self.padding_mode = padding_mode
        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []

        self.fmaps_pyramid.append(fmaps)
        for i in range(self.num_levels - 1):
            fmaps_ = fmaps.reshape(B * S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid.append(fmaps)

    def sample(self, coords, target):
        r = self.radius
        device = coords.device
        B, S, N, D = coords.shape
        assert D == 2

        target = target.permute(0, 1, 3, 2).unsqueeze(-1)

        out_pyramid = []
        for i in range(self.num_levels):
            pyramid = self.fmaps_pyramid[i]
            C, H, W = pyramid.shape[2:]
            centroid_lvl = (
                    torch.cat(
                        [torch.zeros_like(coords[..., :1], device=device), coords], dim=-1
                    ).reshape(B * S, N, 1, 1, 3)
                    / 2 ** i
            )

            dx = torch.linspace(-r, r, 2 * r + 1, device=device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=device)

            xgrid, ygrid = torch.meshgrid(dy, dx, indexing="ij")
            zgrid = torch.zeros_like(xgrid, device=device)
            delta = torch.stack([zgrid, xgrid, ygrid], axis=-1)
            delta_lvl = delta.view(1, 1, 2 * r + 1, 2 * r + 1, 3)
            coords_lvl = centroid_lvl + delta_lvl
            pyramid_sample = bilinear_sampler(
                pyramid.reshape(B * S, C, 1, H, W), coords_lvl
            )

            corr = torch.sum(target * pyramid_sample.reshape(B, S, C, N, -1), dim=2)
            corr = corr / torch.sqrt(torch.tensor(C).float())
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
        out = out.permute(0, 2, 1, 3).contiguous().view(B * N, S, -1).float()
        return out


class CorrBlock:
    def __init__(
            self,
            fmaps,
            num_levels=4,
            radius=4,
            multiple_track_feats=False,
            padding_mode="zeros",
    ):
        B, S, C, H, W = fmaps.shape
        self.S, self.C, self.H, self.W = S, C, H, W
        self.padding_mode = padding_mode
        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []
        self.multiple_track_feats = multiple_track_feats

        self.fmaps_pyramid.append(fmaps)
        for i in range(self.num_levels - 1):
            fmaps_ = fmaps.reshape(B * S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid.append(fmaps)

    def sample(self, coords):
        r = self.radius
        B, S, N, D = coords.shape
        assert D == 2

        H, W = self.H, self.W
        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i]  # B, S, N, H, W
            *_, H, W = corrs.shape

            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1).to(
                coords.device
            )

            centroid_lvl = coords.reshape(B * S * N, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corrs = bilinear_sampler(
                corrs.reshape(B * S * N, 1, H, W),
                coords_lvl,
                padding_mode=self.padding_mode,
            )
            corrs = corrs.view(B, S, N, -1)
            out_pyramid.append(corrs)

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
        out = out.permute(0, 2, 1, 3).contiguous().view(B * N, S, -1).float()
        return out

    def corr(self, targets):
        B, S, N, C = targets.shape
        if self.multiple_track_feats:
            targets_split = targets.split(C // self.num_levels, dim=-1)
            B, S, N, C = targets_split[0].shape

        assert C == self.C
        assert S == self.S

        fmap1 = targets

        self.corrs_pyramid = []
        for i, fmaps in enumerate(self.fmaps_pyramid):
            *_, H, W = fmaps.shape
            fmap2s = fmaps.view(B, S, C, H * W)  # B S C H W ->  B S C (H W)
            if self.multiple_track_feats:
                fmap1 = targets_split[i]
            corrs = torch.matmul(fmap1, fmap2s)
            corrs = corrs.view(B, S, N, H, W)  # B S N (H W) -> B S N H W
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            self.corrs_pyramid.append(corrs)


class Attention(nn.Module):
    def __init__(
            self, query_dim, context_dim=None, num_heads=8, dim_head=48, qkv_bias=False
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        context_dim = default(context_dim, query_dim)
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.heads = num_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, attn_bias=None):
        B, N1, C = x.shape
        h = self.heads

        q = self.to_q(x).reshape(B, N1, h, C // h).permute(0, 2, 1, 3)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        N2 = context.shape[1]
        k = k.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        v = v.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        # print(f'q:{q.shape},k:{k.shape}')
        sim = (q @ k.transpose(-2, -1)) * self.scale
        # print(f'sim:{sim.shape}')
        if attn_bias is not None:
            sim = sim + attn_bias
        attn = sim.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        return self.to_out(x)


# ---------------------------------------------------------------------------------------
class GQAAttention(Attention):
    def __init__(
            self,
            query_dim: int,
            context_dim: Optional[int] = None,
            num_heads: int = 8,
            num_groups: int = 2,  # GQA新增：查询分组数（需满足num_heads % num_groups == 0）
            dim_head: int = 48,
            qkv_bias: bool = False
    ):
        super().__init__(query_dim, context_dim, num_heads, dim_head, qkv_bias)
        # 核心修改：KV按组共享，而非按头独立
        self.num_groups = num_groups
        assert num_heads % num_groups == 0, "头数必须能被组数整除（如8头→2组，每组4头）"
        self.group_size = num_heads // num_groups  # 每组包含的头数

        # 重新定义KV投影层：输出维度 = 2 * 组数 * 头维度（原先是2 * 头数 * 头维度）
        inner_dim = dim_head * num_heads
        self.to_kv = nn.Linear(
            context_dim or query_dim,
            2 * num_groups * dim_head,  # 2: K和V各占一半；num_groups: 按组生成
            bias=qkv_bias
        )

    def forward(self, x, context=None, attn_bias=None):
        B, N1, C = x.shape  # N1：查询序列长度
        h = self.heads
        g = self.num_groups
        d = self.dim_head  # 头维度
        group_size = self.group_size  # 每组头数（如8头/2组=4）

        # 1. 计算查询Q（与原多头注意力一致，每个头独立）
        q = self.to_q(x).reshape(B, N1, h, d).permute(0, 2, 1, 3)  # 形状：(B, h, N1, d)

        # 2. 计算键K和值V（按组共享，每组1套KV）
        context = default(context, x)
        N2 = context.shape[1]  # KV序列长度
        kv = self.to_kv(context).chunk(2, dim=-1)  # 拆分K和V，各为 (B, N2, g*d)
        k, v = [
            t.reshape(B, N2, g, d).permute(0, 2, 1, 3)  # 形状：(B, g, N2, d)
            for t in kv
        ]

        # 3. 将KV广播到每组对应的头（如2组→每组4头，KV复制4次）
        k = k.repeat_interleave(group_size, dim=1)  # 形状：(B, h, N2, d)（g组→h头）
        v = v.repeat_interleave(group_size, dim=1)  # 同上

        # 4. 后续注意力计算与原逻辑一致（点积+softmax+加权求和）
        sim = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, N1, N2)
        if attn_bias is not None:
            sim = sim + attn_bias
        attn = sim.softmax(dim=-1)  # (B, h, N1, N2)
        out = (attn @ v).transpose(1, 2).reshape(B, N1, C)  # (B, N1, C)

        return self.to_out(out)


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
class AttnBlock(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_heads,
            attn_class: Callable[..., nn.Module] = Attention,
            mlp_ratio=4.0,
            **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = attn_class(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )


    def forward(self, x, mask=None):
        attn_bias = mask
        if mask is not None:
            mask = (
                (mask[:, None] * mask[:, :, None])
                .unsqueeze(1)
                .expand(-1, self.attn.num_heads, -1, -1)
            )
            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value
        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x

#========================================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, stride=4):
        super(BaseEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = "instance"
        self.in_planes = output_dim // 2

        self.norm1 = nn.InstanceNorm2d(self.in_planes)

        # 1. 初始卷积层 (Stem)
        self.conv1 = nn.Conv2d(input_dim, self.in_planes, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        # 2. 四个阶段的残差块 (Backbone)
        dim1 = output_dim // 2  # 默认 64
        dim2 = output_dim // 4 * 3  # 默认 96
        dim3 = output_dim  # 默认 128
        dim4 = output_dim  # 默认 128 (如果您改进的Pconv在此为256，请自行修改)

        self.layer1 = self._make_layer(dim1, stride=1)
        self.layer2 = self._make_layer(dim2, stride=2)
        self.layer3 = self._make_layer(dim3, stride=2)
        self.layer4 = self._make_layer(dim4, stride=2)

        # 3. 特征融合 (替换为我们设计的 SemanticRoutingFusion)
        self.fpn = RecoveredGuideRoutingFPN(
            in_channels_list=[dim1, dim2, dim3, dim4],
            out_dim=output_dim,
            # norm=self.norm_fn
        )

        # 4. 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # --- 第1步: 初始特征提取 ---
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # --- 第2步: 通过四个残差阶段，得到不同尺度的特征 ---
        a = self.layer1(x)  # L1: 分辨率 (H/2, W/2)
        b = self.layer2(a)  # L2: 分辨率 (H/4, W/4)
        c = self.layer3(b)  # L3: 分辨率 (H/8, W/8)
        d = self.layer4(c)  # L4: 分辨率 (H/16, W/16)

        # --- 第3步: 多尺度特征语义路由融合 (FPN) ---
        # FPN 内部会自动处理分辨率的插值与对其，最终输出分辨率为 H/4
        out = self.fpn([a, b, c, d])

        return out


class ConservativeBaseEncoder(nn.Module):
    """BasicEncoder-style backbone with conservative routing fusion."""

    def __init__(self, input_dim=3, output_dim=128, stride=4):
        super(ConservativeBaseEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = "instance"
        self.in_planes = output_dim // 2

        self.norm1 = nn.InstanceNorm2d(self.in_planes)
        self.conv1 = nn.Conv2d(input_dim, self.in_planes, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        dim1 = output_dim // 2
        dim2 = output_dim // 4 * 3
        dim3 = output_dim
        dim4 = output_dim

        self.layer1 = self._make_layer(dim1, stride=1)
        self.layer2 = self._make_layer(dim2, stride=2)
        self.layer3 = self._make_layer(dim3, stride=2)
        self.layer4 = self._make_layer(dim4, stride=2)

        self.fpn = ConservativeGuideRoutingFPN(
            in_channels_list=[dim1, dim2, dim3, dim4],
            out_dim=output_dim,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)
        return self.fpn([a, b, c, d])


class StandaloneLRGConservativeFPNEncoder(nn.Module):
    """Standalone LRG backbone with conservative metric-preserving FPN."""

    def __init__(
        self,
        input_dim=3,
        output_dim=128,
        stride=4,
        dims=None,
        depths=None,
        kernels=None,
        geometry_stages=None,
        norm_type="in",
    ):
        super().__init__()
        if dims is None:
            dims = [64, 96, 128, 128]
        if depths is None:
            depths = [1, 2, 3, 1]
        if kernels is None:
            kernels = [3, 3, 5, 5]
        if geometry_stages is None:
            geometry_stages = [False, False, True, True]
        assert len(dims) == 4 and len(depths) == 4 and len(kernels) == 4
        assert len(geometry_stages) == 4

        self.stride = stride
        self.dims = dims
        self.depths = depths
        self.kernels = kernels
        self.geometry_stages = geometry_stages
        self.norm_type = norm_type

        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=2, padding=1, bias=False),
            build_norm(32, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, dims[0], kernel_size=3, stride=1, padding=1, bias=False),
            build_norm(dims[0], norm_type),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(dims[0], depths[0], kernels[0], geometry_stages[0])
        self.down2 = AntiAliasDownsample(dims[0], dims[1], norm_type=norm_type)
        self.stage2 = self._make_stage(dims[1], depths[1], kernels[1], geometry_stages[1])
        self.down3 = AntiAliasDownsample(dims[1], dims[2], norm_type=norm_type)
        self.stage3 = self._make_stage(dims[2], depths[2], kernels[2], geometry_stages[2])
        self.down4 = AntiAliasDownsample(dims[2], dims[3], norm_type=norm_type)
        self.stage4 = self._make_stage(dims[3], depths[3], kernels[3], geometry_stages[3])

        self.fpn = ConservativeGuideRoutingFPN(dims, out_dim=output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_stage(self, dim, depth, kernel_size, use_geometry):
        return nn.Sequential(
            *[
                RigidDescriptorBlock(
                    dim,
                    kernel_size=kernel_size,
                    mlp_ratio=1.0,
                    norm_type=self.norm_type,
                    use_geometry=use_geometry,
                )
                for _ in range(depth)
            ]
        )

    def forward_features(self, x):
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(self.down2(f1))
        f3 = self.stage3(self.down3(f2))
        f4 = self.stage4(self.down4(f3))
        return [f1, f2, f3, f4]

    def forward(self, x):
        return self.fpn(self.forward_features(x))
