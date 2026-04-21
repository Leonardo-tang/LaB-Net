import torch
import torch.nn as nn
import torchvision.models as models
from models.SRA import SRA
from smt import SMT, smt_t,smt_b
from torch.nn import functional as F
from thop import profile, clever_format
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import time
from timm.models.layers import trunc_normal_
import math
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class SAttention(nn.Module):
    def __init__(self, dim, sa_num_heads=8, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.sa_num_heads = sa_num_heads

        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        head_dim = dim // sa_num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) + \
            self.local_conv(v.transpose(1, 2).reshape(B, N, C).transpose(1, 2).view(B, C, H, W)).view(B, C,
                                                                                                      N).transpose(1, 2)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x.permute(0, 2, 1).reshape(B, C, H, W)
class SCA(nn.Module):
    def __init__(self, all_channel=64, head_num=4, window_size=7):
        super(SCA, self).__init__()
        self.sra = SRA(dim=all_channel, head_num=head_num, window_size=window_size)


        self.conv2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)

    def forward(self, x, ir):
        multiplication = x * ir
        summation = self.conv2(x + ir)

        sa = self.sra(multiplication)
        summation_sa = summation.mul(sa)

        sc_feat = self.sra(summation_sa)

        return sc_feat

class EDS(nn.Module):
    def __init__(self, all_channel=64):
        super(EDS, self).__init__()
        self.conv1 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.dconv1 = BasicConv2d(all_channel, int(all_channel / 4), kernel_size=3, padding=1)
        self.dconv2 = BasicConv2d(all_channel, int(all_channel / 4), kernel_size=3, dilation=3, padding=3)
        self.dconv3 = BasicConv2d(all_channel, int(all_channel / 4), kernel_size=3, dilation=5, padding=5)
        self.dconv4 = BasicConv2d(all_channel, int(all_channel / 4), kernel_size=3, dilation=7, padding=7)
        self.fuse_dconv = nn.Conv2d(all_channel, all_channel, kernel_size=3, padding=1)


    def forward(self, x, ir):
        multiplication = self.conv1(x * ir)
        summation = self.conv2(x + ir)
        fusion = (summation + multiplication)
        x1 = self.dconv1(fusion)
        x2 = self.dconv2(fusion)
        x3 = self.dconv3(fusion)
        x4 = self.dconv4(fusion)
        out = self.fuse_dconv(torch.cat((x1, x2, x3, x4), dim=1))

        return out

class GFM(nn.Module):
    def __init__(self, inc, expend_ratio=2):
        super().__init__()
        self.expend_ratio = expend_ratio
        assert expend_ratio in [2, 3], f"expend_ratio {expend_ratio} mismatch"

        self.sa = SAttention(dim=inc)
        self.dw_pw = DWPWConv(inc * expend_ratio, inc)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def forward(self, x, d):
        B, C, H, W = x.shape
        if self.expend_ratio == 2:
            cat = torch.cat((x, d), dim=1)
        else:
            multi = x * d
            cat = torch.cat((x, d, multi), dim=1)
        x_rc = self.dw_pw(cat).flatten(2).permute(0, 2, 1)
        x_ = self.sa(x_rc, H, W)
        x_ = x_ + x
        return self.act(x_)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class SimpleBidirectionalFusion(nn.Module):
    """简化的双向融合增强模块"""

    def __init__(self, fusion_module):
        super().__init__()
        self.fusion_module = fusion_module

        # 简单的可学习融合权重
        self.weight = nn.Parameter(torch.tensor([0.5, 0.5]))  # 可学习的初始权重

    def forward(self, rgb_feat, depth_feat):
        """
        参数:
            rgb_feat: RGB特征 [B, C, H, W]
            depth_feat: 深度特征 [B, C, H, W]
        返回:
            融合特征 [B, C, H, W]
        """
        # 检查输入通道数是否一致
        if rgb_feat.size(1) != depth_feat.size(1):
            print(f"警告: RGB和深度特征通道数不一致: {rgb_feat.size(1)} vs {depth_feat.size(1)}")
            # 如果需要，可以在这里添加通道调整

        # 正向融合: RGB为主，深度为辅
        rgb_enhanced = self.fusion_module(rgb_feat, depth_feat)

        # 反向融合: 深度为主，RGB为辅
        depth_enhanced = self.fusion_module(depth_feat, rgb_feat)

        # 可学习的门控融合
        gate_weights = torch.softmax(self.weight, dim=0)

        # 最终融合
        fused_feat = gate_weights[0] * rgb_enhanced + gate_weights[1] * depth_enhanced

        # 残差连接（可选）
        fused_feat = fused_feat + 0.1 * (rgb_feat + depth_feat)

        return fused_feat
class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# Global Contextual module
class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import scipy.stats as st


def gkern(kernlen=16, nsig=3):
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def min_max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_-min_+1e-8)


class HA(nn.Module):
    def __init__(self):
        super(HA, self).__init__()
        # 创建多个高斯核以适应不同尺度
        gaussian_kernel_96 = np.float32(gkern(31, 4))  # 用于96×96
        gaussian_kernel_48 = np.float32(gkern(15, 2))  # 用于48×48
        gaussian_kernel_24 = np.float32(gkern(7, 1))  # 用于24×24
        gaussian_kernel_12 = np.float32(gkern(5, 0.5))  # 用于12×12

        # 注册为参数
        self.gaussian_kernel_96 = Parameter(torch.from_numpy(gaussian_kernel_96[np.newaxis, np.newaxis, ...]))
        self.gaussian_kernel_48 = Parameter(torch.from_numpy(gaussian_kernel_48[np.newaxis, np.newaxis, ...]))
        self.gaussian_kernel_24 = Parameter(torch.from_numpy(gaussian_kernel_24[np.newaxis, np.newaxis, ...]))
        self.gaussian_kernel_12 = Parameter(torch.from_numpy(gaussian_kernel_12[np.newaxis, np.newaxis, ...]))

    def forward(self, attention, x):
        # 根据特征图尺寸选择合适的卷积核
        h, w = x.shape[-2:]

        if h == 96 and w == 96:
            kernel = self.gaussian_kernel_96
            padding = 15
        elif h == 48 and w == 48:
            kernel = self.gaussian_kernel_48
            padding = 7
        elif h == 24 and w == 24:
            kernel = self.gaussian_kernel_24
            padding = 3
        elif h == 12 and w == 12:
            kernel = self.gaussian_kernel_12
            padding = 2
        else:
            # 默认使用自适应高斯核
            kernel_size = min(31, h // 2 * 2 + 1)  # 确保为奇数
            sigma = max(0.5, kernel_size / 8)
            kernel = np.float32(gkern(kernel_size, sigma))
            kernel = torch.from_numpy(kernel[np.newaxis, np.newaxis, ...]).to(x.device)
            padding = kernel_size // 2

        # 调整attention到x的尺寸
        if attention.shape[-2:] != x.shape[-2:]:
            attention = F.interpolate(attention, size=x.shape[-2:],
                                      mode='bilinear', align_corners=False)

        # 应用高斯平滑
        soft_attention = F.conv2d(attention, kernel, padding=padding)
        soft_attention = min_max_norm(soft_attention)

        # 扩展attention通道数以匹配x
        if soft_attention.size(1) != x.size(1):
            soft_attention = soft_attention.repeat(1, x.size(1), 1, 1)

        # 融合注意力并应用
        fused_attention = torch.max(soft_attention, attention)
        x = torch.mul(x, fused_attention)

        return x


class MCM(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.rc = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, padding=1, stride=1, groups=inc),
            nn.BatchNorm2d(inc),
            nn.GELU(),
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )
        self.predtrans = nn.Sequential(
            nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, padding=1, groups=outc),
            nn.BatchNorm2d(outc),
            nn.GELU(),
            nn.Conv2d(in_channels=outc, out_channels=1, kernel_size=1)
        )

        self.rc2 = nn.Sequential(
            nn.Conv2d(in_channels=outc * 2, out_channels=outc * 2, kernel_size=3, padding=1, groups=outc * 2),
            nn.BatchNorm2d(outc * 2),
            nn.GELU(),
            nn.Conv2d(in_channels=outc * 2, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

        self.apply(self._init_weights)

    def forward(self, x1, x2):
        x2_upsample = self.upsample2(x2)  # 上采样
        x2_rc = self.rc(x2_upsample)  # 减少通道数
        shortcut = x2_rc

        x_cat = torch.cat((x1, x2_rc), dim=1)  # 拼接
        x_forward = self.rc2(x_cat)  # 减少通道数2
        x_forward = x_forward + shortcut
        pred = F.interpolate(self.predtrans(x_forward), 384, mode="bilinear", align_corners=True)  # 预测图

        return pred, x_forward

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation model, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class DWPWConv(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, padding=1, stride=1, groups=inc),
            nn.BatchNorm2d(inc),
            nn.GELU(),
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

class BBSNet_SMT(nn.Module):
    def __init__(self, channel=32):
        super(BBSNet_SMT, self).__init__()

        # Backbone model
        self.smt = smt_b(pretrained=True)
        self.smt_depth = smt_b(pretrained=True)

        # Decoder 1
        self.rfb1 = GCM(128, 128)
        self.rfb2 = GCM(256, 128)
        self.rfb3 = GCM(512, 128)
        self.agg1 = aggregation(128)

        self.rfb1_1 = GCM(64, 128)
        self.rfb2_1 = GCM(128, 128)
        self.agg2 = aggregation(128)

        # upsample function
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Refinement flow
        self.HA = HA()

        # 边缘提取模块也需要相应调整
        self.edge_extractor_final = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),  # 专门用于单通道输入
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )


        self.eds = EDS(all_channel=64)
        self.sca1 = SCA(all_channel=128)
        self.sca2 = SCA(all_channel=256)
        self.gfm = GFM(inc=512)

        # 双向融合包装器
        self.eds_bi = SimpleBidirectionalFusion(self.eds)
        self.sca1_bi = SimpleBidirectionalFusion(self.sca1)
        self.sca2_bi = SimpleBidirectionalFusion(self.sca2)
        self.gfm_bi = SimpleBidirectionalFusion(self.gfm)

        self.mcm1 = MCM(inc=128, outc=64)


        self.predtrans4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1)
        )
        self.predtrans3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=256),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1)
        )
        self.predtrans2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1)
        )
        self.predtrans1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1)
        )

    def load_smt_pretrained(self, pretrained_path):
        """加载SMT预训练权重到两个分支"""
        print(f'Loading SMT pretrained weights from {pretrained_path}')

        # 加载预训练权重文件
        checkpoint = torch.load(pretrained_path, weights_only=False)

        # 处理不同的权重格式
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        else:
            pretrained_dict = checkpoint

        # 加载到RGB分支
        rgb_success = self._load_to_branch(self.smt, pretrained_dict, 'RGB')
        # 加载到Depth分支
        depth_success = self._load_to_branch(self.smt_depth, pretrained_dict, 'Depth')

        return rgb_success and depth_success

    def _load_to_branch(self, branch, pretrained_dict, branch_name):
        """将预训练权重加载到指定分支"""
        model_dict = branch.state_dict()

        # 筛选出可以加载的权重
        pretrained_weights = {}
        for k, v in pretrained_dict.items():
            # 直接匹配键
            if k in model_dict and v.size() == model_dict[k].size():
                pretrained_weights[k] = v
            # 尝试去掉前缀（如'module.'）
            elif k.replace('module.', '') in model_dict:
                new_key = k.replace('module.', '')
                if v.size() == model_dict[new_key].size():
                    pretrained_weights[new_key] = v

        # 更新模型权重
        model_dict.update(pretrained_weights)
        branch.load_state_dict(model_dict, strict=False)

        return len(pretrained_weights) > 0

    def forward(self, x, x_depth):
        _, (x,x1,x2,x3) = self.smt(x)  # feature_list 包含多尺度特征
        _, (x_depth, x1_depth, x2_depth, x3_depth) = self.smt(x_depth)

        # layer0 merge
        x = self.eds_bi(x, x_depth)


        # layer1 merge
        x1 = self.sca1_bi(x1, x1_depth)
        pred_1 = F.interpolate(self.predtrans2(x1), 384, mode="bilinear", align_corners=True)
        # layer1 merge end

        # layer2 merge
        x2 = self.sca2_bi(x2, x2_depth)
        pred_2 = F.interpolate(self.predtrans3(x2), 384, mode="bilinear", align_corners=True)
        # layer2 merge end

        x3 = self.gfm_bi(x3, x3_depth)
        pred_3 = F.interpolate(self.predtrans4(x3), 384, mode="bilinear", align_corners=True)

        # layer3_1 merge end
        # produce initial saliency map by decoder1
        x_0 = self.rfb1_1(x)
        x1_0 = self.rfb2_1(x1)
        x2_0 = self.rfb2(x2)

        attention_map1 = self.agg2(x2_0, x1_0,x_0)
        x3_1 = self.HA(attention_map1.sigmoid(), x3)
        x2_1 = self.HA(attention_map1.sigmoid(), x2)

        x3_2 = self.rfb3(x3_1)
        x2_2 = self.rfb2(x2_1)

        attention_map2 = self.agg1(x3_2, x2_2, x1_0)
        x1_1 = self.HA(attention_map2.sigmoid(), x1)
        x_1 = self.HA(attention_map2.sigmoid(), x)

        # Refine low-layer features by initial map
        y, xf_1 = self.mcm1(x_1, x1_1)

        return pred_1,pred_2,pred_3, y

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)


if __name__ == '__main__':
    # 初始化模型
    model = BBSNet_SMT().to('cuda')

    # 创建输入张量
    r = torch.randn([1, 3, 384, 384]).to('cuda')
    d = torch.randn([1, 3, 384, 384]).to('cuda')

    # 使用fvcore计算FLOPs和参数量
    model.eval()


    # 设置前向传播函数用于fvcore
    def forward_func(rgb, depth):
        return model(rgb, depth)


    # 计算FLOPs
    flops = FlopCountAnalysis(model, (r, d))
    total_flops = flops.total()

    # 计算参数量
    params = sum(p.numel() for p in model.parameters())


    # 格式化输出
    def clever_format(nums, format_str):
        if not isinstance(nums, (list, tuple)):
            nums = [nums]
        result = []
        for num in nums:
            if num >= 1e9:
                result.append(format_str % (num / 1e9) + "G")
            elif num >= 1e6:
                result.append(format_str % (num / 1e6) + "M")
            elif num >= 1e3:
                result.append(format_str % (num / 1e3) + "K")
            else:
                result.append(format_str % num)
        return result if len(result) > 1 else result[0]


    flops_str, params_str = clever_format([total_flops, params], "%.2f")
    print(f"FLOPs: {flops_str}, Params: {params_str}")

    # 可选：打印详细的层级别统计
    print("\n层级别FLOPs统计:")
    print(flops.by_module())

    # FPS测试
    num_runs = 100  # 增加到100次以获得更准确的结果
    warmup_runs = 10  # 预热运行

    # 预热
    print("\n预热...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            model(r, d)

    # 同步GPU
    torch.cuda.synchronize()

    # 正式测试
    print("开始FPS测试...")
    start_time = time.time()

    for _ in range(num_runs):
        with torch.no_grad():
            model(r, d)

    # 同步GPU
    torch.cuda.synchronize()
    end_time = time.time()

    # 计算FPS
    elapsed_time = end_time - start_time
    fps = num_runs / elapsed_time

    print(f"推理次数: {num_runs}")
    print(f"总耗时: {elapsed_time:.4f}秒")
    print(f"平均单次推理时间: {elapsed_time / num_runs * 1000:.2f}毫秒")
    print(f"FPS: {fps:.2f}")
