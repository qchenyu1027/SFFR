
from copy import copy
from pathlib import Path
import warnings
from typing import Optional
import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from networkx.generators.degree_seq import directed_configuration_model
from torch import einsum
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _triple, _pair, _single
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.DEA import DECA
from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized
from timm.models.layers import DropPath
######################################## KAN begin ########################################
# from fuliye import CombinedBasisLayer
from models.kan_convs.kagn_conv import *
from models.kan_convs.kangn_bottlekan import *
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    def forward_fuse(self, x):
        return self.act(self.conv(x))
class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x)
        return torch.cat(x, self.d)
class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out


class LearnableWeights(nn.Module):
    def __init__(self):
        super(LearnableWeights, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x1, x2):
        out = x1 * self.w1 + x2 * self.w2
        return out

class RadialBasisFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=8):#benlai 8
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.inputdim = inputdim
        self.outdim = outdim

        # 初始化参数，确保类型正确
        self.register_parameter(
            'fouriercoeffs',
            nn.Parameter(
                torch.randn(2, outdim, inputdim, gridsize) /
                (np.sqrt(inputdim) * np.sqrt(self.gridsize))
            )
        )

    def to(self, device):
        # 确保继承的to方法被调用
        super(NaiveFourierKANLayer, self).to(device)
        # 明确将fouriercoeffs移到指定设备
        if self.fouriercoeffs is not None:
            self.fouriercoeffs = nn.Parameter(self.fouriercoeffs.to(device))
        return self

    def forward(self, x):
        # 确保权重在正确的设备和类型上
        device = x.device
        dtype = x.dtype
        self.fouriercoeffs = self.fouriercoeffs.to(device=device, dtype=dtype)

        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)
        x = x.view(-1, self.inputdim)  # (b*h*w, c)

        # 生成k并确保在正确的设备和类型上
        k = torch.arange(1, self.gridsize + 1, device=device, dtype=dtype).reshape(1, 1, 1, self.gridsize)

        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))

        # 确保拼接和计算都在正确的设备和类型上
        fourier_input = torch.cat([c, s], dim=0)
        y = torch.einsum("dbik,djik->bj", fourier_input, self.fouriercoeffs)

        # 重塑输出
        y = y.view(b, h, w, self.outdim)
        y = y.permute(0, 3, 1, 2)

        return y

class AdaptivePool2d(nn.Module):
    def __init__(self, output_h, output_w, pool_type='avg'):
        super(AdaptivePool2d, self).__init__()

        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, input_h, input_w = x.shape

        if (input_h > self.output_h) or (input_w > self.output_w):
            self.stride_h = input_h // self.output_h
            self.stride_w = input_w // self.output_w
            self.kernel_size = (input_h - (self.output_h - 1) * self.stride_h, input_w - (self.output_w - 1) * self.stride_w)

            if self.pool_type == 'avg':
                y = nn.AvgPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
            else:
                y = nn.MaxPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
        else:
            y = x

        return y

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
#####new
class NaiveFourierKANLayer1(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=8):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.inputdim = inputdim
        self.outdim = outdim
        self.register_parameter(
            'fouriercoeffs',
            nn.Parameter(
                torch.randn(2, outdim, inputdim, gridsize) /
                (np.sqrt(inputdim) * np.sqrt(self.gridsize))
            )
        )

    def forward(self, x_main, x_aux):  # 接收两个模态的输入
        device = x_main.device
        dtype = x_main.dtype
        fouriercoeffs = self.fouriercoeffs.to(device=device, dtype=dtype)

        b, c, h, w = x_main.shape
        x_main = x_main.permute(0, 2, 3, 1).contiguous().view(-1, self.inputdim)  # [B*H*W, C]
        x_aux = x_aux.permute(0, 2, 3, 1).contiguous().view(-1, self.inputdim)

        k = torch.arange(1, self.gridsize + 1, device=device, dtype=dtype).reshape(1, 1, 1, self.gridsize)

        c = torch.cos(k * x_main.view(x_main.shape[0], 1, x_main.shape[1], 1))
        s = torch.sin(k * x_aux.view(x_aux.shape[0], 1, x_aux.shape[1], 1))

        c = c.reshape(1, x_main.shape[0], x_main.shape[1], self.gridsize)
        s = s.reshape(1, x_aux.shape[0], x_aux.shape[1], self.gridsize)

        fourier_input = torch.cat([c, s], dim=0)  # shape: [2, B*H*W, C, G]
        y = torch.einsum("dbik,djik->bj", fourier_input, fouriercoeffs)
        y = y.view(b, h, w, self.outdim).permute(0, 3, 1, 2)
        return y

class FastKANConvNDLayer1(nn.Module):
    def __init__(self, conv_class, norm_class, input_dim, output_dim, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0):
        super(FastKANConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.grid_size = grid_size
        self.base_activation = base_activation()
        self.grid_range = grid_range

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        self.shared_dim_reduction = nn.ModuleList([
            nn.Linear(input_dim // groups, (input_dim // groups) // 2)
            for _ in range(groups)
        ])


        self.base_conv = nn.ModuleList([DepthwiseSeparableConv(
            input_dim // groups,
            output_dim // groups,
            kernel_size,
            stride,
            padding
        ) for _ in range(groups)])

        self.spline_conv = nn.ModuleList([conv_class(grid_size * input_dim // groups,
                                                     output_dim // groups,
                                                     kernel_size,
                                                     stride,
                                                     padding,
                                                     dilation,
                                                     groups=1,
                                                     bias=False) for _ in range(groups)])

        self.fourier_kan = nn.ModuleList([
            NaiveFourierKANLayer1(input_dim // groups, output_dim // groups)
            for _ in range(groups)
        ])

        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups) for _ in range(groups)])
        self.rbf = RadialBasisFunction(grid_range[0], grid_range[1], grid_size)

        self.weight_generator = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(input_dim // groups, 64, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, 1),
                nn.Softmax(dim=1)
            ) for _ in range(groups)
        ])

        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)
        self.dwconv = DepthwiseSeparableConv(input_dim, output_dim, padding=1)


    def forward_fast_kan(self, rgb_x, ir_x, group_index, input_dim, output_dim):

        self.inputdim = input_dim
        self.outdim = output_dim

        spline_basisrgb = (self.rbf(self.layer_norm[group_index](rgb_x))).moveaxis(-1, 2).flatten(1, 2)  # 1 1024 16 16
        spline_rgb = self.spline_conv[group_index](spline_basisrgb)

        spline_basisir= (self.rbf(self.layer_norm[group_index](ir_x))).moveaxis(-1, 2).flatten(1, 2)  # 1 1024 16 16
        spline_ir = self.spline_conv[group_index](spline_basisir)

        rgb_output = self.fourier_kan[group_index](spline_rgb,spline_ir)  # 1 128 16 16
        ir_output = self.fourier_kan[group_index](spline_ir,spline_rgb)  # 1 128 16 16

        return rgb_output,ir_output


    def forward(self, x):
        rgb_fea_flat = x[0]  # [B, L, C]
        ir_fea_flat = x[1]  # [B, L, C]

        bs, nx, c = rgb_fea_flat.size()
        h = w = int(math.sqrt(nx))

        # 将BLC转换为BCHW格式
        rgb_fea_flat = rgb_fea_flat.transpose(1, 2).contiguous().view(bs, c, h, w)  # [B, C, H, W]
        ir_fea_flat = ir_fea_flat.transpose(1, 2).contiguous().view(bs, c, h, w)  # [B, C, H, W]

        # 将RGB和IR特征分别按组划分
        split_x1 = torch.split(rgb_fea_flat, self.inputdim // self.groups, dim=1)
        split_x2 = torch.split(ir_fea_flat, self.inputdim // self.groups, dim=1)

        output1 = []
        output2 = []

        # 处理RGB特征
        for group_ind, (r, t) in enumerate(zip(split_x1, split_x2)): # bchw
            y1,y2 = self.forward_fast_kan(split_x1.clone(), split_x2.clone(),group_ind, self.inputdim, self.outdim,mode="rgb")
            output1.append(y1)
            output2.append(y2)

        # 分别连接RGB和IR的输出
        out_vis = torch.cat(output1, dim=1)  # [B, C, H, W]
        out_ir = torch.cat(output2, dim=1)  # [B, C,` H, W]
        ###4.7
        out_vis = self.dwconv(out_vis)
        out_ir = self.dwconv(out_ir)
        out_vis = out_vis.view(bs, self.outdim, -1).transpose(1, 2)  # [B, L, C]
        out_ir = out_ir.view(bs, self.outdim, -1).transpose(1, 2)  # [B, L, C]

        return [out_vis, out_ir]
####
class FastKANConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, input_dim, output_dim, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0):
        super(FastKANConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.grid_size = grid_size
        self.base_activation = base_activation()
        self.grid_range = grid_range

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        self.shared_dim_reduction = nn.ModuleList([
            nn.Linear(input_dim // groups, (input_dim // groups) // 2)
            for _ in range(groups)
        ])


        self.base_conv = nn.ModuleList([DepthwiseSeparableConv(
            input_dim // groups,
            output_dim // groups,
            kernel_size,
            stride,
            padding
        ) for _ in range(groups)])

        self.spline_conv = nn.ModuleList([conv_class(grid_size * input_dim // groups,
                                                     output_dim // groups,
                                                     kernel_size,
                                                     stride,
                                                     padding,
                                                     dilation,
                                                     groups=1,
                                                     bias=False) for _ in range(groups)])

        self.fourier_kan = nn.ModuleList([
            NaiveFourierKANLayer(input_dim // groups, output_dim // groups)
            for _ in range(groups)
        ])

        # self.wav = nn.ModuleList([
        #     WaveletTransform((input_dim // groups), output_dim // groups)
        #     for _ in range(groups)
        # ])
        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups) for _ in range(groups)])
        self.rbf = RadialBasisFunction(grid_range[0], grid_range[1], grid_size)


        # self.gsr=GaussianReLUKANLayer(input_dim,grid_size,output_dim)
        self.weight_generator = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(input_dim // groups, 64, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, 1),
                nn.Softmax(dim=1)
            ) for _ in range(groups)
        ])

        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)
        self.dwconv = DepthwiseSeparableConv(input_dim, output_dim, padding=1)


    def forward_fast_kan(self, x, group_index, input_dim, output_dim, mode="all"):

        self.inputdim = input_dim
        self.outdim = output_dim
        device = x.device
        dtype = x.dtype

        # 确保所有组件在正确的设备和类型上
        self = self.to(device)
        x = x.to(device, dtype=dtype)
        # self.coefficient1 = LearnableCoefficient()
        # self.coefficient2 = LearnableCoefficient()
        # self.coefficient3 = LearnableCoefficient()
        # weights = self.weight_generator[group_index](x)
        # alpha = weights[:, 0].view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        # beta = weights[:, 1].view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        # gamma = weights[:, 2].view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        # dynamic_weights = self.weight_generator[group_index](x)
        # Apply base activation to input and then linear transform with base weights
        if self.dropout is not None:
            x = self.dropout(x)
        base_output = torch.zeros_like(x)
        fourier_output = torch.zeros_like(x)
        # 1
        if mode == "rgb":
            base_output = self.base_conv[group_index](self.base_activation(x))
        # 2
        elif mode == "ir":
            fourier_output = self.fourier_kan[group_index](x)  # 1 128 16 16
        # 3
        spline_basis = self.rbf(self.layer_norm[group_index](x))  # 1 128 16 16 8
        spline_basis = spline_basis.moveaxis(-1, 2).flatten(1, 2)  # 1 1024 16 16
        spline_output = self.spline_conv[group_index](spline_basis)
        x=base_output+spline_output+fourier_output
        # x = (alpha * base_output +
        #      beta * spline_output +
        #      gamma * fourier_output)

        return x


    def forward(self, x):
        rgb_fea_flat = x[0]  # [B, L, C]
        ir_fea_flat = x[1]  # [B, L, C]

        bs, nx, c = rgb_fea_flat.size()
        h = w = int(math.sqrt(nx))

        # 将BLC转换为BCHW格式
        rgb_fea_flat = rgb_fea_flat.transpose(1, 2).contiguous().view(bs, c, h, w)  # [B, C, H, W]
        ir_fea_flat = ir_fea_flat.transpose(1, 2).contiguous().view(bs, c, h, w)  # [B, C, H, W]

        # 将RGB和IR特征分别按组划分
        split_x1 = torch.split(rgb_fea_flat, self.inputdim // self.groups, dim=1)
        split_x2 = torch.split(ir_fea_flat, self.inputdim // self.groups, dim=1)

        output1 = []
        output2 = []

        # 处理RGB特征
        for group_ind, x_group in enumerate(split_x1):  # bchw
            y1 = self.forward_fast_kan(x_group.clone(), group_ind, self.inputdim, self.outdim,mode="rgb")
            output1.append(y1)

        # 处理IR特征
        for group_ind, x_group in enumerate(split_x2):
            y2 = self.forward_fast_kan(x_group.clone(), group_ind, self.inputdim, self.outdim,mode="ir")
            output2.append(y2)

        # 分别连接RGB和IR的输出
        out_vis = torch.cat(output1, dim=1)  # [B, C, H, W]
        out_ir = torch.cat(output2, dim=1)  # [B, C,` H, W]
        ###4.7
        out_vis = self.dwconv(out_vis)
        out_ir = self.dwconv(out_ir)
        ###
        # out_vis = self.gsconv(out_vis)
        # out_ir = self.gsconv(out_ir)

        # out_vis = self.conv(out_vis)
        # out_ir = self.conv(out_ir)

        # 将BCHW转回BLC格式
        out_vis = out_vis.view(bs, self.outdim, -1).transpose(1, 2)  # [B, L, C]
        out_ir = out_ir.view(bs, self.outdim, -1).transpose(1, 2)  # [B, L, C]

        return [out_vis, out_ir]

class FastKANConv2DLayer(FastKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0):
        kernel_size = (1, kernel_size)  # 使用1×K的卷积核
        padding = (0, padding)
        super(FastKANConv2DLayer, self).__init__(nn.Conv2d, nn.InstanceNorm2d,
                                                 input_dim, output_dim,
                                                 kernel_size,
                                                 groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                 ndim=2,
                                                 grid_size=grid_size, base_activation=base_activation,
                                                 grid_range=grid_range,
                                                 dropout=dropout)


class KanConvBlock(nn.Module):
    def __init__(self, d_model, block_exp, attn_pdrop=.1, resid_pdrop=.1, k=3, h=8, loops_num=1, e=0.5,
                 dropoutrate=0.25):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super(KanConvBlock, self).__init__()
        # c_=int(d_model*e)
        self.loops = loops_num
        d_k = d_model
        d_v = d_model
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.FFKConv= FastKANConv2DLayer(d_model, d_model, kernel_size=k, padding=k // 2)
        # self.crossatt = CrossAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        # self.crossatt = FastKANConv2DLayer(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),

                                     # nn.SiLU(),  # changed from GELU
                                     nn.GELU(),  # changed from GELU
                                     nn.Linear(block_exp * d_model, d_model),
                                     nn.Dropout(resid_pdrop),
                                     )
        self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                    # nn.SiLU(),  # changed from GELU
                                    nn.GELU(),  # changed from GELU
                                    nn.Linear(block_exp * d_model, d_model),
                                    nn.Dropout(resid_pdrop),
                                    )
        self.mlp = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                 # nn.SiLU(),  # changed from GELU
                                 nn.GELU(),  # changed from GELU
                                 nn.Linear(block_exp * d_model, d_model),
                                 nn.Dropout(resid_pdrop),
                                 )

        # Layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(resid_pdrop)
        # Learnable Coefficient
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()

    def forward(self, x):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]

        assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
        bs, nx, c = rgb_fea_flat.size()
        h = w = int(math.sqrt(nx))

        for loop in range(self.loops):
            # with Learnable Coefficient
            rgb_fea_out, ir_fea_out = self.FFKConv([rgb_fea_flat, ir_fea_flat])  # 1  256 128
            ####ablation
            # rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])  # 1  256 128
            rgb_att_out = self.coefficient1(rgb_fea_flat) + self.coefficient2(rgb_fea_out)
            ir_att_out = self.coefficient3(ir_fea_flat) + self.coefficient4(ir_fea_out)
            rgb_att_out = self.LN1(rgb_att_out)
            ir_att_out = self.LN2(ir_att_out)
            rgb_fea_flat = self.coefficient5(rgb_att_out) + self.coefficient6(self.mlp_vis(self.dropout(rgb_att_out)))
            ir_fea_flat = self.coefficient7(ir_att_out) + self.coefficient8(self.mlp_ir(self.dropout(ir_att_out)))

            # rgb_fea_flat = self.coefficient5(rgb_att_out) + self.coefficient6(self.mlp_vis(self.LN2(rgb_att_out)))
            # ir_fea_flat = self.coefficient7(ir_att_out) + self.coefficient8(self.mlp_ir(self.LN2(ir_att_out)))
            #######
            # rgb_att_out = rgb_fea_flat + rgb_fea_out
            # ir_att_out = ir_fea_flat + ir_fea_out
            # rgb_att_out = self.LN1(rgb_att_out)
            # ir_att_out = self.LN2(ir_att_out)
            # rgb_fea_flat = rgb_att_out + self.mlp_vis(self.dropout(rgb_att_out))
            # ir_fea_flat = ir_att_out + self.mlp_ir(self.dropout(ir_att_out))
            # without Learnable Coefficient
            # rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            # rgb_att_out = rgb_fea_flat + rgb_fea_out
            # ir_att_out = ir_fea_flat + ir_fea_out
            # rgb_fea_flat = rgb_att_out + self.mlp_vis(self.LN2(rgb_att_out))
            # ir_fea_flat = ir_att_out + self.mlp_ir(self.LN2(ir_att_out))

        return [rgb_fea_flat, ir_fea_flat]


class KanConvFusionBlock(nn.Module):
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, block_exp=4, n_layer=1, resid_pdrop=0.1):
        super(KanConvFusionBlock, self).__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        # d_k = d_model
        # d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        ####4.8
        # self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        # self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        ###4.8
        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')

        # LearnableCoefficient
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        # init weights
        self.apply(self._init_weights)

        # cross transformer
        self.Kanconv = nn.Sequential(*[KanConvBlock(d_model=d_model,
                                                    block_exp=block_exp,
                                                    resid_pdrop=resid_pdrop, k=3) for layer in range(n_layer)])
        # self.dynamic_snake_conv = DysnakeConvBlock(d_model, d_model,resid_pdrop=resid_pdrop,block_exp=block_exp, k=3)
        # Concat
        self.concat = Concat(dimension=1)

        # conv1x1
        self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)
        # self.wcmf = WCMF(channel=d_model)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):

        rgb_fea = x[0]
        ir_fea = x[1]
        # print(f"rgb_fea",rgb_fea.shape)   [1,128,20,20]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # ------------------------- cross-modal feature fusion -----------------------#
        # new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]

        # new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
        # print("Original shape:", new_rgb_fea.shape)  #[1,128,16,16]
        ###4.8
        # rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis  # blc
        # ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir  #
        ###4.8
        rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1)   # blc
        ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1)   #

        rgb_fea_flat, ir_fea_flat = self.Kanconv([rgb_fea_flat, ir_fea_flat])  # bx l c

        rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
        else:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
        new_rgb_fea = rgb_fea_CFE + rgb_fea
        ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
        else:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
        new_ir_fea = ir_fea_CFE + ir_fea
        # print("After view1:", new_ir_fea.shape)
        new_fea = self.concat([new_rgb_fea, new_ir_fea])
        new_fea = self.conv1x1_out(new_fea)

        return new_fea