import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import math
from typing import *


class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


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
    def __init__(self, inputdim, outdim, gridsize=8):
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

        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
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
        y = y.view(outshape)

        return y

class FastKANLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 2,
            use_base_update: bool = True,
            base_activation=F.silu,
            spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

        # self.drop = nn.Dropout(p=0.1) # dropout

        self.fourier_kan = NaiveFourierKANLayer(input_dim, output_dim )

        self.spline_weight = nn.Parameter(torch.ones(1))  # Weight for spline basis output
        self.base_weight = nn.Parameter(torch.ones(1))  # Weight for base update output
        # self.fourier_weight = nn.Parameter(torch.ones(1))  #

    def forward(self, x, time_benchmark=False):

        if not time_benchmark:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)

        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))

        base = self.base_linear(self.base_activation(x)) if self.use_base_update else 0
        # fourier_output = self.fourier_kan(x)
        # ret = ret * self.spline_weight
        # base = base * self.base_weight
        # fourier_output = fourier_output * self.fourier_weight
        ret = ret + base

        # dropout
        # ret = self.drop(ret)
        return ret


class FastKAN(nn.Module):
    def __init__(
            self,
            layers_hidden: List[int],
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 2,  # original: 8
            use_base_update: bool = True,
            base_activation=F.silu,
            spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FastKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class KANLinear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            share_spline_weights=False,
            groups=1
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.share_spline_weights = share_spline_weights
        self.groups = groups

        self.base_weight = nn.Linear(in_features, out_features)

        spline_features = grid_size + spline_order
        if share_spline_weights:
            self.spline_weight = nn.Parameter(torch.Tensor(out_features, spline_features))
        else:
            self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, spline_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight.weight, a=math.sqrt(5))
        if self.base_weight.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base_weight.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.base_weight.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))

    def forward(self, x):
        base_output = self.base_weight(x)

        # Simple spline computation
        spline_input = torch.linspace(0, 1, self.grid_size + self.spline_order, device=x.device)
        if self.share_spline_weights:
            spline_output = F.linear(spline_input.repeat(x.size(0), 1), self.spline_weight)
        else:
            spline_output = torch.sum(F.linear(spline_input.repeat(x.size(0), self.in_features, 1),
                                               self.spline_weight.view(-1, self.grid_size + self.spline_order))
                                      * x.unsqueeze(-1), dim=1)

        return base_output + spline_output

    def regularization_loss(self):
        return torch.sum(torch.abs(self.spline_weight))