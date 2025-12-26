from functools import lru_cache
import torch
import torch.nn as nn
from torch.nn.functional import conv2d
import math
# from utils import NoiseInjection
class NoiseInjection(nn.Module):
    def __init__(self, p: float = 0.0, alpha: float = 0.05):
        super(NoiseInjection, self).__init__()
        self.p = p
        self.alpha = alpha

    def get_noise(self, x):
        dims = tuple(i for i in range(len(x.shape)) if i != 1)
        std = torch.std(x, dim=dims, keepdim=True)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype) * std
        return noise

    def forward(self, x):
        if self.training:
            mask = torch.rand(x.shape, device=x.device, dtype=x.dtype)
            mask = (mask < self.p).float() * 1
            x = x + self.alpha * mask * self.get_noise(x)
            return x
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class BottleNeckKAGNConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, conv_w_fun, input_dim, output_dim, degree, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1, dropout: float = 0.0, ndim: int = 2.,
                 dim_reduction: float = 4, min_internal: int = 16,
                 **norm_kwargs):
        super(BottleNeckKAGNConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.base_activation = nn.SiLU()
        self.conv_w_fun = conv_w_fun
        self.ndim = ndim
        self.dropout = None
        self.norm_kwargs = norm_kwargs
        self.p_dropout = dropout

        inner_dim = int(max((input_dim // groups) / dim_reduction,
                            (output_dim // groups) / dim_reduction))
        if inner_dim < min_internal:
            self.inner_dim = min(min_internal, input_dim // groups, output_dim // groups)
        else:
            self.inner_dim = inner_dim
        if dropout > 0:
            self.dropout = NoiseInjection(p=dropout, alpha=0.05)

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        # self.base_conv = nn.ModuleList([conv_class(input_dim // groups,
        #                                            output_dim // groups,
        #                                            kernel_size,
        #                                            stride,
        #                                            padding,
        #                                            dilation,
        #                                            groups=1,
        #                                            bias=False) for _ in range(groups)])

        self.base_conv = nn.ModuleList([DepthwiseSeparableConv(
            input_dim // groups,
            output_dim // groups,
            kernel_size,
            stride,
            padding
        ) for _ in range(groups)])
        self.inner_proj = nn.ModuleList([conv_class(input_dim // groups,
                                                    self.inner_dim,
                                                    1,
                                                    1,
                                                    0,
                                                    1,
                                                    groups=1,
                                                    bias=False) for _ in range(groups)])
        self.out_proj = nn.ModuleList([conv_class(self.inner_dim,
                                                  output_dim // groups,
                                                  1,
                                                  1,
                                                  0,
                                                  1,
                                                  groups=1,
                                                  bias=False) for _ in range(groups)])

        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **norm_kwargs) for _ in range(groups)])

        poly_shape = (groups, self.inner_dim, self.inner_dim * (degree + 1)) + tuple(
            kernel_size for _ in range(ndim))

        self.poly_weights = nn.Parameter(torch.randn(*poly_shape))
        self.beta_weights = nn.Parameter(torch.zeros(degree + 1, dtype=torch.float32))

        # Initialize weights using Kaiming uniform distribution for better training start
        # for conv_layer in self.base_conv:
        #     nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
        # for conv_layer in self.inner_proj:
        #     nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
        # for conv_layer in self.out_proj:
        #     nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

        nn.init.kaiming_uniform_(self.poly_weights, nonlinearity='linear')
        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0 / ((kernel_size ** ndim) * self.inputdim * (self.degree + 1.0)),
        )

    def beta(self, n, m):
        return (
                       ((m + n) * (m - n) * n ** 2) / (m ** 2 / (4.0 * n ** 2 - 1.0))
               ) * self.beta_weights[n]

    @lru_cache(maxsize=128)  # Cache to avoid recomputation of Gram polynomials
    def gram_poly(self, x, degree):
        p0 = x.new_ones(x.size())

        if degree == 0:
            return p0.unsqueeze(-1)

        p1 = x
        grams_basis = [p0, p1]

        for i in range(2, degree + 1):
            p2 = x * p1 - self.beta(i - 1, i) * p0
            grams_basis.append(p2)
            p0, p1 = p1, p2

        return torch.concatenate(grams_basis, dim=1)

    def forward_kag(self, x, group_index):

        if self.dropout is not None:
            x = self.dropout(x)

        # Apply base activation to input and then linear transform with base weights
        basis = self.base_conv[group_index](self.base_activation(x))

        # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
        x = self.inner_proj[group_index](x)
        x = torch.tanh(x).contiguous()

        grams_basis = self.base_activation(self.gram_poly(x, self.degree))

        y = self.conv_w_fun(grams_basis, self.poly_weights[group_index],
                            stride=self.stride, dilation=self.dilation,
                            padding=self.padding, groups=1)
        y = self.out_proj[group_index](y)

        y = self.base_activation(self.layer_norm[group_index](y + basis))

        return y
    def forward(self, x):
        rgb_fea_flat = x[0]  # [B, L, C]
        ir_fea_flat = x[1]  # [B, L, C]

        bs, nx, c = rgb_fea_flat.size()
        h = w = int(math.sqrt(nx))

        # 将BLC转换为BCHW格式
        rgb_fea_flat = rgb_fea_flat.transpose(1, 2).contiguous().view(bs, c, h, w)  # [B, C, H, W]
        ir_fea_flat = ir_fea_flat.transpose(1, 2).contiguous().view(bs, c, h, w)  # [B, C, H, W]


        # print(f"ir_fea_flat", ir_fea_flat.shape)
        # rgb_fea_flat = self.base_conv(rgb_fea_flat)
        # ir_fea_flat = self.base_conv(ir_fea_flat)
        # 将RGB和IR特征分别按组划分
        split_x1 = torch.split(rgb_fea_flat, self.inputdim // self.groups, dim=1)
        split_x2 = torch.split(ir_fea_flat, self.inputdim // self.groups, dim=1)

        output1 = []
        output2 = []

        # 处理RGB特征
        for group_ind, x_group in enumerate(split_x1):#bchw
            y1 = self.forward_kag(x_group.clone(), group_ind)
            output1.append(y1)

        # 处理IR特征
        for group_ind, x_group in enumerate(split_x2):
            y2 = self.forward_kag(x_group.clone(), group_ind)
            output2.append(y2)

        # 分别连接RGB和IR的输出
        out_vis = torch.cat(output1, dim=1)  # [B, C, H, W]
        out_ir = torch.cat(output2, dim=1)  # [B, C, H, W]
        # print(f"out_vis1", out_vis.shape)
        # 将BCHW转回BLC格式
        out_vis = out_vis.view(bs, self.outdim, -1).transpose(1, 2)  # [B, L, C]
        out_ir = out_ir.view(bs, self.outdim, -1).transpose(1, 2)  # [B, L, C]
        # print(f"out_vis2", out_vis.shape)
        return [out_vis, out_ir]
    # def forward(self, x):
    #
    #     split_x = torch.split(x, self.inputdim // self.groups, dim=1)
    #     output = []
    #     for group_ind, _x in enumerate(split_x):
    #         y = self.forward_kag(_x, group_ind)
    #         output.append(y.clone())
    #     y = torch.cat(output, dim=1)
    #     return y




class BottleNeckKAGNConv2DLayer(BottleNeckKAGNConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm2d, dim_reduction: float = 4, **norm_kwargs):
        super(BottleNeckKAGNConv2DLayer, self).__init__(nn.Conv2d, norm_layer, conv2d,
                                                        input_dim, output_dim,
                                                        degree, kernel_size, dim_reduction=dim_reduction,
                                                        groups=groups, padding=padding, stride=stride,
                                                        dilation=dilation,
                                                        ndim=2, dropout=dropout, **norm_kwargs)
