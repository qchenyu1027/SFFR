import torch
import torch.nn as nn
import numpy as np
# class NaiveFourierKANLayer(nn.Module):
#     def __init__(self, inputdim, outdim, gridsize=300):
#         super(NaiveFourierKANLayer, self).__init__()
#         self.gridsize = gridsize
#         self.inputdim = inputdim
#         self.outdim = outdim
#
#         self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) /
#                                           (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
#     def forward(self, x):
#         xshp = x.shape
#         outshape = xshp[0:-1] + (self.outdim,)
#         x = x.view(-1, self.inputdim)
#         # Starting at 1 because constant terms are in the bias
#         k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
#         xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
#         # This should be fused to avoid materializing memory
#         c = torch.cos(k * xrshp)
#         s = torch.sin(k * xrshp)
#         c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
#         s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
#         y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
#
#         y = y.view(outshape)
#         return y

class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=300):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.inputdim = inputdim
        self.outdim = outdim
        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) /
                                          (np.sqrt(inputdim) * np.sqrt(self.gridsize)))

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)
        x = x.view(-1, self.inputdim)  # (b*h*w, c)

        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)

        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))

        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        y = y.view(b, h, w, self.outdim)  # 重新整形为原始空间维度
        y = y.permute(0, 3, 1, 2)  # 转换回(b, c, h, w)格式
        return y

class RadialBasisFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -2,
            grid_max: float = 2,
            num_grids: int = 8,
            denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)
class CombinedBasisLayer(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            fourier_grid=300,
            rbf_grids=8,
            rbf_min=-2,
            rbf_max=2,
            groups=4
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.groups = groups

        # 傅里叶层
        self.fourier_layer = NaiveFourierKANLayer(input_dim, output_dim, fourier_grid)

        # RBF层
        self.rbf = RadialBasisFunction(
            grid_min=rbf_min,
            grid_max=rbf_max,
            num_grids=rbf_grids
        )

        # 计算总通道数 (傅里叶输出 + RBF输出)
        total_channels = output_dim + (rbf_grids * input_dim)

        # 合并后的卷积层
        self.combined_conv = nn.ModuleList([
            nn.Conv2d(
                total_channels // groups,
                output_dim // groups,
                kernel_size=3,
                padding=1,
                bias=False
            ) for _ in range(groups)
        ])

        # 可选：添加归一化层
        self.norm = nn.BatchNorm2d(total_channels)
        self.activation = nn.ReLU()

        # 特征权重
        self.fourier_weight = nn.Parameter(torch.ones(1))
        self.rbf_weight = nn.Parameter(torch.ones(1))

    def rbf_transform(self, x):
        # 输入 x: (B, C, H, W)
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, self.input_dim)
        # 应用RBF
        rbf_out = self.rbf(x_flat)  # (-1, input_dim, num_grids)
        # 重塑回空间维度
        rbf_out = rbf_out.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return rbf_out

    def forward(self, x):
        # 傅里叶变换
        fourier_out = self.fourier_layer(x)  # (B, output_dim, H, W)

        # RBF变换
        rbf_out = self.rbf_transform(x)  # (B, rbf_grids*input_dim, H, W)

        # 合并特征
        combined = torch.cat([
            self.fourier_weight * fourier_out,
            self.rbf_weight * rbf_out
        ], dim=1)

        # 归一化
        combined = self.norm(combined)

        # 分组卷积处理
        group_size = combined.shape[1] // self.groups
        outputs = []
        for i in range(self.groups):
            group_input = combined[:, i * group_size:(i + 1) * group_size]
            group_output = self.combined_conv[i](group_input)
            outputs.append(group_output)

        # 合并输出并激活
        output = torch.cat(outputs, dim=1)
        output = self.activation(output)

        return output


# # 测试代码
# if __name__ == "__main__":
#     # 创建模型
#     model = CombinedBasisLayer(
#         input_dim=128,
#         output_dim=256,
#         fourier_grid=300,
#         rbf_grids=8,
#         rbf_min=-2,
#         rbf_max=2,
#         groups=4
#     )
#
#     # 测试输入
#     x = torch.randn(2, 128, 32, 32)
#     output = model(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {output.shape}")
# if __name__ == '__main__':
#     block = NaiveFourierKANLayer(128,128)
#
#     input1 = torch.randn(1, 128, 20, 20)  # 随机初始化
#     # input2 = torch.randn(1, 128, 20, 20)
#     y=block(input1)
#     print(y.shape)