# Loss functions

import torch
import torch.nn as nn
import numpy as np

from utils.general import *
from utils.torch_utils import is_parallel
from utils.plots import plot_samples
from torch.autograd import Variable
from descriptor.LSS import denseLSS
from descriptor.CFOG import denseCFOG





def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        #*********
        batch_pos = true.sum()
        batch_neg = (1 - true).sum()
        # 根据批次中正负样本比例动态调整alpha
        # dynamic_alpha = batch_neg / (batch_pos + batch_neg)
        # alpha_factor = true * dynamic_alpha + (1 - true) * (1 - dynamic_alpha)
        #***********
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class FocalLoss1(nn.Module):
    def __init__(self, loss_fcn, min_alpha=0.25, max_alpha=0.75, min_gamma=1.5, max_gamma=4.0):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)

        positive_samples = true.sum()
        negative_samples = (1 - true).sum()

        # 自适应 alpha 和 gamma
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * (positive_samples / (positive_samples + negative_samples))
        gamma = self.max_gamma - (self.max_gamma - self.min_gamma) * (positive_samples / (positive_samples + negative_samples))

        modulating_factor = (1.0 - p_t) ** gamma
        alpha_factor = true * alpha + (1 - true) * (1 - alpha)
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss





# class ImprovedFocalLoss(nn.Module):
#     def __init__(
#             self,
#             loss_fcn,
#             min_alpha=0.25,
#             max_alpha=0.75,
#             min_gamma=1.5,  # 提高最小gamma值
#             max_gamma=4.0,  # 提高最大gamma值
#             class_weights=None,  # 添加类别权重
#             momentum=0.9  # 添加动量更新
#     ):
#         super(ImprovedFocalLoss, self).__init__()
#         self.loss_fcn = loss_fcn
#         self.min_alpha = min_alpha
#         self.max_alpha = max_alpha
#         self.min_gamma = min_gamma
#         self.max_gamma = max_gamma
#         self.class_weights = class_weights
#         self.momentum = momentum
#         self.reduction = loss_fcn.reduction
#         self.loss_fcn.reduction = 'none'
#
#         # 添加历史统计
#         self.register_buffer('running_pos_ratio', torch.tensor(0.5))
#         self.register_buffer('running_class_ratios', None)
#
#     def update_running_stats(self, pos_ratio, class_ratios=None):
#         """使用动量更新运行时统计"""
#         self.running_pos_ratio = self.momentum * self.running_pos_ratio + (1 - self.momentum) * pos_ratio
#         if class_ratios is not None and self.running_class_ratios is not None:
#             self.running_class_ratios = self.momentum * self.running_class_ratios + (1 - self.momentum) * class_ratios
#
#     def forward(self, pred, true):
#         loss = self.loss_fcn(pred, true)
#         pred_prob = torch.sigmoid(pred)
#         p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
#
#         # 计算正负样本比例
#         positive_samples = true.sum()
#         total_samples = true.numel()
#         pos_ratio = positive_samples / total_samples
#
#         # 更新运行时统计
#         self.update_running_stats(pos_ratio)
#
#         # 使用平滑后的正负样本比例计算自适应参数
#         smooth_pos_ratio = self.running_pos_ratio
#
#         # 根据样本比例自适应调整alpha和gamma
#         alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * smooth_pos_ratio
#         gamma = self.max_gamma - (self.max_gamma - self.min_gamma) * smooth_pos_ratio
#
#         # 计算调制因子
#         modulating_factor = (1.0 - p_t) ** gamma
#
#         # 为高难度样本增加权重
#         hard_example_weight = 1.0 + torch.exp(-torch.abs(pred_prob - 0.5))
#
#         # 应用alpha平衡和难度权重
#         alpha_factor = true * alpha + (1 - true) * (1 - alpha)
#         loss *= alpha_factor * modulating_factor * hard_example_weight
#
#         # 应用类别权重（如果提供）
#         if self.class_weights is not None:
#             class_weights = self.class_weights.to(loss.device)
#             loss *= class_weights.view(-1, 1) if len(loss.shape) > 1 else class_weights
#
#         # 返回损失
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss

import torch.nn.functional as F


class QualityfocalLoss_YOLO(nn.Module):
    def __init__(self, beta=2.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred_score, gt_score):  # 修改为接收两个参数
        # 自动生成正样本掩码
        gt_target_pos_mask = gt_score > 0

        # 原有的损失计算逻辑
        pred_sigmoid = pred_score.sigmoid()
        scale_factor = pred_sigmoid
        zerolabel = scale_factor.new_zeros(pred_score.shape)

        with torch.cuda.amp.autocast(enabled=False):
            loss = F.binary_cross_entropy_with_logits(pred_score, zerolabel, reduction='none') * scale_factor.pow(
                self.beta)

        if gt_target_pos_mask.sum() > 0:  # 如果有正样本
            scale_factor = gt_score[gt_target_pos_mask] - pred_sigmoid[gt_target_pos_mask]
            with torch.cuda.amp.autocast(enabled=False):
                loss[gt_target_pos_mask] = F.binary_cross_entropy_with_logits(
                    pred_score[gt_target_pos_mask],
                    gt_score[gt_target_pos_mask],
                    reduction='none'
                ) * scale_factor.abs().pow(self.beta)

        return loss.mean()  # 返回平均损失
class QualityfocalLoss_YOLO1(nn.Module):
    def __init__(self, beta=2.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred_score, gt_score, gt_target_pos_mask):
        # negatives are supervised by 0 quality score
        pred_sigmoid = pred_score.sigmoid()
        scale_factor = pred_sigmoid
        zerolabel = scale_factor.new_zeros(pred_score.shape)
        with torch.cuda.amp.autocast(enabled=False):
            loss = F.binary_cross_entropy_with_logits(pred_score, zerolabel, reduction='none') * scale_factor.pow(
                self.beta)

        scale_factor = gt_score[gt_target_pos_mask] - pred_sigmoid[gt_target_pos_mask]
        with torch.cuda.amp.autocast(enabled=False):
            loss[gt_target_pos_mask] = F.binary_cross_entropy_with_logits(pred_score[gt_target_pos_mask],
                                                                          gt_score[gt_target_pos_mask],
                                                                          reduction='none') * scale_factor.abs().pow(
                self.beta)
        return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class VarifocalLoss_YOLO(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_score, gt_score):
        pred_sigmoid = pred_score.sigmoid()
        base_weight = self.alpha * (pred_sigmoid - gt_score).abs().pow(self.gamma) * (gt_score <= 0.0).float() + \
                      gt_score * (gt_score > 0.0).float()

        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(
                pred_score.float(),
                gt_score.float(),
                reduction='none'
            ) * base_weight).mean()

        return loss



class VFLoss1(nn.Module):
    def __init__(self, loss_fcn, gamma=2.0, alpha=0.75):
        super(VFLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply VFL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)  # prob from logits

        # 原有的focal weight计算
        focal_weight = true * (true > 0.0).float() + \
                       self.alpha * (pred_prob - true).abs().pow(self.gamma) * (true <= 0.0).float()

        # 添加热成像特定的权重调整
        if pred.dim() > 1:  # 对于objectness预测
            # 增强边界区域的权重
            edge_weight = 1.0 + 0.2 * torch.exp(-torch.abs(pred_prob - 0.5))
            # 对小目标增加权重
            area_weight = 1.0
            if true.sum() > 0:  # 如果有正样本
                area_weight = torch.exp(-true.sum(-1).mean())  # 根据目标密度调整权重

            focal_weight = focal_weight * edge_weight * area_weight

        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# varifocal loss
class VFLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=2.0, alpha=0.75):
        super(VFLoss, self).__init__()
        # 传递 nn.BCEWithLogitsLoss() 损失函数  must be nn.BCEWithLogitsLoss()
        self.loss_fcn = loss_fcn  #
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply VFL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits

        focal_weight = true * (true > 0.0).float() + self.alpha * (pred_prob - true).abs().pow(self.gamma) * (true <= 0.0).float()
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# Ranking Loss
class RankingLoss2(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, true):
        loss = 0
        bs, c, h, w = pred.shape

        for x, y in zip(pred, true):
            x = self.sigmoid(x)

            loss -= y * torch.log(1 - (y - x) + 1e-7) + (1 - y) * torch.log(torch.where(self.threshold - (1 - x) > 0, 1 - self.threshold + (1 - x) + 1e-7, torch.ones([1], dtype=x.dtype, device='cuda')))
        return loss.sum() / (bs * c * h * w)


# Ranking Loss
class RankingLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, true):
        loss = 0.0
        bs, c, h, w = pred.shape

        for pred_i, true_i in zip(pred, true):
            for x, y in zip(pred_i, true_i):
                x = self.sigmoid(x)
                mask_negative = y < 0.3
                mask_positive = y > 0.5
                s = x[mask_positive]
                if len(s) == 0:
                    val = 0
                else:
                    pos_pred = x[mask_positive].min()
                    neg_pred = x[mask_negative].max()
                    val = torch.exp(neg_pred - pos_pred)
                    #val = (((1 + neg_pred - pos_pred) / 2) ** self.gamma) * torch.exp(neg_pred - pos_pred)
                    #val = torch.log(1 - (1 + neg_pred - pos_pred) / 2.0 + 1e-7)
                    '''
                    pos_pred = x[mask_positive].mean()
                    neg_pred = x[~mask_positive].mean()
                    if (pos_pred - neg_pred).item() >= 0.7:
                        val = 0
                    else:
                        val = torch.exp(neg_pred - pos_pred)
                    '''

                loss += val
        return loss / (bs * c)


class SlideLoss(nn.Module):
    def __init__(self, loss_fcn):
        super(SlideLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply SL to each element

    def forward(self, pred, true, auto_iou=0.5):
        loss = self.loss_fcn(pred, true)
        if auto_iou < 0.2:
            auto_iou = 0.2
        b1 = true <= auto_iou - 0.1
        a1 = 1.0
        b2 = (true > (auto_iou - 0.1)) & (true < auto_iou)
        a2 = math.exp(1.0 - auto_iou)
        b3 = true >= auto_iou
        a3 = torch.exp(-(true - 1.0))
        modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
        loss *= modulating_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


# class EMASlideLoss:
#     def __init__(self, loss_fcn, decay=0.999, tau=2000):
#         super(EMASlideLoss, self).__init__()
#         self.loss_fcn = loss_fcn
#         self.reduction = loss_fcn.reduction
#         self.loss_fcn.reduction = 'none'  # required to apply SL to each element
#         self.decay = lambda x: decay * (1 - math.exp(-x / tau))
#         self.is_train = True
#         self.updates = 0
#         self.iou_mean = 1.0
#
#     def __call__(self, pred, true, auto_iou=0.5):
#         if self.is_train and auto_iou != -1:
#             self.updates += 1
#             d = self.decay(self.updates)
#             self.iou_mean = d * self.iou_mean + (1 - d) * float(auto_iou.detach())
#         auto_iou = self.iou_mean
#         loss = self.loss_fcn(pred, true)
#         if auto_iou < 0.2:
#             auto_iou = 0.2
#         b1 = true <= auto_iou - 0.1
#         a1 = 1.0
#         b2 = (true > (auto_iou - 0.1)) & (true < auto_iou)
#         a2 = math.exp(1.0 - auto_iou)
#         b3 = true >= auto_iou
#         a3 = torch.exp(-(true - 1.0))
#         modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
#         loss *= modulating_weight
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:  # 'none'
#             return loss


# Similarity Loss


class SimLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid()

    def des_SSD(self, i, j, descriptor):
        mask_i = torch.ge(i.squeeze(0).squeeze(0), 1)
        mask_i = torch.tensor(mask_i, dtype=torch.float32)
        mask_j = torch.ge(j.squeeze(0).squeeze(0), 1)
        mask_j = torch.tensor(mask_j, dtype=torch.float32)
        mask = torch.mul(mask_i, mask_j)
        num = mask[mask.ge(1)].size()[0]
        if descriptor == 'CFOG':
            des_i = denseCFOG(i)
            des_j = denseCFOG(j)
        elif descriptor == 'LSS':
            des_i = denseLSS(i)
            des_j = denseLSS(j)
        des_i = torch.mul(des_i, mask)
        des_j = torch.mul(des_j, mask)
        SSD_loss = nn.MSELoss(reduction='sum')
        loss = SSD_loss(des_i, des_j) / num
        return loss

    def des_NCC(self, i, j, descriptor):
        mask_i = torch.ge(i.squeeze(0).squeeze(0), 1)
        mask_i = torch.tensor(mask_i, dtype=torch.float32)
        mask_j = torch.ge(j.squeeze(0).squeeze(0), 1)
        mask_j = torch.tensor(mask_j, dtype=torch.float32)
        mask = torch.mul(mask_i, mask_j)
        num = mask[mask.ge(1)].size()[0]
        if descriptor == 'CFOG':
            des_i = denseCFOG(i)
            des_j = denseCFOG(j)
        elif descriptor == 'LSS':
            des_i = denseLSS(i)
            des_j = denseLSS(j)
        des_i = torch.mul(des_i, mask)
        des_j = torch.mul(des_j, mask)
        loss = self.gncc_loss(des_i, des_j) * 512 * 512 / num
        return loss

    def gradient_loss(self, s, penalty='l2'):
        dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
        dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
        d = torch.mean(dx) + torch.mean(dy)
        return d / 2.0

    def mse_loss(self, x, y):
        return torch.mean((x - y) ** 2)

    def DSC(self, pred, target):
        smooth = 1e-5
        m1 = pred.flatten()
        m2 = target.flatten()
        intersection = (m1 * m2).sum()
        return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

    def gncc_loss(self, I, J, eps=1e-5):
        I2 = I.pow(2)
        J2 = J.pow(2)
        IJ = I * J
        I_ave, J_ave = I.mean(), J.mean()
        I2_ave, J2_ave = I2.mean(), J2.mean()
        IJ_ave = IJ.mean()
        cross = IJ_ave - I_ave * J_ave
        I_var = I2_ave - I_ave.pow(2)
        J_var = J2_ave - J_ave.pow(2)
        cc = cross / (I_var.sqrt() * J_var.sqrt() + eps)  # 1e-5
        return -1.0 * cc + 1

    def compute_local_sums(self, I, J, filt, stride, padding, win):
        I2, J2, IJ = I * I, J * J, I * J
        I_sum = nn.functional.conv2d(I, filt, stride=stride, padding=padding)
        J_sum = nn.functional.conv2d(J, filt, stride=stride, padding=padding)
        I2_sum = nn.functional.conv2d(I2, filt, stride=stride, padding=padding)
        J2_sum = nn.functional.conv2d(J2, filt, stride=stride, padding=padding)
        IJ_sum = nn.functional.conv2d(IJ, filt, stride=stride, padding=padding)
        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        return I_var, J_var, cross

    def cc_loss(self, x, y):
        dim = [2, 3, 4]
        mean_x = torch.mean(x, dim, keepdim=True)
        mean_y = torch.mean(y, dim, keepdim=True)
        mean_x2 = torch.mean(x ** 2, dim, keepdim=True)
        mean_y2 = torch.mean(y ** 2, dim, keepdim=True)
        stddev_x = torch.sum(torch.sqrt(mean_x2 - mean_x ** 2), dim, keepdim=True)
        stddev_y = torch.sum(torch.sqrt(mean_y2 - mean_y ** 2), dim, keepdim=True)
        return -torch.mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))

    def Get_Ja(self, flow):
        D_y = (flow[:, 1:, :-1, :-1, :] - flow[:, :-1, :-1, :-1, :])
        D_x = (flow[:, :-1, 1:, :-1, :] - flow[:, :-1, :-1, :-1, :])
        D_z = (flow[:, :-1, :-1, 1:, :] - flow[:, :-1, :-1, :-1, :])
        D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
        D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
        D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
        return D1 - D2 + D3

    def NJ_loss(self, ypred):
        Neg_Jac = 0.5 * (torch.abs(self.Get_Ja(ypred)) - self.Get_Ja(ypred))
        return torch.sum(Neg_Jac)

    def lncc_loss(self, i, j, win=[9, 9], eps=1e-5):
        I = i
        J = j
        I2 = I.pow(2)
        J2 = J.pow(2)
        IJ = I * J
        filters = Variable(torch.ones(1, 1, win[0], win[1])).cuda()
        padding = (win[0] // 2, win[1] // 2)
        I_sum = nn.functional.conv2d(I, filters, stride=1, padding=padding)
        J_sum = nn.functional.conv2d(J, filters, stride=1, padding=padding)
        I2_sum = nn.functional.conv2d(I2, filters, stride=1, padding=padding)
        J2_sum = nn.functional.conv2d(J2, filters, stride=1, padding=padding)
        IJ_sum = nn.functional.conv2d(IJ, filters, stride=1, padding=padding)
        win_size = win[0] * win[1]
        u_I = I_sum / win_size
        u_J = J_sum / win_size
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        cc = cross * cross / (I_var * J_var + eps)
        lcc = -1.0 * torch.mean(cc) + 1
        return lcc

    def forward(self, reference, sensed_tran, sensed, reference_inv_tran, descriptor, similarity):
        if similarity == 'SSD':  # Similarity: SSD or NCC based on descriptors
            loss1 = Variable(self.des_SSD(reference, sensed_tran, descriptor), requires_grad=True)
            loss2 = Variable(self.des_SSD(sensed, reference_inv_tran, descriptor), requires_grad=True)
        elif similarity == 'NCC':
            loss1 = Variable(self.des_NCC(reference, sensed_tran, descriptor), requires_grad=True)
            loss2 = Variable(self.des_NCC(sensed, reference_inv_tran, descriptor), requires_grad=True)

        loss = (loss1 + loss2) * 0.5
        return loss


class EMASlideLoss:
    def __init__(self, loss_fcn, decay=0.999, tau=2000):
        super(EMASlideLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply SL to each element
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        self.is_train = True
        self.updates = 0
        self.iou_mean = 1.0

    def __call__(self, pred, true, auto_iou=0.5):
        if self.is_train and auto_iou != -1:
            self.updates += 1
            d = self.decay(self.updates)
            # self.iou_mean = d * self.iou_mean + (1 - d) * float(auto_iou.detach())
            self.iou_mean = d * self.iou_mean + (1 - d) * float(auto_iou)
        auto_iou = self.iou_mean
        loss = self.loss_fcn(pred, true)
        if auto_iou < 0.2:
            auto_iou = 0.2
        b1 = true <= auto_iou - 0.1
        a1 = 1.0
        b2 = (true > (auto_iou - 0.1)) & (true < auto_iou)
        a2 = math.exp(1.0 - auto_iou)
        b3 = true >= auto_iou
        a3 = torch.exp(-(true - 1.0))
        modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
        loss *= modulating_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss



class VarifocalLoss(nn.Module):
    """ 改进目标存在性检测，适配热成像高响应区域 """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        pred_sigmoid = pred.sigmoid()
        focal_weight = self.alpha * target * torch.pow(target - pred_sigmoid, self.gamma) + \
                      (1 - self.alpha) * (1 - target) * torch.pow(pred_sigmoid - target, self.gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
        return loss.mean() if self.reduction == 'mean' else loss


class QualityfocalLoss_YOLO(nn.Module):
    def __init__(self, beta=2.0):
        super(QualityfocalLoss_YOLO, self).__init__()
        self.beta = beta

    def forward(self, pred, target):
        # 使用sigmoid获得预测概率
        pred_sigmoid = torch.sigmoid(pred)

        # 计算pt (probability of the true class)
        pt = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)

        # 计算权重因子
        weight = (1 - pt) ** self.beta

        # 应用BCE损失
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # 应用权重
        loss = weight * bce

        return loss.mean()



class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = True  # 在计算objectness时是否对ciou进行排序
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        ###3.12
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        # BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        BCEobj = VFLoss(BCEobj)

        ###3.12
        # BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        # BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        # BCEcls = QualityfocalLoss_YOLO(beta=2.0)  # 替换原来的 BCEobj

        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        self.RKobj = RankingLoss(2.0)
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj, lrk = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors, offsets = self.build_targets(p, targets)  # targets

        #plot_samples(batch_index, imgs, path, tcls, tbox, indices, anchors, offsets, targets)
        # dynamic_balance = [1.0 / (3 ** i) for i in range(len(p))]
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)

                # MPDIoU
                # iou = bbox_iou1(pbox, tbox[i], MDPIoU=True, hw=tobj.size()[2] * tobj.size()[3]).squeeze()
                # iou = bbox_iou1(pbox, tbox[i], MDPIoU=True, hw=tobj.size()[2] * tobj.size()[3]).squeeze()
                # iou = bbox_mpdiou(pbox, tbox[i],  xywh=False, mpdiou_hw=tobj.size()[2] * tobj.size()[3]).squeeze()
                # iou = bbox_inner_mpdiou(pbox, tbox[i], xywh=False,mpdiou_hw=tobj.size()[2] * tobj.size()[3], ratio=0.7).squeeze()
                # iou = bbox_focaler_mpdiou(pbox, tbox[i], xywh=False,mpdiou_hw=tobj.size()[2] * tobj.size()[3], d=0.0, u=0.95).squeeze()
                #####
                lbox += (1.0 - iou).mean()  # iou loss
                #####
                # lbox += (1.0 - iou).mean() * (1 + 0.2 * torch.log(torch.tensor(i + 1)))
                ####
                # 计算目标物体 IoU
                # thermal_mask = targets[:, -1] == 1  # 红外通道标识
                # if thermal_mask.any():
                #     thermal_iou = iou[thermal_mask]
                #     thermal_weight = torch.exp(-thermal_iou.mean() * 3).clamp(0.2, 0.6)
                #     alpha = 0.9  # EMA 平滑因子
                #     if hasattr(self, "ema_iou"):
                #         self.ema_iou = alpha * self.ema_iou + (1 - alpha) * thermal_iou.mean()
                #     else:
                #         self.ema_iou = thermal_iou.mean()
                #     alpha_iou = (thermal_iou ** 1.5).mean()  # 让 IoU 更关注低 IoU 目标
                #     mpd_iou = bbox_iou1(pbox, tbox[i], MDPIoU=True, hw=tobj.size()[2] * tobj.size()[3]).squeeze()
                #     thermal_loss = (1.0 - mpd_iou).mean() + 0.05 * (1.0 - alpha_iou).mean() + 0.1 * (1.0 - self.ema_iou)
                #     lbox += thermal_weight * thermal_loss

                #####3.12
                # thermal_mask = targets[:, -1] == 1  # 假设最后一列为红外通道标识
                # if thermal_mask.any():
                #     thermal_iou = iou[thermal_mask]
                #     lbox += 0.3 * (1.0 - thermal_iou).mean()  # 强化红外通道损失权重
                ####
                # Objectness
                # score_iou = iou.detach().clamp(0).type(tobj.dtype)
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)  # 将iou按照从小到大进行排序，返回下标索引
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # 根据model.gr设置真实框的标签值

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Ranking
                # lrk += self.RKobj(pi[..., 4], tobj)

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lrk *= 0.1
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + lrk
        return loss * bs, torch.cat((lbox, lobj, lcls, lrk)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, offset = [], [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append(
                (b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            offset.append(offsets)

        return tcls, tbox, indices, anch, offset





# class ComputeLoss:
#     # Compute losses
#     def __init__(self, model, autobalance=False):
#         super(ComputeLoss, self).__init__()
#         self.sort_obj_iou = True  # 在计算objectness时是否对ciou进行排序
#         device = next(model.parameters()).device  # get model device
#         h = model.hyp  # hyperparameters
#
#
#
#         # Define criteria
#         # BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
#         BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
#         ###3.9
#         BCEobj = VFLoss(BCEobj)
#         ####
#         # BCEcls = VFLoss(BCEcls)
#         # BCEcls = EMASlideLoss(BCEcls)
#         #######3.9
#         ## BCEobj = QualityfocalLoss_YOLO(beta=2.0)  # 替换原来的 BCEobj
#         ###
#         BCEcls = QualityfocalLoss_YOLO(beta=2.0)  # 替换原来的 BCEobj
#         #######
#
#         self.beta = 0.95  # 更平滑的移动平均
#         self.interval = 20  # 适中的更新间隔
#         self.step = 0
#         self.running_losses = {}
#         self.warmup_steps = 300  # 较长的warmup
#         self.original_weights = {'box': h['box'], 'obj': h['obj'], 'cls': h['cls']}
#         self.dynamic_weights = self.original_weights.copy()
#         self.loss_history = {k: [] for k in self.original_weights}
#         self.iou_history = []  # 记录IOU分布
#         self.weight_bounds = {
#             'box': (0.8, 1.2),  # box损失权重变化范围相对保守
#             'obj': (0.7, 1.3),  # obj损失允许较大变化
#             'cls': (0.7, 1.3)  # cls损失允许较大变化
#         }
#
#         #######
#
#         # BCEcls = VarifocalLoss_YOLO(alpha=0.75, gamma=2.0)
#         self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets
#
#         # Focal loss
#         g = h['fl_gamma']  # focal loss gamma
#         if g > 0:
#             BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
#
#         det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
#         self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
#         self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
#         self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
#         self.RKobj = RankingLoss(2.0)
#         for k in 'na', 'nc', 'nl', 'anchors':
#             setattr(self, k, getattr(det, k))
#             # 针对短训练周期的动态权重调整`
#
#
#
#
#     def update_weights(self, loss_dict, iou_scores=None):
#         """改进的权重更新策略"""
#         # 首次初始化
#         if not self.running_losses:
#             self.running_losses = {k: v.item() for k, v in loss_dict.items()}
#             return
#
#         # Warmup期间
#         if self.step < self.warmup_steps:
#             warmup_ratio = self.step / self.warmup_steps
#             for k in self.dynamic_weights:
#                 self.dynamic_weights[k] = (self.original_weights[k] * (1 - warmup_ratio) +
#                                            self.dynamic_weights[k] * warmup_ratio)
#             return
#
#         # 更新移动平均和历史记录
#         for k, v in loss_dict.items():
#             curr_loss = v.item()
#             self.running_losses[k] = (
#                     self.beta * self.running_losses[k] +
#                     (1 - self.beta) * curr_loss
#             )
#             self.loss_history[k].append(curr_loss)
#
#         if iou_scores is not None:
#             self.iou_history.append(iou_scores.mean().item())
#
#         # 基于历史数据调整权重
#         window_size = min(30, len(self.loss_history[k]))
#         if window_size >= 10:
#             for k in self.dynamic_weights:
#                 recent_losses = self.loss_history[k][-window_size:]
#                 avg_recent = sum(recent_losses) / len(recent_losses)
#
#                 # 计算损失趋势
#                 if len(recent_losses) > 1:
#                     trend = (recent_losses[-1] - recent_losses[0]) / len(recent_losses)
#                 else:
#                     trend = 0
#
#                 # 基于趋势的温和调整
#                 if trend > 0:  # 损失在增加
#                     adjust_ratio = 1.02
#                 elif trend < 0:  # 损失在减少
#                     adjust_ratio = 0.98
#                 else:
#                     adjust_ratio = 1.0
#
#                 # 应用调整
#                 self.dynamic_weights[k] *= adjust_ratio
#
#                 # 应用范围限制
#                 lower, upper = self.weight_bounds[k]
#                 orig_w = self.original_weights[k]
#                 self.dynamic_weights[k] = max(orig_w * lower, min(orig_w * upper, self.dynamic_weights[k]))
#
#         # 额外的box loss调整（基于IOU分布）
#         if len(self.iou_history) > 10:
#             recent_ious = self.iou_history[-10:]
#             avg_iou = sum(recent_ious) / len(recent_ious)
#             if avg_iou < 0.5:  # 如果IOU普遍较低
#                 self.dynamic_weights['box'] = min(
#                     self.dynamic_weights['box'] * 1.01,
#                     self.original_weights['box'] * self.weight_bounds['box'][1]
#                 )
#
#
#
#
#     def __call__(self, p, targets):  # predictions, targets, model
#         device = targets.device
#
#         lcls, lbox, lobj, lrk = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
#         tcls, tbox, indices, anchors, offsets = self.build_targets(p, targets)  # targets
#
#         #plot_samples(batch_index, imgs, path, tcls, tbox, indices, anchors, offsets, targets)
#         ############
#         total_iou_scores = []
#         ############
#         # Losses
#         for i, pi in enumerate(p):  # layer index, layer predictions
#             b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
#             tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
#
#             n = b.shape[0]  # number of targets
#             if n:
#                 ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
#
#                 # Regression
#                 pxy = ps[:, :2].sigmoid() * 2. - 0.5
#                 pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
#                 pbox = torch.cat((pxy, pwh), 1)  # predicted box
#                 iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
#                 # iou = bbox_iou1(pbox, tbox[i], MDPIoU=True, hw=tobj.size()[2] * tobj.size()[3]).squeeze()
#                 ############
#                 total_iou_scores.append(iou)
#                 ############
#                 lbox += (1.0 - iou).mean()  # iou loss
#
#                 # ********
#                 # iou = bbox_inner_iou(pbox, tbox[i], xywh=False,SIoU=True, ratio=0.7)
#                 # iou = bbox_focaler_iou(pbox, tbox[i], xywh=False, PIoU=True, d=0.0, u=0.95)
#                 # lbox += iou.mean()  # iou loss
#                 # lbox += loss_box
#                 # ********
#
#                 # # Objectness
#                 score_iou = iou.detach().clamp(0).type(tobj.dtype)
#                 if self.sort_obj_iou:
#                     sort_id = torch.argsort(score_iou)  # 将iou按照从小到大进行排序，返回下标索引
#                     # thermal_conf = ps[:, 4].sigmoid()
#                     # combined_score = score_iou * (1 + 0.2 * thermal_conf)  # 结合热特征和IOU分数
#                     # sort_id = torch.argsort(combined_score)
#                     b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
#                 tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # 根据model.gr设置真实框的标签值
#
#                 # Classification
#                     # ********
#                     # thermal_conf = ps[:, 4].sigmoid()
#                     # combined_score = score_iou * (1 + 0.2 * thermal_conf)  # 结合热特征和IOU分数
#                     # sort_id = torch.argsort(combined_score)
#                     # ********
#                 # ********
#
#                 # Classification
#                 if self.nc > 1:  # cls loss (only if multiple classes)
#                     t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
#                     t[range(n), tcls[i]] = self.cp
#                     lcls += self.BCEcls(ps[:, 5:], t)  # BCE
#
#
#             # Ranking
#             #lrk += self.RKobj(pi[..., 4], tobj)
#
#             obji = self.BCEobj(pi[..., 4], tobj)
#             lobj += obji * self.balance[i]  # obj loss
#             if self.autobalance:
#                 self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
#
#         if self.autobalance:
#             self.balance = [x / self.balance[self.ssi] for x in self.balance]
# ###*****
#         loss_components = {
#             'box': lbox,
#             'obj': lobj,
#             'cls': lcls
#         }
#
#         # 收集所有IOU分数
#         if total_iou_scores:
#             total_iou = torch.cat(total_iou_scores)
#         else:
#             total_iou = torch.zeros(1, device=device)
#
#         self.step += 1
#         if self.step % self.interval == 0:
#             self.update_weights(loss_components, total_iou)
#
#         # 应用权重
#         lbox *= self.dynamic_weights['box']
#         lobj *= self.dynamic_weights['obj']
#         lcls *= self.dynamic_weights['cls']
#         ###*****
#         # lbox *= self.hyp['box']
#         # lobj *= self.hyp['obj']
#         # lcls *= self.hyp['cls']
#
#         lrk *= 0.1
#         bs = tobj.shape[0]  # batch size
#
#
#         loss = lbox + lobj + lcls + lrk
#         return loss * bs, torch.cat((lbox, lobj, lcls, lrk)).detach()
#
#
#
#     def build_targets(self, p, targets):
#         # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
#         na, nt = self.na, targets.shape[0]  # number of anchors, targets
#         tcls, tbox, indices, anch, offset = [], [], [], [], []
#         gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
#         ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
#         targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
#
#         g = 0.5  # bias
#         off = torch.tensor([[0, 0],
#                             [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
#                             # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
#                             ], device=targets.device).float() * g  # offsets
#
#         for i in range(self.nl):
#             anchors, shape = self.anchors[i], p[i].shape
#             gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
#
#             # Match targets to anchors
#             t = targets * gain
#             if nt:
#                 # Matches
#                 r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
#                 j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
#                 # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
#
#                 t = t[j]  # filter
#
#                 # Offsets
#                 gxy = t[:, 2:4]  # grid xy
#                 gxi = gain[[2, 3]] - gxy  # inverse
#                 j, k = ((gxy % 1. < g) & (gxy > 1.)).T
#                 l, m = ((gxi % 1. < g) & (gxi > 1.)).T
#                 j = torch.stack((torch.ones_like(j), j, k, l, m))
#                 t = t.repeat((5, 1, 1))[j]
#                 offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
#             else:
#                 t = targets[0]
#                 offsets = 0
#
#             # Define
#             b, c = t[:, :2].long().T  # image, class
#             gxy = t[:, 2:4]  # grid xy
#             gwh = t[:, 4:6]  # grid wh
#             gij = (gxy - offsets).long()
#             gi, gj = gij.T  # grid xy indices
#
#             # Append
#             a = t[:, 6].long()  # anchor indices
#             indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid indices
#             tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
#             anch.append(anchors[a])  # anchors
#             tcls.append(c)  # class
#             offset.append(offsets)
#
#         return tcls, tbox, indices, anch, offset



def get_inner_iou(box1, box2, xywh=True, eps=1e-7, ratio=0.7):
    def xyxy2xywh(x):
        """
        Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
        top-left corner and (x2, y2) is the bottom-right corner.

        Args:
            x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.

        Returns:
            y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
        """
        assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
        y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
        y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
        y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
        y[..., 2] = x[..., 2] - x[..., 0]  # width
        y[..., 3] = x[..., 3] - x[..., 1]  # height
        return y

    if not xywh:
        box1, box2 = xyxy2xywh(box1), xyxy2xywh(box2)
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - (w1 * ratio) / 2, x1 + (w1 * ratio) / 2, y1 - (h1 * ratio) / 2, y1 + (
                h1 * ratio) / 2
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - (w2 * ratio) / 2, x2 + (w2 * ratio) / 2, y2 - (h2 * ratio) / 2, y2 + (
                h2 * ratio) / 2

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 * ratio * ratio + w2 * h2 * ratio * ratio - inter + eps
    return inter / union


def bbox_iou1(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, MDPIoU=False, hw=None, eps=1e-7, ratio=0.7, scale=0.0, Lambda=1.3):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    innner_iou = get_inner_iou(box1, box2, xywh=xywh, ratio=ratio)

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    elif MDPIoU:
        d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
        d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
        return iou - d1 / hw - d2 / hw  # MPDIoU
    return iou  # IoU














