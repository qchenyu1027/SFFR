import time

import torch
import torch.nn.functional as F


def find_yolo_layer(model, layer_name):
    """Find yolov5 layer to calculate GradCAM and GradCAM++

    Args:
        model: yolov5 model.
        layer_name (str): the name of layer with its hierarchical information.

    Return:
        target_layer: found layer
    """
    hierarchy = layer_name.split('_')
    target_layer = model.model._modules[hierarchy[0]]

    for h in hierarchy[1:]:
        target_layer = target_layer._modules[h]
    return target_layer


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def show_cam_on_image(img, mask, use_rgb=True, colormap=cv2.COLORMAP_JET):
    """
    将GradCAM热力图叠加到原始图像上

    Args:
        img: 原始图像，格式为 (H, W, C)，值范围为 [0, 255]
        mask: GradCAM生成的热力图，格式为 (H, W)，值范围为 [0, 1]
        use_rgb: 是否使用RGB格式，OpenCV默认为BGR
        colormap: 使用的颜色映射，cv2.COLORMAP_JET为红黄蓝渐变

    Returns:
        叠加了热力图的图像，值范围为 [0, 255]
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 热力图和原始图像的加权叠加
    cam = np.float32(heatmap) + np.float32(img)
    cam = cam / np.max(cam)  # 归一化
    cam = np.uint8(255 * cam)  # 转回 [0, 255] 范围

    return cam


# 使用示例
def visualize_gradcam(original_img, gradcam_mask):
    """
    可视化GradCAM结果

    Args:
        original_img: 原始图像，RGB格式，值范围为 [0, 255]
        gradcam_mask: GradCAM生成的热力图，值范围为 [0, 1]
    """
    # 确保图像是RGB格式，值范围为[0, 255]
    if original_img.max() <= 1.0:
        original_img = original_img * 255
    original_img = np.uint8(original_img)

    # 确保热力图的值范围为[0, 1]
    if gradcam_mask.max() > 1.0:
        gradcam_mask = gradcam_mask / 255.0

    # 调整热力图大小以匹配原始图像
    if gradcam_mask.shape[:2] != original_img.shape[:2]:
        gradcam_mask = cv2.resize(gradcam_mask,
                                  (original_img.shape[1], original_img.shape[0]))

    # 生成热力图叠加图
    cam_image = show_cam_on_image(original_img, gradcam_mask)

    # 显示图像
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(gradcam_mask, cmap='jet')
    plt.title('GradCAM Heatmap')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cam_image)
    plt.title('GradCAM Overlay')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

class YOLOV5GradCAM:

    def __init__(self, model, layer_name, img_size=(640, 640)):
        self.model = model
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None


        target_layer = find_yolo_layer(self.model, layer_name)
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        self.model(torch.zeros(1, 3, *img_size, device=device), torch.zeros(1, 3, *img_size, device=device))
        print('[INFO] saliency_map size :', self.activations['value'].shape[2:])

    def forward(self, img_vis, img_ir, class_idx=True):
        """
        Args:
            input_img: input image with shape of (1, 3, H, W)
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
            preds: The object predictions
        """
        saliency_maps = []

        b, c, h, w = img_vis.size()
        tic = time.time()
        preds, logits = self.model(img_vis, img_ir)
        print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')

        for logit, cls, cls_name, conf in zip(logits[0], preds[1][0], preds[2][0], preds[3][0]):
            if class_idx:
                score = logit[cls]
            else:
                score = logit.max()
            self.model.zero_grad()
            tic = time.time()

            score.backward(retain_graph=True)

            print(f"[INFO] {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')
            gradients = self.gradients['value']
            activations = self.activations['value'].clone()
            b, k, u, v = gradients.size()
            alpha = gradients.view(b, k, -1).mean(2)
            weights = alpha.view(b, k, 1, 1)

            saliency_map = (weights * activations).sum(1, keepdim=True)
            saliency_map = F.relu(saliency_map)
            saliency_map = F.interpolate(input=saliency_map, size=(h, w), mode='bilinear', align_corners=False)
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            # saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
            saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)
            saliency_maps.append(saliency_map)
        return saliency_maps, logits, preds
    #     overlayed_images = [self.overlay_gradcam_on_image(saliency_map, img_vis[0], alpha) for saliency_map in
    #                         saliency_maps]
    #
    #     return overlayed_images, logits, preds
    #
    # @staticmethod
    # def overlay_gradcam_on_image(saliency_map, original_image, alpha=0.5):
    #     """
    #     叠加 Grad-CAM 到原始图像
    #     Args:
    #         saliency_map: Grad-CAM 热图 (Tensor, shape=[1, 1, H, W])
    #         original_image: 原始 RGB 图像 (Tensor, shape=[3, H, W])
    #         alpha: 叠加比例
    #     Returns:
    #         overlayed_image: 叠加后的 NumPy 图像 (shape=[H, W, 3])
    #     """
    #     # 处理 Grad-CAM 热图
    #     saliency_map = saliency_map.squeeze().cpu().numpy()
    #     saliency_map = (saliency_map * 255).astype(np.uint8)  # 归一化到 0-255
    #     saliency_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)  # 伪彩色映射
    #
    #     # 处理原始图像
    #     original_image = original_image.permute(1, 2, 0).cpu().numpy()  # 变换为 (H, W, 3)
    #     original_image = (original_image * 255).astype(np.uint8)  # 归一化到 0-255
    #     original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)  # 转换为 OpenCV 格式 (BGR)
    #
    #     # 叠加热图
    #     overlayed_image = cv2.addWeighted(original_image, 1 - alpha, saliency_map, alpha, 0)
    #
    #     return overlayed_image
    # def forward(self, img_vis, img_ir, class_idx=True):
    #     """
    #     Args:
    #         input_img: input image with shape of (1, 3, H, W)
    #     Return:
    #         mask: saliency map of the same spatial dimension with input
    #         logit: model output
    #         preds: The object predictions
    #     """
    #     saliency_maps = []
    #     b, c, h, w = img_vis.size()
    #     preds, logits = self.model(img_vis, img_ir)
    #     # 遍历每个目标
    #     for logit, cls, cls_name in zip(logits[0], preds[1][0], preds[2][0]):
    #         if class_idx:
    #             score = logit[cls]
    #         else:
    #             score = logit.max()
    #         self.model.zero_grad()
    #         # 模型的对于某一类别的输出，进行反向传播
    #         score.backward(retain_graph=True)
    #         gradients = self.gradients['value']
    #         activations = self.activations['value']
    #         b, k, u, v = gradients.size()
    #
    #         # 对梯度的每个通道进行GAP(全局平均池化操作)
    #         alpha = gradients.view(b, k, -1).mean(2)
    #         # 维度调整，为了后续和目标层输出值逐点相乘
    #         weights = alpha.view(b, k, 1, 1)
    #         # GAP后的梯度值与目标层的输出值逐点相乘
    #         saliency_map = (weights * activations).sum(1, keepdim=True)
    #         # 剔除负值
    #         saliency_map = F.relu(saliency_map)
    #
    #         saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
    #         saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
    #         saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
    #         saliency_maps.append(saliency_map)
    #     return saliency_maps, logits, preds


# class YOLOV5GradCAM:
#
#     def __init__(self, model, layer_name, img_size=(640, 640)):
#         self.model = model
#         self.gradients = dict()
#         self.activations = dict()
#
#         # 使用 register_full_backward_hook 替代 register_backward_hook
#         def backward_hook(module, grad_input, grad_output):
#             self.gradients['value'] = grad_output[0].detach()  # 使用 detach 防止梯度链被修改
#             return None
#
#         def forward_hook(module, input, output):
#             self.activations['value'] = output.detach()  # 使用 detach 防止梯度链被修改
#             return None
#
#         target_layer = find_yolo_layer(self.model, layer_name)
#         target_layer.register_forward_hook(forward_hook)
#         target_layer.register_full_backward_hook(backward_hook)  # 使用新的钩子函数
#
#         device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
#         self.model(torch.zeros(1, 3, *img_size, device=device), torch.zeros(1, 3, *img_size, device=device))
#         print('[INFO] saliency_map size :', self.activations['value'].shape[2:])
#
#     def forward(self, img_vis, img_ir, class_idx=True):
#         """
#         Args:
#             input_img: input image with shape of (1, 3, H, W)
#         Return:
#             mask: saliency map of the same spatial dimension with input
#             logit: model output
#             preds: The object predictions
#         """
#         saliency_maps = []
#
#         b, c, h, w = img_vis.size()
#         tic = time.time()
#         # 确保不使用原地操作
#         with torch.no_grad():
#             preds, logits = self.model(img_vis, img_ir)
#         print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')
#
#         for logit, cls, cls_name, conf in zip(logits[0], preds[1][0], preds[2][0], preds[3][0]):
#             # 对每个类别单独计算 GradCAM
#             self.model.zero_grad()
#
#             # 创建新的输入张量以避免原地操作影响
#             img_vis_copy = img_vis.clone().requires_grad_(True)
#             img_ir_copy = img_ir.clone().requires_grad_(True)
#
#             # 前向传播
#             new_preds, new_logits = self.model(img_vis_copy, img_ir_copy)
#
#             # 选择对应的评分
#             if class_idx:
#                 score = new_logits[0][0][cls]  # 假设批量大小为1
#             else:
#                 score = new_logits[0][0].max()
#
#             tic = time.time()
#
#             # 反向传播
#             score.backward(retain_graph=False)  # 不保留计算图，避免内存累积
#
#             print(f"[INFO] {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')
#
#             if 'value' in self.gradients:
#                 gradients = self.gradients['value']
#                 activations = self.activations['value']
#
#                 if gradients is not None and activations is not None:
#                     b, k, u, v = gradients.size()
#
#                     # 计算权重
#                     alpha = gradients.view(b, k, -1).mean(2)
#                     weights = alpha.view(b, k, 1, 1)
#
#                     # 计算 GradCAM
#                     saliency_map = (weights * activations).sum(1, keepdim=True)
#                     saliency_map = F.relu(saliency_map)
#                     saliency_map = F.interpolate(input=saliency_map, size=(h, w), mode='bilinear', align_corners=False)
#
#                     # 归一化
#                     saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
#                     if saliency_map_max > saliency_map_min:  # 避免除以零
#                         saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)
#                         saliency_maps.append(saliency_map)
#
#         return saliency_maps, logits, preds
    def __call__(self, img_vis, img_ir):
        return self.forward(img_vis, img_ir)

