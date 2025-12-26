from skimage import io
import cv2
import numpy as np
import os

# def process_tiff_folder(input_folder, visible_folder, ir_folder):
#     # 确保输出目录存在
#     os.makedirs(visible_folder, exist_ok=True)
#     os.makedirs(ir_folder, exist_ok=True)
#
#     # 遍历文件夹中的所有 .tiff 文件
#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith(".tiff") or filename.lower().endswith(".tif"):  # 兼容 .tif 和 .tiff
#             img_path = os.path.join(input_folder, filename)
#             process_tiff(img_path, visible_folder, ir_folder)
#
# def process_tiff(img_path, visible_folder, ir_folder):
#     # 读取 TIFF 图像
#     img = io.imread(img_path)
#
#     # 归一化并转换到 0-255
#     img = (img / img.max()) * 255
#     img = img.astype(np.uint8)
#
#     # 如果 TIFF 图像少于 4 通道，无法提取 NIR
#     if img.shape[-1] < 4:
#         print(f"⚠️ 警告：{img_path} 没有近红外 (NIR) 通道，跳过该文件！")
#         return
#
#     # 提取 RGB 和 NIR
#     r, g, b = img[:, :, 2], img[:, :, 1], img[:, :, 0]  # OpenCV 用 BGR
#     nir = img[:, :, 4]  # 近红外通道
#
#     # 创建 RGB 图像
#     rgb = np.dstack([r, g, b])
#
#     # 生成保存路径
#     base_name = os.path.splitext(os.path.basename(img_path))[0]  # 获取文件名（不带后缀）
#     rgb_path = os.path.join(visible_folder, f"{base_name}.png")
#     nir_path = os.path.join(ir_folder, f"{base_name}.png")
#
#     # 保存 RGB 和 NIR 图像
#     cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))  # OpenCV 保存 BGR
#     cv2.imwrite(nir_path, nir)  # 直接保存灰度 NIR
#
#     print(f"✅ 处理完成: {img_path} -> RGB: {rgb_path}, NIR: {nir_path}")
#
# # 指定输入和输出文件夹路径
# input_folder = "/media/ailab/EXTERNAL_USB/sea_drones_see_multispectral/images/val"
# visible_folder = "/media/ailab/EXTERNAL_USB/SeaDronesee/visible1/val"
# ir_folder = "/media/ailab/EXTERNAL_USB/SeaDronesee/infrared1/val"
#
# # 处理整个文件夹
# process_tiff_folder(input_folder, visible_folder, ir_folder)






#
# import cv2
# import numpy as np
# import torch
# import os
# import sys
#
# def pack_gbrg_raw(raw):                                               # 定义一个名为pack_gbrg_raw的函数，它接受一个参数raw，这个参数是一个GBRG格式的Bayer原始图像
#     #pack GBRG Bayer raw to 4 channels
#     black_level = 240                                                 # 定义黑色和白色的级别。这两个值用于后续的图像归一化。
#     white_level = 2**12-1
#     im = raw.astype(np.float32)                                       # 将输入的原始图像转换为浮点数类型
#     im = np.maximum(im - black_level, 0) / (white_level-black_level)  # 对图像进行归一化处理，使其值在0到1之间
#
#     im = np.expand_dims(im, axis=2)                                   # 在第三个维度（即通道维度）上为图像增加一个维度。
#     img_shape = im.shape
#     H = img_shape[0]                                                  # 获取图像的形状，并将高度和宽度分别存储在H和W中。
#     W = img_shape[1]
#
#     out = np.concatenate((im[1:H:2, 0:W:2, :],          # r           # 将图像的四个通道（R，Gr，B，Gb）沿着第三个维度（即通道维度）进行拼接。
#                           im[1:H:2, 1:W:2, :],          # gr
#                           im[0:H:2, 1:W:2, :],          # b
#                           im[0:H:2, 0:W:2, :]), axis=2) # gb
#     return out
#
# def tensor2numpy(raw):  # raw: 1 * 4 * H * W
#     input_full = raw.permute((0, 2, 3, 1))   # 1 * H * W * 4
#     input_full = input_full.data.cpu().numpy()
#     output = np.clip(input_full,0,1)
#     return output
#
# def preprocess(raw):
#     input_full = raw.transpose((0, 3, 1, 2))
#     input_full = torch.from_numpy(input_full)
#     input_full = input_full.cuda()
#     return input_full
#
# img_path = "CRVD_dataset/indoor_raw_noisy/indoor_raw_noisy_scene7/scene7/ISO1600"                        # tiff格式图片路径
# save_path = "CRVD_dataset/indoor_raw_noisy/indoor_raw_noisy_scene7/scene7/ISO1600_png"                   # 转换后的保存路径
#
# file = os.listdir(path=img_path)
#
# for item in file:
#     img = cv2.imread(os.path.join(img_path,item),-1)
#
#     img = np.expand_dims(pack_gbrg_raw(img), axis=0)
#
#     isp = torch.load('isp/ISP_CNN.pth')
#     test_gt = (img - 240) / (2 ** 12 - 1 - 240)                   # 计算地面真实图像的标准化值
#     gt_raw_frame = test_gt * (2 ** 12 - 1 - 240) + 240            # 计算输出的最终值
#     gt_srgb_frame = tensor2numpy(isp(preprocess(gt_raw_frame)))[0]   # 将预处理后的图像转换为sRGB格式
#
#     img_png = np.uint8(gt_srgb_frame * 255)
#
#
#     out_file = item[:-5] + ".jpg"
#     print('图片{}转换为png成功！'.format(item))
#
#     cv2.imwrite(os.path.join(save_path,out_file), img_png)
#
#     key = cv2.waitKey(30) & 0xff
#     if key == 27:
#         sys.exit(0)

