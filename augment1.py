import numpy as np
import tifffile as tf           #tifffile是tiff文件的读取库
from PIL import Image
import cv2 as  cv

# import cv2
# import numpy as np
# import tifffile as tif
# def tif2png(imgpath):
#     img = tif.imread(imgpath)# 读取图片 imgpath为图片所在位置
#     #将图片数据类型转换为无符号8位
#     img = img/img.max()
#     img =img*255-0.001 # 减去0.001防止变成负整型
#     img =img.astype(np.uint8)
#     print(img.shape)#显示图片大小和通道数通道数可能大于3
#     b=img[:,:,0]# 蓝通道
#     g=img[:,:,1]# 绿通道
#     r = img[:,:,2]# 红通道
#     nir =img[:,:,3]#近红外通道
#     # 通遒拼接
#     rgb= np.dstack([r,g,b])
#     # 存储png格式图像
#     cv2.imwrite("image.png", rgb)
#     cv2.waitkey(0)#窗口等待响应
#     cv2.destroyAllwindows()#消除窗囗
# if __name__ == '__main__':
#     tif2png('/media/ailab/EXTERNAL_USB/sea_drones_see_multispectral/images/train/2.tiff')

import os
import cv2
import numpy as np
import tifffile as tif


def tif2png(imgpath, output_folder):
    # 读取TIFF图像
    img = tif.imread(imgpath)

    # 将图片数据类型转换为无符号8位
    img = img / img.max()  # 归一化
    img = img * 255 - 0.001  # 减去0.001防止变成负值
    img = img.astype(np.uint8)

    print(f"Processing {imgpath} - Shape: {img.shape}")  # 显示图片大小和通道数

    # 假设图片是有4个通道（RGB + NIR），根据需要选择处理的通道
    b = img[:, :, 0]  # 蓝通道
    g = img[:, :, 1]  # 绿通道
    r = img[:, :, 2]  # 红通道
    # nir = img[:, :, 3]  # 近红外通道，不做任何处理（不在输出中）

    # 拼接RGB通道
    rgb = np.dstack([r, g, b])

    # 获取文件名（去除扩展名）并生成新的文件路径
    file_name = os.path.basename(imgpath)  # 获取文件名
    new_file_name = os.path.splitext(file_name)[0] + '.png'  # 更改扩展名为.png
    output_path = os.path.join(output_folder, new_file_name)  # 拼接输出文件路径

    # 存储为PNG格式图像
    cv2.imwrite(output_path, rgb)
    print(f"Saved to: {output_path}")


def process_folder(input_folder, output_folder):
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有 TIFF 文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.tif') or file_name.endswith('.tiff'):  # 仅处理 TIFF 文件
            input_path = os.path.join(input_folder, file_name)
            tif2png(input_path, output_folder)

    print("所有文件处理完成。")


if __name__ == '__main__':
    # 输入和输出文件夹路径
    input_folder = r'/media/ailab/EXTERNAL_USB/sea_drones_see_multispectral/images/train'
    output_folder = r'/media/ailab/EXTERNAL_USB/sea_drones_see_multispectral/images/processed'

    # 处理文件夹
    process_folder(input_folder, output_folder)
