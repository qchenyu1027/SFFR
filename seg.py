import os
import shutil

# 原始数据集路径
original_dataset_path = '/media/ailab/EXTERNAL_USB/DroneVehicle/labels/train'
# 新数据集路径
new_dataset_path = '/media/ailab/EXTERNAL_USB/DroneVehiclesmall/labels/train'

# 获取原始数据集中的所有图片文件
all_images = os.listdir(original_dataset_path)

# 假设你想要按文件名顺序选择前400张图片
all_images.sort()

# 创建新的数据集文件夹
if not os.path.exists(new_dataset_path):
    os.makedirs(new_dataset_path)

# 复制前400张图片
for i in range(min(400, len(all_images))):  # 防止原始数据集小于400张
    image_path = os.path.join(original_dataset_path, all_images[i])
    shutil.copy(image_path, new_dataset_path)

print(f"前400张图片已成功复制到 {new_dataset_path}")
