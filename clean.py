import os

# 定义infrared和visible文件夹的路径
infrared_dir = "/home/ailab/qqqcy/DroneVehicle/infrared/train"
visible_dir = "/home/ailab/qqqcy/DroneVehicle/visible/train"

# 获取infrared文件夹中的所有图片文件名
infrared_files = os.listdir(infrared_dir)

# 获取visible文件夹中的所有图片文件名
visible_files = os.listdir(visible_dir)

# 找出visible文件夹中infrared文件夹没有对应的图片
extra_visible_files = [f for f in visible_files if f not in infrared_files]

# 删除visible文件夹中多余的图片
for file in extra_visible_files:
    os.remove(os.path.join(visible_dir, file))
    print(f"Deleted {file} from visible folder.")

print("Done!")