import json

# 读取原始的 COCO 格式标注文件
input_json_path = '/home/ailab/qqqcy/DroneVehicle/annotations/instances_val2017.json'
output_json_path = '/home/ailab/qqqcy/DroneVehicle/instances_val2017.json'

with open(input_json_path, 'r') as f:
    coco_data = json.load(f)

# 获取所有带有标注的图像 ID
annotated_img_ids = {ann['image_id'] for ann in coco_data['annotations']}

# 过滤掉没有标注的图像
cleaned_images = [img for img in coco_data['images'] if img['id'] in annotated_img_ids]

# 更新 images 字段
coco_data['images'] = cleaned_images

# 打印删除了多少没有标注的图像
print(f"Removed {len(coco_data['images']) - len(cleaned_images)} images without annotations.")

# 保存清理后的 JSON 文件
with open(output_json_path, 'w') as f:
    json.dump(coco_data, f)

print(f"Cleaned JSON saved to {output_json_path}")
