import cv2
import os


# def load_yolo_labels(txt_path, img_shape):
#     """
#     读取 YOLO 格式的 GT 标签，并转换为 (x1, y1, x2, y2)
#     :param txt_path: YOLO 格式的标签文件路径
#     :param img_shape: (H, W) 图像大小
#     :return: bbox_list
#     """
#     h, w = img_shape
#     bboxes = []
#
#     if not os.path.exists(txt_path):
#         print(f"[WARNING] 标签文件 {txt_path} 不存在")
#         return []
#
#     with open(txt_path, "r") as f:
#         lines = f.readlines()
#
#     for line in lines:
#         values = line.strip().split()
#         if len(values) != 5:
#             continue  # 跳过格式错误的行
#         _, x, y, bw, bh = map(float, values)  # 忽略类别 ID
#
#         # 转换为 (x1, y1, x2, y2) 绝对坐标
#         x1 = int((x - bw / 2) * w)
#         y1 = int((y - bh / 2) * h)
#         x2 = int((x + bw / 2) * w)
#         y2 = int((y + bh / 2) * h)
#
#         bboxes.append([x1, y1, x2, y2])
#
#     return bboxes
#
#
# def draw_groundtruth(img_path, txt_path, output_path):
#     """
#     在图片上绘制 Ground Truth (GT) 并保存（仅绘制框，不加文本）
#     :param img_path: 图片路径
#     :param txt_path: YOLO 标签路径
#     :param output_path: 结果保存路径
#     """
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"[ERROR] 无法读取图像 {img_path}")
#         return
#
#     h, w = img.shape[:2]
#     gt_boxes = load_yolo_labels(txt_path, (h, w))
#
#     for bbox in gt_boxes:
#         x1, y1, x2, y2 = bbox
#         # 画红色 GT 边界框（无文本）
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     cv2.imwrite(output_path, img)
#     print(f"[INFO] Ground Truth 图像已保存: {output_path}")

import os
import cv2

def load_yolo_labels(txt_path, img_shape):
    """
    读取 YOLO 格式的标签并转换为 (x1, y1, x2, y2)
    """
    h, w = img_shape
    bboxes = []

    if not os.path.exists(txt_path):
        print(f"[WARNING] 标签文件 {txt_path} 不存在")
        return []

    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        values = line.strip().split()
        if len(values) != 5:
            continue
        _, x, y, bw, bh = map(float, values)
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        bboxes.append([x1, y1, x2, y2])
    return bboxes

def draw_groundtruth_batch(img_dir, txt_dir, output_dir):
    """
    批量绘制 Ground Truth 边框到图片上并保存
    :param img_dir: 图片文件夹
    :param txt_dir: YOLO 标签文件夹
    :param output_dir: 绘制结果保存路径
    """
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(img_dir):
        if file_name.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            img_path = os.path.join(img_dir, file_name)
            txt_name = os.path.splitext(file_name)[0] + '.txt'
            txt_path = os.path.join(txt_dir, txt_name)
            output_path = os.path.join(output_dir, file_name)

            img = cv2.imread(img_path)
            if img is None:
                print(f"[ERROR] 无法读取图像 {img_path}")
                continue

            h, w = img.shape[:2]
            gt_boxes = load_yolo_labels(txt_path, (h, w))

            for bbox in gt_boxes:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.imwrite(output_path, img)
            print(f"[INFO] 保存成功: {output_path}")
# 示例：绘制 Ground Truth
# img_path = "/media/ailab/5385f1e6-0f46-4865-90c2-287b9a0f3c16/qcy/SeeDroneSee/visible/val/20.jpg"  # 输入图像
img_path = "/home/ailab/qqqcy/FLIR-align-3class/infrared/test"  # 输入图像
txt_path = "/media/ailab/EXTERNAL_USB/latest/FLIR/exp85/labels/pred6"  # YOLO 标签路径
output_path = "/media/ailab/5385f1e6-0f46-4865-90c2-287b9a0f3c16/qcy/sample/flir"  # 结果图像

# draw_groundtruth(img_path, txt_path, output_path)
draw_groundtruth_batch(img_path, txt_path, output_path)




