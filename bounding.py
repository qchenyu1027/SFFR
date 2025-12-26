import cv2
import os
import os.path


def select(text_path, outpath, image_path):
    # 打开完整的路径文件
    with open(text_path, 'r') as text_file:
        line = text_file.readline()
        text_tables = []
        i = 0
        # 逐行读取文本文件
        while line:
            text_data = eval(line)  # 将文本行转换为Python对象
            text_tables.append(text_data)
            line = text_file.readline()  # 读取下一行

    # 遍历图片目录
    for filename in os.listdir(image_path):
        # 只处理图片文件，跳过其他文件
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_path, filename)
            img = cv2.imread(img_path)  # 使用 os.path.join 拼接路径

            # 检查图像是否成功读取
            if img is None:
                print(f"Warning: Could not read image {img_path}, skipping this file.")
                continue  # 跳过该图像文件

            # j = len(text_tables)
            if i < len(text_tables):
                # 从文本数据中提取坐标和宽高
                x1 = int(text_tables[i][0])
                y1 = int(text_tables[i][1])
                w = int(text_tables[i][2])
                h = int(text_tables[i][3])
                x2 = x1 + w
                y2 = y1 + h
                # 在图片上绘制矩形框
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
                cv2.putText(img, 'gabion', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)
                i = i + 1
            # 保存带有矩形框的图片
            output_img_path = os.path.join(outpath, filename + '.png')
            cv2.imwrite(output_img_path, img)
        else:
            print(f"Skipping non-image file: {filename}")


if __name__ == '__main__':

    image_path = r'/home/ailab/qqqcy/Anti-UAV-RGBT/visible/valvisible/images/'
    txt_path = r'/home/ailab/qqqcy/ICAFusion-main/runs/train/exp44/labels/pred'
    outpath = r'/home/ailab/qqqcy/valbounding'
    for textpath in os.listdir(txt_path):
        full_text_path = os.path.join(txt_path, textpath)
        select(full_text_path, outpath, image_path)
