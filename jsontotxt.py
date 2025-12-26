"""
# File       : json_to_txt.py
# Time       ：2022/10/28 16:57
# Author     ：qch
# version    ：python 3.6
# Description：
"""
import os
import json

json_dir = r'/media/ailab/EXTERNAL_USB/sea_drones_see_multispectral/annotations/instances_train.json'  # json文件路径
out_dir = r'/media/ailab/EXTERNAL_USB/sea_drones_see_multispectral/images/processed/'  # 输出的 txt 文件路径


def main():
    # 读取 json 文件数据
    with open(json_dir, 'r') as load_f:
        content = json.load(load_f)
    for t in content['annotations']:
        tmp = t['image_id']

        filename = out_dir + str(tmp) + '.txt'
        if not os.path.exists(filename):
            fp = open(filename, mode="w", encoding="utf-8")
            fp.close()
        x = t['bbox'][0]
        y = t['bbox'][1]
        w = t['bbox'][2]
        h = t['bbox'][3]

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y
        x3 = x + w
        y3 = y + h
        x4 = x
        y4 = y + h

        fp = open(filename, mode="r+", encoding="utf-8")

        file_str = str(t['category_id']) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(
            y2) + ' ' + str(x3) + ' ' + str(y3) + ' ' + str(x4) + ' ' + str(y4)
        line_data = fp.readlines()

        if len(line_data) != 0:
            fp.write('\n' + file_str)
        else:
            fp.write(file_str)
        fp.close()


if __name__ == '__main__':
    main()