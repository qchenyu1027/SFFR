import os

#+lujing
# def writejpg2txt(images_path, txt_name):
#     # 打开图片列表清单txt文件
#     file_name = open(txt_name, "w")
#     # 将路径改为绝对路径
#     images_path = os.path.abspath(images_path)
#     # 查看文件夹下的图片
#     images_name = os.listdir(images_path)
#
#     count = 0
#     # 遍历所有文件
#     for eachname in images_name:
#         # 按照需要的格式写入目标txt文件
#         file_name.write(os.path.join(images_path, eachname) + '\n')
#         count += 1
#     print('生成txt成功！')
#     print('{} 张图片地址已写入'.format(count))
#     file_name.close()
#
#
# if __name__ == "__main__":
#
#     # 图片存放目录
#     images_path = r'/media/ailab/EXTERNAL_USB/SeeDroneSee/visible/train'
#     # 生成图片txt文件命名
#     txt_name = r'D:/media/ailab/EXTERNAL_USB/SeeDroneSee/path/train_rgb.txt'
#     txt_name = os.path.abspath(txt_name)
#     if not os.path.exists(txt_name):
#         os.system(r"touch {}".format(txt_name))  # 调用系统命令行来创建文件
#     # 将jpg绝对地址写入到txt中
#     writejpg2txt(images_path, txt_name)
#picture name
import os


import os


import os
from natsort import natsorted  # 导入自然排序库

def writejpg2txt(images_path, txt_name):
    with open(txt_name, "w") as file_name:
        # 获取文件夹内的所有文件，并按自然顺序排序
        images_name = natsorted(os.listdir(images_path))

        count = 0
        for eachname in images_name:
            name, ext = os.path.splitext(eachname)  # 分离文件名和扩展名
            if ext.lower() in ['.jpg', '.jpeg', '.png']:  # 仅处理图片格式
                file_name.write(name + '\n')
                count += 1

    print('生成txt成功！')
    print(f'{count} 张图片名称已写入')








if __name__ == "__main__":
    # 图片存放目录
    images_path = r'/media/ailab/5385f1e6-0f46-4865-90c2-287b9a0f3c16/qcy/SeeDroneSee/infrared/train'
    # 生成图片txt文件命名
    txt_name = r'/media/ailab/5385f1e6-0f46-4865-90c2-287b9a0f3c16/qcy/SeeDroneSee/path/train_ir.txt'

    # 确保txt文件存在
    if not os.path.exists(os.path.dirname(txt_name)):
        os.makedirs(os.path.dirname(txt_name))

    # 仅保存图片的文件名
    writejpg2txt(images_path, txt_name)
