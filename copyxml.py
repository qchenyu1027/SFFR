import os
import shutil
from pathlib import Path


def copy_matching_xml(image_folder, xml_folder, output_folder):
    """
    将与图片对应的XML文件复制到输出文件夹

    Parameters:
        image_folder (str): 图片所在文件夹路径
        xml_folder (str): XML文件所在文件夹路径
        output_folder (str): 输出文件夹路径
    """
    # 创建输出文件夹(如果不存在)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件名(不含扩展名)
    image_names = {
        Path(f).stem
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    }

    # 遍历XML文件夹
    matched_count = 0
    for xml_file in os.listdir(xml_folder):
        if xml_file.lower().endswith('.xml'):
            xml_name = Path(xml_file).stem

            # 检查是否存在对应的图片文件名
            if xml_name in image_names:
                # 构建源文件和目标文件的完整路径
                src_path = os.path.join(xml_folder, xml_file)
                dst_path = os.path.join(output_folder, xml_file)

                # 复制文件
                shutil.copy2(src_path, dst_path)
                matched_count += 1

    return matched_count


# 使用示例
if __name__ == "__main__":
    # 设置相关文件夹路径
    image_folder = "/media/ailab/EXTERNAL_USB/LLVIP/visible/test"  # 替换为实际的图片文件夹路径
    xml_folder = "/media/ailab/EXTERNAL_USB/LLVIP/Annotations"  # 替换为实际的XML文件夹路径
    output_folder = "/media/ailab/EXTERNAL_USB/LLVIP/labels/test/xml"  # 替换为实际的输出文件夹路径

    # 执行复制操作
    matched_files = copy_matching_xml(image_folder, xml_folder, output_folder)
    print(f"成功复制了 {matched_files} 个匹配的XML文件到输出文件夹")