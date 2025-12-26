import os
import shutil
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def copy_single_file(src, dest):
    shutil.copy(src, dest)


def split_multispectral_data(ir_path, vis_path, labels_path, new_file_path, train_rate=0.8, num_workers=8):
    # 获取所有文件名并确保它们有相同的基础名称
    ir_images = sorted([f for f in os.listdir(ir_path) if f.endswith(('.jpg', '.png', '.tif'))])
    vis_images = sorted([f for f in os.listdir(vis_path) if f.endswith(('.jpg', '.png', '.tif'))])
    labels = sorted([f for f in os.listdir(labels_path) if f.endswith('.txt')])

    # 提取基础文件名（不含扩展名）用于匹配
    ir_bases = [os.path.splitext(f)[0] for f in ir_images]
    vis_bases = [os.path.splitext(f)[0] for f in vis_images]
    label_bases = [os.path.splitext(f)[0] for f in labels]

    # 验证所有文件是否匹配
    assert set(ir_bases) == set(vis_bases) == set(label_bases), "文件名不匹配，请确保IR、VIS和标签文件具有相同的基础名称"

    # 使用基础文件名创建对应关系
    file_groups = []
    for base_name in ir_bases:
        ir_file = next(f for f in ir_images if os.path.splitext(f)[0] == base_name)
        vis_file = next(f for f in vis_images if os.path.splitext(f)[0] == base_name)
        label_file = next(f for f in labels if os.path.splitext(f)[0] == base_name)
        file_groups.append((ir_file, vis_file, label_file))

    # 设置随机种子确保可重现性
    random.seed(0)
    # 随机打乱文件组
    random.shuffle(file_groups)

    # 计算训练集大小
    total = len(file_groups)
    train_size = int(total * train_rate)

    # 划分训练集和验证集
    train_groups = file_groups[:train_size]
    val_groups = file_groups[train_size:]

    def copy_file_groups(groups, mode):
        # 创建目标目录
        ir_dir = os.path.join(new_file_path, 'infrared', mode)
        vis_dir = os.path.join(new_file_path, 'visible', mode)
        labels_dir = os.path.join(new_file_path, 'labels', mode)

        for dir_path in [ir_dir, vis_dir, labels_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # 创建线程池并并行复制文件
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for ir_file, vis_file, label_file in groups:
                futures.append(
                    executor.submit(copy_single_file, os.path.join(ir_path, ir_file), os.path.join(ir_dir, ir_file)))
                futures.append(executor.submit(copy_single_file, os.path.join(vis_path, vis_file),
                                               os.path.join(vis_dir, vis_file)))
                futures.append(executor.submit(copy_single_file, os.path.join(labels_path, label_file),
                                               os.path.join(labels_dir, label_file)))

            # 等待所有线程完成
            for future in tqdm(futures, desc=f"复制{mode}集", unit="文件", ncols=100):
                future.result()

    # 复制训练集和验证集
    copy_file_groups(train_groups, 'train')
    copy_file_groups(val_groups, 'val')

    # 打印数据集划分信息
    print(f"数据集划分完成：")
    print(f"训练集：{len(train_groups)}组样本 ({train_rate * 100}%)")
    print(f"验证集：{len(val_groups)}组样本 ({(1 - train_rate) * 100}%)")


if __name__ == '__main__':
    # 定义数据路径
    ir_path = "/media/ailab/EXTERNAL_USB/M3FD_Detection1/ir"
    vis_path = "/media/ailab/EXTERNAL_USB/M3FD_Detection1/vi"
    labels_path = "/media/ailab/EXTERNAL_USB/M3FD_Detection1/labels"
    new_file_path = "/media/ailab/EXTERNAL_USB/M3FD_Detection"

    split_multispectral_data(
        ir_path=ir_path,
        vis_path=vis_path,
        labels_path=labels_path,
        new_file_path=new_file_path,
        train_rate=0.8,
        num_workers=8  # Use 8 threads for parallel copying
    )
