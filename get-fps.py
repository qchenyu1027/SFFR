import warnings
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
warnings.filterwarnings('ignore')
import argparse
import logging
import math
import os
import random
import time
import sys
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from thop import profile
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.tasks import attempt_load_weights


def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str, default='/media/ailab/5385f1e6-0f46-4865-90c2-287b9a0f3c16/qcy/sea/exp22/weights/best.pt',help='trained weights path')
    parser.add_argument('--weights', type=str, default='/media/ailab/5385f1e6-0f46-4865-90c2-287b9a0f3c16/qcy/latest/sea/exp30/weights/best.pt',help='trained weights path')
    parser.add_argument('--batch', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--imgs', nargs='+', type=int, default=[640, 640], help='[height, width] image sizes')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--warmup', default=20, type=int, help='warmup time')
    parser.add_argument('--testtime', default=60, type=int, help='test time')
    parser.add_argument('--half', action='store_true', default=False, help='fp16 mode.')
    parser.add_argument('--dual-channel', action='store_true', default=True, help='dual channel mode (IR + visible)')
    parser.add_argument('--channel-mode', type=str, default='concat', choices=['concat', 'separate'],
                        help='dual channel mode: concat (6 channels) or separate (two 3-channel inputs)')
    opt = parser.parse_args()

    device = select_device(opt.device, batch=opt.batch)

    weights = opt.weights
    if weights.endswith('.pt'):
        model = attempt_load_weights(weights, device=device, fuse=True)
        print(f'Loaded {weights}')  # report
    else:
        model = YOLO(weights).model
        model.fuse()

    model = model.to(device)
    # example_inputs = torch.randn((opt.batch, 3, *opt.imgs)).to(device)
    input_rgb = torch.randn((opt.batch, 3, 640, 640),device=device)
    input_ir = torch.randn((opt.batch, 3, 640, 640),device=device)
    if opt.half:
        model = model.half()
        input_rgb = input_rgb.half()
        input_ir = input_ir.half()

    print('begin warmup...')
    for i in tqdm(range(opt.warmup), desc='warmup....'):
        model(input_rgb,input_ir)

    print('begin test latency...')
    time_arr = []

    from ultralytics.utils.torch_utils import model_info
    # n_l, n_p, n_g, flops = model_info(model.model)
    n_l, n_p, n_g, _ = model_info(model)


    # 计算 GFLOPS
    # flops, params =profile(model, (input_rgb, input_ir), verbose=False)
    # Gflops = 2 * flops / 1e9  # 计算双精度 GFLOPS
    # #
    # print(f'GFLOPs: {Gflops:.2f}')  # 输出参数量和 GFLOPS
    # print(f'params: {n_p}')

    # print(f'{flops:.1f},{n_p:,}')

    # flops, params = profile(model, (input_rgb,input_ir),verbose=False)
    # Gflops = 2*flops/1e9
    for i in tqdm(range(opt.testtime), desc='test latency....'):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()

        model(input_rgb,input_ir)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        time_arr.append(end_time - start_time)

    # std_time = np.std(time_arr)
    # infer_time_per_image = np.sum(time_arr) / (opt.testtime * opt.batch)
    std_time = np.std(time_arr)
    # 计算每张图片的平均推理时间
    infer_time_per_image = np.sum(time_arr) / (opt.testtime * opt.batch)
    # 计算 FPS
    fps = 1 / infer_time_per_image  # FPS 计算公式
    print(f"Inference time per image: {infer_time_per_image:.4f} s")
    print(f"FPS: {fps:.1f}")

    # print(f"refer",infer_time_per_image)
    if weights.endswith('.pt'):
        print(
            f'model weights:{opt.weights} size:{get_weight_size(opt.weights)}M  fps:{1 / infer_time_per_image:.1f}')
    else:
        print(
            f'model yaml:{opt.weights} (bs:{opt.batch})Latency:{infer_time_per_image:.5f}s +- {std_time:.5f}s fps:{1 / infer_time_per_image:.1f}')