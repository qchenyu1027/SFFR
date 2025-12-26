# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging,check_version
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation


except ImportError:
    thop = None



class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    # save_path="/home/ailab/qqqcy/valbounding/boundingbox.txt"
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    import os
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    # class Detect(nn.Module):
    #     stride = None  # strides computed during build
    #     onnx_dynamic = False  # ONNX export parameter
    #
    #     def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
    #         super().__init__()
    #         self.nc = nc  # number of classes
    #         self.no = nc + 5  # number of outputs per anchor
    #         self.nl = len(anchors)  # number of detection layers
    #         self.na = len(anchors[0]) // 2  # number of anchors
    #         self.grid = [torch.zeros(1)] * self.nl  # init grid
    #         self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
    #         self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
    #         self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
    #         self.inplace = inplace  # use in-place ops (e.g. slice assignment)
    #
    #     def forward(self, x):
    #         z = []  # inference output
    #         for i in range(self.nl):
    #             x[i] = self.m[i](x[i])  # conv
    #             bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
    #             x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
    #
    #             if not self.training:  # inference
    #                 if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
    #                     self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
    #
    #                 y = x[i].sigmoid()
    #                 if self.inplace:
    #                     y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
    #                     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
    #                 else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
    #                     xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
    #                     wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
    #                     y = torch.cat((xy, wh, y[..., 4:]), -1)
    #                 z.append(y.view(bs, -1, self.no))
    #
    #         return x if self.training else (torch.cat(z, 1), x)
    #
    #     def _make_grid(self, nx=20, ny=20, i=0):
    #         d = self.anchors[i].device
    #         if check_version(torch.__version__,
    #                          '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
    #             # yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
    #             yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
    #         else:
    #             yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
    #         grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
    #         anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
    #             .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
    #         return grid, anchor_grid
    #


    # def forward(self, x, save_path):
    #     z = []  # inference output
    #     self.training |= self.export  # 判断是否处于导出模式
    #     bounding_boxes = []  # 用于存储每个图片的bounding box信息
    #
    #     for i in range(self.nl):
    #         x[i] = self.m[i](x[i])  # 卷积操作
    #         bs, _, ny, nx = x[i].shape  # 获取特征图维度 (batch_size, channels, height, width)
    #         x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
    #
    #         if not self.training:  # 推理模式
    #             if self.grid[i].shape[2:4] != x[i].shape[2:4]:
    #                 self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
    #
    #             y = x[i].sigmoid()
    #             y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy 坐标
    #             y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
    #             z.append(y.view(bs, -1, self.no))  # 添加到输出列表中
    #
    #             # 提取并保存bounding box
    #             for batch in range(bs):
    #                 for anchor in range(y.shape[1]):
    #                     pred = y[batch, anchor, :]  # 每个预测
    #                     x_center, y_center, width, height = pred[:4]  # 提取x, y, w, h
    #                     class_scores = pred[5:]  # 类别置信度
    #                     class_id = class_scores.argmax()  # 预测的类别ID
    #
    #                     # 只保存置信度大于阈值的结果（可选）
    #                     if pred[4] > 0.5:  # 如果需要，可以根据置信度过滤
    #                         bounding_boxes.append([x_center.item(), y_center.item(), width.item(), height.item(),
    #                                                class_id.item()])
    #
    #     # 保存bounding box信息到文本文件
    #     if save_path and len(bounding_boxes) > 0:
    #         with open(save_path, 'w') as f:
    #             for box in bounding_boxes:
    #                 # 格式: x_center, y_center, width, height, class_id
    #                 f.write(' '.join(map(str, box)) + '\n')
    #
    #     return x if self.training else (torch.cat(z, 1), x)




class CLLA(nn.Module):
    def __init__(self, range, c):
        super().__init__()
        self.c_ = c
        self.q = nn.Linear(self.c_, self.c_)
        self.k = nn.Linear(self.c_, self.c_)
        self.v = nn.Linear(self.c_, self.c_)
        self.range = range
        self.attend = nn.Softmax(dim = -1)

    def forward(self, x1, x2):
        b1, c1, w1, h1 = x1.shape
        b2, c2, w2, h2 = x2.shape
        assert b1 == b2 and c1 == c2

        x2_ = x2.permute(0, 2, 3, 1).contiguous().unsqueeze(3)
        pad = int(self.range / 2 - 1)
        padding = nn.ZeroPad2d(padding=(pad, pad, pad, pad))
        x1 = padding(x1)

        local = []
        for i in range(int(self.range)):
            for j in range(int(self.range)):
                tem = x1
                tem = tem[..., i::2, j::2][..., :w2, :h2].contiguous().unsqueeze(2)
                local.append(tem)
        local = torch.cat(local, 2)

        x1 = local.permute(0, 3, 4, 2, 1)

        q = self.q(x2_)
        k, v = self.k(x1), self.v(x1)

        dots = torch.sum(q * k / self.range, 4)
        irr = torch.mean(dots, 3).unsqueeze(3) * 2 - dots
        att = self.attend(irr)

        out = v * att.unsqueeze(4)
        out = torch.sum(out, 3)
        out = out.squeeze(3).permute(0, 3, 1, 2).contiguous()
        # x2 = x2.squeeze(3).permute(0, 3, 1, 2).contiguous()
        return (out + x2) / 2
        # return out

class CLLABlock(nn.Module):
    def __init__(self, range=2, ch=256, ch1=128, ch2=256, out=0):
        super().__init__()
        self.range = range
        self.c_ = ch
        self.cout = out
        self.conv1 = nn.Conv2d(ch1, self.c_, 1)
        self.conv2 = nn.Conv2d(ch2, self.c_, 1)

        self.att = CLLA(range = range, c = self.c_)

        self.det = nn.Conv2d(self.c_, out, 1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        f = self.att(x1, x2)

        return self.det(f)


class CLLADetect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.det = CLLABlock(range = 2, ch = ch[0], ch1 = ch[0], ch2 = ch[1], out = self.no * self.na)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[2:])  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        p = []
        for i in range(self.nl):
            if i == 0:
                p.append(self.det(x[0], x[1]))
            else:
                p.append(self.m[i-1](x[i+1]))  # conv
            bs, _, ny, nx = p[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)

            p[i] = p[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != p[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = p[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
        return p if self.training else (torch.cat(z, 1), p)
        # return x if self.training else (z[0], x) # (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict

        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict
            print("YAML")
            print(self.yaml)

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        # print("Detect")
        # print(m)
        if isinstance(m, Detect)or isinstance(m, CLLADetect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            # print("m.stride", m.stride)
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # logger.info('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            logger.info('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             logger.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        logger.info('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR, C3STR, C3SPP, C3Ghost,ASPP, CBAM, nn.ConvTranspose2d]:
            # c1, c2 = ch[f], args[0]
            # if c2 != no:  # if not output
            #     c2 = make_divisible(c2 * gw, 8)
            #
            # args = [c1, c2, *args[1:]]
            # if m in [BottleneckCSP, C3, C3TR,C3STR,C3Ghost]:
            #     args.insert(2, n)  # number of repeats
            #     n = 1

            if m is Focus:

                c1, c2 = 3, args[0]

                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
            else:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, C3, C3TR, C3STR, C3Ghost]:
                    args.insert(2, n)  # number of repeats
                    n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Add:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            args = [c2]
        elif m is Add2:
            c2 = ch[f[0]]
            args = [c2, args[1]]
        # elif m is CMAFF:
        #     c2 = ch[f[0]]
        #     args = [c2]
        # elif m is GPT:
        #     c2 = ch[f[0]]
        #     args = [c2]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is CLLADetect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * (len(f) - 1)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is NiNfusion:
            c1 = sum([ch[x] for x in f])
            c2 = c1 // 2
            args = [c1, c2, *args]
        # elif m is DysnakeConvFusionBlock:
        #     c2 = ch[f[0]]
        #     args = [c2, *args[1:]]
        elif m is TransformerFusionBlock:
            c2 = ch[f[0]]
            args = [c2, *args[1:]]
        elif m is KanConvFusionBlock:
            c2 = ch[f[0]]
            args = [c2, *args[1:]]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
from thop import profile

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='/home/ailab/qqqcy/ICAFusion-main/models/yolov5l-CLLADetect.yaml', help='model.yaml')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    input_rgb = torch.Tensor(8, 3, 640, 640).to(device)

    flops, params = profile(model, (input_rgb,))
    print(f'params: {params}, GFLOPs: {2 * flops}')  # 注意这里的flops要×2，才跟yolo打印出来的值对应得上
    print('params: %.2f M, GFLOPs: %.1f B' % (params / 1e6, 2 * flops / 1e9))
    output = model(input_rgb)
    # print(model)
    # model.train()
    # torch.save(model, "yolov5s.pth")

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 320, 320).to(device)
    # y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # logger.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
