import warnings
import torch
import yaml
import cv2
import os
import shutil
import sys
import numpy as np
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
np.random.seed(0)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


class ActivationsAndGradients:
    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return

        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        if not self.model.end2end:
            logits_ = result[:, 4:]
            boxes_ = result[:, :4]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
                indices[0]]
        else:
            logits_ = result[:, :, 4:]
            boxes_ = result[:, :, :4]
            sorted, indices = torch.sort(logits_[:, :, 0], descending=True)
            return logits_[0][indices[0]], boxes_[0][indices[0]]

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        # 确保x是包含两个张量的元组/列表
        if not isinstance(x, (tuple, list)):
            x = [x, x]  # 或者根据需要处理第二个输入
        model_output = self.model(*x)  # 解包输入
        post_result, pre_post_boxes = self.post_process(model_output[0])
        return [[post_result, pre_post_boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()


class yolov8_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio, end2end) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
        self.end2end = end2end

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in range(int(post_result.size(0) * self.ratio)):
            if (self.end2end and float(post_result[i, 0]) < self.conf) or (
                    not self.end2end and float(post_result[i].max()) < self.conf):
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                if self.end2end:
                    result.append(post_result[i, 0])
                else:
                    result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)


class MultiModalYOLOHeatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_box, renormalize):
        self.device = torch.device(device)
        ckpt = torch.load(weight)
        self.model_names = ckpt['model'].names
        self.model = attempt_load_weights(weight, self.device)
        self.model.info()
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.eval()

        if not hasattr(self.model, 'end2end'):
            self.model.end2end = False

        self.target = yolov8_target(backward_type, conf_threshold, ratio, self.model.end2end)
        device = torch.device(device)  # 先创建device对象
        self.target_layers = [self.model.model[l] for l in layer]
        self.method = eval(method)(self.model, self.target_layers, use_cuda=device.type == 'cuda')
        self.method.activations_and_grads = ActivationsAndGradients(self.model, self.target_layers, None)

        self.colors = np.random.uniform(0, 255, size=(len(self.model_names), 3)).astype(np.int32)
        self.conf_threshold = conf_threshold
        self.show_box = show_box
        self.renormalize = renormalize

    def post_process(self, result):
        return non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def process_image(self, img_path, save_path, is_thermal=False):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED if is_thermal else cv2.IMREAD_COLOR)

        # Handle thermal images
        if is_thermal:
            # Normalize thermal image to 0-255 range
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # Convert single channel to 3 channels
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        try:
            grayscale_cam = self.method(tensor, [self.target])
        except AttributeError as e:
            print(f"Error processing {img_path}: {e}")
            return

        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        pred = self.model(tensor)[0]
        if not self.model.end2end:
            pred = self.post_process(pred)
        else:
            pred = pred[0][pred[0, :, 4] > self.conf_threshold]

        if self.renormalize:
            cam_image = self.renormalize_cam_in_bounding_boxes(pred[:, :4].cpu().detach().numpy().astype(np.int32), img,
                                                               grayscale_cam)
        if self.show_box:
            for data in pred:
                data = data.cpu().detach().numpy()
                cam_image = self.draw_detections(data[:4], self.colors[int(data[5])],
                                                 f'{self.model_names[int(data[5])]} {float(data[4]):.2f}', cam_image)

        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)

    def __call__(self, rgb_path, ir_path, save_path):
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'ir'), exist_ok=True)

        # Process RGB images
        if os.path.isdir(rgb_path):
            for img_name in os.listdir(rgb_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.process_image(
                        os.path.join(rgb_path, img_name),
                        os.path.join(save_path, 'rgb', img_name),
                        is_thermal=False
                    )

        # Process IR images
        if os.path.isdir(ir_path):
            for img_name in os.listdir(ir_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.process_image(
                        os.path.join(ir_path, img_name),
                        os.path.join(save_path, 'ir', img_name),
                        is_thermal=True
                    )


def get_params():
    return {
        'weight': 'runs/train/LLVIP/exp3/weights/best.pt',  # Your model weights path
        'device': 'cuda:0',
        'method': 'GradCAM',
        # Options: GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
        'layer': [20, 21, 22],  # Target layers for visualization
        'backward_type': 'class',  # Options: class, box, all
        'conf_threshold': 0.2,
        'ratio': 0.02,
        'show_box': True,
        'renormalize': False
    }


if __name__ == '__main__':
    model = MultiModalYOLOHeatmap(**get_params())
    model(
        rgb_path='/media/ailab/EXTERNAL_USB/KAIST/visible/test/set06_V000_I00019.jpg',
        ir_path='/media/ailab/EXTERNAL_USB/KAIST/infrared/test/set06_V000_I00019.jpg',
        save_path='/media/ailab/EXTERNAL_USB/heatmap_results'
    )
    print("ok!!")