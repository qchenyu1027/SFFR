import os
import time
import argparse
import numpy as np
import cv2
import torch
from torchvision.transforms import functional as F
from models.gradcam import YOLOV5GradCAM
from models.yolo_test import Model  # 替代 YOLOV5TorchObjectDetector

# target = ['model_30_cv1_act', 'model_30_cv2_act', 'model_30_cv3_act',
#           'model_33_cv1_act', 'model_33_cv2_act', 'model_33_cv3_act',
#           'model_36_cv1_act', 'model_36_cv2_act', 'model_36_cv3_act']

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default='runs/train/LLVIP/exp3/weights/best.pt', help='Path to the model')
parser.add_argument('--source1', type=str, default='/home/ailab/FLIR-align-3class/visible/test', help='source')  # file/folder, 0 for webcam
parser.add_argument('--source2', type=str, default='/home/ailab/FLIR-align-3class/infrared/test', help='source')  # file/folder, 0 for webcam
parser.add_argument('--output-dir', type=str, default='/home/ailab/FLIR-align-3class/outputs', help='output dir')
parser.add_argument('--img-size', type=int, default=640, help="input image size")
parser.add_argument('--target-layer', type=str, default=[20,21,22],
                    help='The layer hierarchical address to which gradcam will applied,'
                         ' the names should be separated by underline')
parser.add_argument('--method', type=str, default='gradcam', help='gradcam or gradcampp')
parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
parser.add_argument('--names', type=str, default='person',
                    help='The name of the classes. The default is set to None and is set to coco classes. Provide your custom names as follow: object1,object2,object3')
# 'person, car, bicycle'
args = parser.parse_args()


def preprocess_image(img, img_size):
    """Resize and normalize image for YOLO input."""
    img_resized = cv2.resize(img, (img_size, img_size))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0  # HWC -> CHW, normalize
    img_tensor = F.pad(img_tensor, [0, max(0, img_size - img.shape[1]), 0, max(0, img_size - img.shape[0])])
    return img_tensor.unsqueeze(0)  # Add batch dimension


def draw_bbox(img, bbox, color=(0, 255, 0), thickness=2):
    """Draw bounding box on an image."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def split_extension(file_path, suffix=""):
    """Split the file name and add a suffix before the extension."""
    base, ext = os.path.splitext(file_path)
    return f"{base}{suffix}{ext}"


def main(img_vis_path, img_ir_path):
    device = args.device
    input_size = args.img_size

    # Load visible and infrared images
    img_vis, img_ir = cv2.imread(img_vis_path), cv2.imread(img_ir_path)

    print('[INFO] Loading the model')
    # Load YOLOv5 model
    model = Model(cfg='yolov5s.yaml').to(device)  # Load YOLOv5 model structure (ensure to match the pretrained model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))  # Load weights
    model.eval()

    # Preprocess images
    torch_img_vis = preprocess_image(img_vis, input_size).to(device)
    torch_img_ir = preprocess_image(img_ir, input_size).to(device)

    result = img_vis.astype(np.float32) / 255.0  # Normalize for visualization
    images = []

    if args.method == 'gradcam':
        for layer in args.target_layer:
            saliency_method = YOLOV5GradCAM(model=model, layer_name=layer, img_size=(input_size, input_size))
            tic = time.time()
            masks, logits, [boxes, _, class_names, confs] = saliency_method(torch_img_vis, torch_img_ir)
            print("total time:", round(time.time() - tic, 4))
            res_img = result.copy()
            heat = []
            for i, mask in enumerate(masks):
                bbox = boxes[0][i]
                mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
                heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                n_heatmat = (heatmap / 255).astype(np.float32)
                heat.append(n_heatmat)

            if len(heat) != 0:
                heat_all = heat[0]
                for h in heat[1:]:
                    heat_all += h
                heat_avg = heat_all / len(heat)
                res_img = cv2.addWeighted(res_img, 0.3, heat_avg, 0.7, 0)
            res_img = (res_img / res_img.max())
            cv2.imwrite('temp.jpg', (res_img * 255).astype(np.uint8))
            heat_map = cv2.imread('temp.jpg')
            final_image = heat_map
            images.append(final_image)
            # Save individual layer results
            suffix = '-res-' + layer
            img_name = split_extension(os.path.split(img_vis_path)[-1], suffix=suffix)
            output_path = f'{args.output_dir}/{img_name}'
            os.makedirs(args.output_dir, exist_ok=True)
            print(f'[INFO] Saving the final image at {output_path}')
            cv2.imwrite(output_path, final_image)

        # Save averaged results
        img_name = split_extension(os.path.split(img_vis_path)[-1], suffix='_avg')
        output_path = f'{args.output_dir}/{img_name}'
        img_all = images[0].astype(np.uint16)
        for img in images[1:]:
            img_all += img
        img_avg = img_all / len(images)
        cv2.imwrite(output_path, img_avg.astype(np.uint8))


if __name__ == '__main__':
    if os.path.isdir(args.source1):
        img_vis_list = os.listdir(args.source1)
        img_vis_list.sort()
        for item in img_vis_list[1127:]:
            img_vis_path = os.path.join(args.source1, item)
            if args.source1 == '/home/ailab/FLIR-align-3class/visible/test':
                new_item = item[:-4] + '.jpeg'
                img_ir_path = os.path.join(args.source2, new_item)
            else:
                img_ir_path = os.path.join(args.source2, item)
            main(img_vis_path, img_ir_path)
            print(item)
    else:
        main(args.source1, args.source2)
