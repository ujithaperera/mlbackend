import numpy as np
import os, cv2, torch
from pathlib import Path

from utils.plots import plot_one_box
from utils.datasets import LoadImages
from models.experimental import attempt_load
from utils.torch_utils import select_device, TracedModel
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
##########################################################################################

agnostic_nms=False
save_conf=False
exist_ok=False
save_txt=False
view_img=False
save_img=True
augment=False
update=False
trace = True
nosave=False
classes=None

device='0'
img_size=640
iou_thres=0.45
conf_thres=0.25

save_dir='inference/detected'
weights='weights/spine_detector.pt'

###########################################################################################
set_logging()
device = select_device(device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size

if trace:
    model = TracedModel(model, device, img_size)

if half:
    model.half()  # to FP16
###############################################################################################

def detect(source):
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    bbox_predictions = []
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for _ in range(3):
                model(img, augment=augment)[0]

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        # Process detections
        bbox_prediction = []
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, _ = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            p_dir = p.name.split('.')[0]
            os.makedirs(save_dir + '/' + p_dir, exist_ok=True)  # make dir
            save_path = save_dir + '/' + p_dir + '/' + p.name  # img.jpg
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        bbox_prediction.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)
                print(f" The image with the result is saved in: {save_path}")

    bbox_predictions.append(bbox_prediction) 
    return bbox_predictions