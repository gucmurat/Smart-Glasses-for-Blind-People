from ultralytics import YOLO
import cv2
import torch
import time
import numpy as np
from transformers import pipeline
from PIL import Image

yolov8_path = '../../models/object-detection/v1/yolov8m.pt'
wotr_path = '../../models/object-detection/v3/best_v3.pt'

model_yolov8 = None
model_wotr = None
model_midas = None
model_captioning = None
transform_midas = None
device_midas = None

def initialize_models():
    global model_yolov8
    global model_wotr
    global model_midas
    global model_captioning
    global transform_midas
    global device_midas

    model_yolov8 = YOLO(yolov8_path)
    
    model_wotr = YOLO(wotr_path)
    
    model_type = "MiDaS_small" 
    model_midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device_midas = torch.device("mps") if torch.cuda.is_available() else torch.device("cpu")
    model_midas.to(device_midas)
    model_midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform_midas = midas_transforms.dpt_transform
    else:
        transform_midas = midas_transforms.small_transform

    model_captioning = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

def detected_labels_and_boxes_result(model, image):
    results = model.predict(image)
    result = results[0]
    return {"classes": result.boxes.cls, "boxes": result.boxes.xyxy, "confs": result.boxes.conf}

def depth_map_result(image):
    if isinstance(image, str):
        input_image = cv2.imread(image)
    else:
        input_image = image
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = transform_midas(input_image).to(device_midas)

    with torch.no_grad():
        depth_prediction = model_midas(input_image)

    depth_map = depth_prediction.squeeze().cpu().numpy()

    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
    depth_map = depth_map.astype(np.uint8)
    return depth_map

def image_captioning_result(image):
    if not isinstance(image, str):
        image = Image.fromarray(np.uint8(image)).convert('RGB')
    return model_captioning(image)

def stereo_vision_distance_result(image_left, image_right, labels_boxes_json):
    classes = labels_boxes_json["classes"]
    boxes = labels_boxes_json["boxes"]
    # TODO 
    # returns distances dict for each boxes
    # example output 
    # {"classes": result.boxes.cls, "boxes": result.boxes.xyxy, "confs": result.boxes.conf, "distances": distances}
    pass

# example usage: get_model_output_from_camera(image_captioning_result, printable=True)
def get_model_output_from_camera(model_method, show=False, printable=False):
    cap = cv2.VideoCapture(0) 
    while True:
        ret, frame = cap.read()  
        if not ret:
            break

        result = model_method(frame)
        if printable:
            print(result)
        if show:
            cv2.imshow("output", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()

def get_model_output_from_frame(model_method, frame ,show=False, printable=False):
    result = model_method(frame)
    if printable:
        print(result)
    if show:
        cv2.imshow("output", result)
    cv2.destroyAllWindows()
