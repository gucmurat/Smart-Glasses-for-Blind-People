from ultralytics import YOLO
import cv2
import torch
import time
import numpy as np
from transformers import pipeline
from PIL import Image
import scipy
import hashlib
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../utils/')
import dist_measurement
import direction_detection
import midas_object_detection

yolov8_path = '../../models/object-detection/v1/yolov8m.pt'
wotr_path = '../../models/object-detection/v4/best_v4.pt'

# model should pay attention to these class indexes
wotr_class_pruned_indexes = [0, 1, 2, 3, 4, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24]

# for these indexes, we may not measure distance
wotr_class_important_indexes = [3, 4, 20, 21, 22, 23, 24]

model_yolov8 = None
model_wotr = None
model_midas = None
model_captioning = None
transform_midas = None
device_midas = None

detected_objects = {}

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

def detected_labels_and_boxes_result(model, image, model_type):
    results = None
    if model_type=="wotr":
        results = model.predict(image, classes=wotr_class_pruned_indexes)
        result = results[0]
    elif model_type=="yolo":
        results = model.predict(image)
        result = results[0]
    return {'names': result.names, "classes": result.boxes.cls, "boxes": result.boxes.xyxy, "confs": result.boxes.conf}

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
    return midas_object_detection.detect_boxes_with_mask_midas(depth_map,image)

def update_detected_labels_and_boxes_with_midas(detected_yolo_left, detected_yolo_right, image_left, image_right):
    next_class = max(detected_yolo_left['names'].keys()) + 1
    detected_yolo_left['names'][next_class] = 'undetected obstacle'
    detected_yolo_right['names'][next_class] = 'undetected obstacle'
    
    depth_map_boxes_left = depth_map_result(image_left)
    depth_map_boxes_right = depth_map_result(image_right)
    
    detected_midas_left = midas_object_detection.compare_detections_from_alg_midas(depth_map_boxes_left,detected_yolo_left,image_left)
    detected_midas_right = midas_object_detection.compare_detections_from_alg_midas(depth_map_boxes_right,detected_yolo_right,image_right)
    
    for detected_box in detected_midas_left:
        new_box = torch.tensor(detected_box)
        detected_yolo_left['classes'] = torch.cat((detected_yolo_left['classes'], torch.tensor([float(next_class)])))
        detected_yolo_left['boxes'] = torch.cat((detected_yolo_left['boxes'],  new_box.unsqueeze(0)))
        detected_yolo_left['confs'] = torch.cat((detected_yolo_left['confs'], torch.tensor([1.0]))) 
    for detected_box in detected_midas_right:
        new_box = torch.tensor(detected_box)
        detected_yolo_right['classes'] = torch.cat((detected_yolo_right['classes'], torch.tensor([float(next_class)])))
        detected_yolo_right['boxes'] = torch.cat((detected_yolo_right['boxes'], new_box.unsqueeze(0)))
        detected_yolo_right['confs'] = torch.cat((detected_yolo_right['confs'], torch.tensor([1.0])))  
        
    return detected_yolo_left, detected_yolo_right
    
def image_captioning_result(image):
    if not isinstance(image, str):
        image = Image.fromarray(np.uint8(image)).convert('RGB')
    return model_captioning(image)

def get_distance_values_from_objects(image_left, image_right):
    detected_yolo_left, detected_yolo_right = update_detected_labels_and_boxes_with_midas(detected_labels_and_boxes_result(model_yolov8,image_left, "yolo"), detected_labels_and_boxes_result(model_yolov8,image_right, "yolo"), image_left, image_right)
    result_from_yolov8 = stereo_vision_distance_result(image_left, 
                                           image_right, detected_yolo_left, detected_yolo_right)
    result_from_wotr = stereo_vision_distance_result(image_left, 
                                           image_right, 
                              detected_labels_and_boxes_result(model_wotr,image_left, "wotr"), 
                              detected_labels_and_boxes_result(model_wotr,image_right, "wotr"))
    return result_from_yolov8 + result_from_wotr
    
def stereo_vision_distance_result(image_left, image_right, labels_boxes_json_left, labels_boxes_json_right):   
    # TODO 
    # returns distances dict for each boxes
    # example output 
    # [[dist, direction, class_name],[60.0, 12, 'cup'],...]
    image_left_initial = image_left
    image_right_initial = image_right
    image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
    image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)
    
    class_to_tensors = match_and_eliminate_detection_results(labels_boxes_json_left, labels_boxes_json_right)
    
    dist_direction_class_list = []
    for tuple in class_to_tensors:
        dists_away, det = dist_measurement.measure_dist(image_left, image_right, {"classes": tuple[0], "boxes": tuple[1][0]}, {"classes": tuple[0], "boxes": tuple[1][1]})
        d = dists_away[0][0]    
        c = dists_away[0][1]
        # handle wotr_class_important_indexes 
        if c in wotr_class_important_indexes:
            direction = direction_detection.get_object_direction(image_left, image_right, tuple[1])
            obj = [-1,direction,labels_boxes_json_left["names"][c]]
            obj.append(obj_to_sha256(obj))
            dist_direction_class_list.append(obj)
        # eliminate negative distance measurements
        if d<0:
            continue
        # eliminate undetected objects that have distance higher than 100 cm 
        if c==80 and d>100:
            continue
        direction = direction_detection.get_object_direction(image_left, image_right, tuple[1])
        obj = [d,direction,labels_boxes_json_left["names"][c]]
        obj.append(obj_to_sha256(obj))
        dist_direction_class_list.append(obj)
        
    return dist_direction_class_list


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

def match_and_eliminate_detection_results(labels_boxes_json_left, labels_boxes_json_right):
    class_to_tensors = []
    
    threshold = 0.60
    
    left_classes = labels_boxes_json_left["classes"].numpy()
    right_classes = labels_boxes_json_right["classes"].numpy()
    left_boxes = labels_boxes_json_left["boxes"]
    right_boxes = labels_boxes_json_right["boxes"]
    left_confs = labels_boxes_json_left["confs"].numpy()
    right_confs = labels_boxes_json_right["confs"].numpy()
    
    for c in left_classes:
        if c in right_classes:        
            index_left = np.where(left_classes == c)[0][0]
            index_right = np.where(right_classes == c)[0][0]
            
            avg = (left_confs[index_left] + right_confs[index_right]) / 2
            
            if avg < threshold:
                continue
            
            key = int(left_classes[index_left:index_left+1][0])
            class_to_tensors.append((torch.from_numpy(np.array([key])), [left_boxes[index_left:index_left+1, :],right_boxes[index_right:index_right+1, :]]))
            
            left_classes[index_left] = -1
            right_classes[index_right] = -1
            
        else:
            continue
    return class_to_tensors

def result_to_sentence(input_list):
    # input: [[dist, direction, class_name],[60.0, 12, 'cup'],...]
    sentence = ""
    
    # input list is sorted to tell nearest objects first. 
    # sentence will contain at most 3 objects.
    input_list = sorted(input_list, key=lambda x: x[0])
    #if len(input_list)>4:
    #    input_list=input_list[:4]
        
    obj_ids = []
    for obj in input_list:
        obj_id = obj[3]
        obj_ids.append(obj_id)
        if obj_id not in detected_objects:
            detected_objects[obj_id] = obj
            if obj[0] == -1:
                sentence += f"{obj[2]} is detected, on the direction of {obj[1]} oclock. "
            else:
                sentence += f"{obj[2]} is detected {int(obj[0])} cm away, on the direction of {obj[1]} oclock. "
        else:
            continue
    for key, value in dict(detected_objects).items():
        if key not in obj_ids:
            removed_value = detected_objects.pop(key)
    return sentence
            
def obj_to_sha256(obj):
    # input: [[dist, direction, class_name],[60.0, 12, 'cup'],...]
    m = hashlib.sha256()
    dist_cut = int(obj[0]/40)
    m.update(dist_cut.to_bytes(2, 'big'))
    m.update(obj[1].to_bytes(2, 'big'))
    m.digest()
    # output = '27cf47520c67cb4949dd5f6796529562f1800de86e339970e9cad9dddd0e2ab8'
    return str(m.hexdigest())