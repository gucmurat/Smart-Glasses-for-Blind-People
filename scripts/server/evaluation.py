from ultralytics import YOLO
import cv2
import torch
import time
import numpy as np
from transformers import pipeline
from PIL import Image
import dist_measurement
import scipy

yolov8_path = '../../models/object-detection/v1/yolov8m.pt'
wotr_path = '../../models/object-detection/v3/best_v3.pt'

# model should pay attention to these class indexes
wotr_class_pruned_indexes = [0, 1, 2, 3, 4, 12, 13, 14, 15, 16, 20, 21]

# for these indexes, we may not measure distance
wotr_class_important_indexes = [3, 4, 20, 21]

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
    return depth_map

def image_captioning_result(image):
    if not isinstance(image, str):
        image = Image.fromarray(np.uint8(image)).convert('RGB')
    return model_captioning(image)

def stereo_vision_distance_result(image_left, image_right, labels_boxes_json_left, labels_boxes_json_right):   
    # TODO 
    # returns distances dict for each boxes
    # example output 
    # {'names': result.names, "classes": result.boxes.cls, "boxes": result.boxes.xyxy, "confs": result.boxes.conf, "distances": distances}
    #fl = 75-246.3441162109375*155/490.97529220581055
    fl = 59.4-59.858795166015625*89.1/78.972900390625
    image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
    image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)
    sz1 = image_right.shape[1]
    sz2 = image_right.shape[0]
    det = [labels_boxes_json_left["boxes"], labels_boxes_json_right["boxes"]]
    lbls = [labels_boxes_json_left["classes"], labels_boxes_json_right["classes"]]
    
    centre = sz1/2
    def get_dist_to_centre_tl(box, cntr = centre):
        pnts = np.array(dist_measurement.tlbr_to_corner(box))[:,0]
        return abs(pnts - centre)
    
    def get_dist_to_centre_br(box, cntr = centre):
        pnts = np.array(dist_measurement.tlbr_to_corner_br(box))[:,0]
        return abs(pnts - centre)
    cost = dist_measurement.get_cost(det, lbls = lbls)
    tracks = scipy.optimize.linear_sum_assignment(cost)
    
    dists_tl =  dist_measurement.get_horiz_dist_corner_tl(det)
    dists_br =  dist_measurement.get_horiz_dist_corner_br(det)
    final_dists = []
    dctl = get_dist_to_centre_tl(det[0])
    dcbr = get_dist_to_centre_br(det[0])
    for i, j in zip(*tracks):
        if len(lbls) <= i:
            continue
        if dctl[i] < dcbr[i]:
            final_dists.append((dists_tl[i][j],lbls[i]))
        else:
            final_dists.append((dists_br[i][j],lbls[i]))
        
        
    #tantheta = (1/(155-fl))*(59.0/2)*sz1/246.3441162109375
    tantheta = (1/(89.1-fl))*(5.5/2)*sz1/59.858795166015625

    fd = [i for (i,j) in final_dists]
    #dists_away = (59.0/2)*sz1*(1/tantheta)/np.array(fd)+fl
    dists_away = (5.5/2)*sz1*(1/tantheta)/np.array(fd)+fl
    cat_dist = []
    for i in range(len(dists_away)):
        cat_dist.append(f'{lbls[i]} {dists_away[i]:.1f}cm')
        print("Estimation:")
        print(f'{lbls[i]} is {dists_away[i]:.1f}cm away')
    coordinates = det[0]
    for coord_ind in range(len(cat_dist)):
        label = "{:.1f}".format(dists_away[coord_ind]) + " cm away"
        x_min, y_min, x_max, y_max = coordinates[coord_ind].tolist()
        cv2.rectangle(image_left, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        cv2.putText(image_left, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #output_filename = os.path.join(output_folder, f"output_image_{counter}.jpg")
        #cv2.imwrite(output_filename, image_left)
    cv2.imshow("Image with Distances", image_left)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
