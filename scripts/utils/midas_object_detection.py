import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import cv2
"""
NOTE:
detect_boxes_with_mask_midas function gives better results than detect_boxes_with_edges_midas function in the test cases.
"""
def detect_boxes_with_mask_midas(depth_map,image):
    input_image = image
    depth_map_blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)

    
    binary_mask = cv2.adaptiveThreshold(
        depth_map_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 100  
    bounding_boxes = []
    #result = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    #result = input_image.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            x = int((x / depth_map.shape[1]) * input_image.shape[1])
            y = int((y / depth_map.shape[0]) * input_image.shape[0])
            w = int((w / depth_map.shape[1]) * input_image.shape[1])
            h = int((h / depth_map.shape[0]) * input_image.shape[0])
            bounding_boxes.append((x, y, x + w, y + h))
            #cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    """
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Object Detection with Boxes (Filtered)')
    plt.axis('off')
    plt.show()
    """
    return bounding_boxes

def detect_boxes_with_edges_midas(depth_map,input_image):
    threshold = 10 
    #resized_depth_map = cv2.resize(depth_map, (input_image.shape[1], input_image.shape[0]))
    smoothed_depth_map = cv2.bilateralFilter(depth_map, d=9, sigmaColor=75, sigmaSpace=75)
    edges = cv2.Canny(smoothed_depth_map, threshold, threshold * 1.8)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #bounding_boxes_image = input_image.copy()
    min_box_area = 10
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_box_area:
            x, y, w, h = cv2.boundingRect(contour)
            x = int((x / depth_map.shape[1]) * input_image.shape[1])
            y = int((y / depth_map.shape[0]) * input_image.shape[0])
            w = int((w / depth_map.shape[1]) * input_image.shape[1])
            h = int((h / depth_map.shape[0]) * input_image.shape[0])
            #cv2.rectangle(bounding_boxes_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(depth_map, cmap='gray')
    plt.title('Midas Depth Map')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.imshow(edges, cmap='gray')
    plt.title('Edges')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(bounding_boxes_image, cv2.COLOR_BGR2RGB))
    plt.title('Bounding Boxes')
    plt.axis('off')
    plt.show()
    """
def calculate_intersection_over_union(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou1 = intersection_area / float(area_box1)
    iou2 = intersection_area / float(area_box2)
    if iou1 > 0.6 or iou2 > 0.6:
        return True
    else:
        return False

def draw_boxes_on_image(image, boxes, color=(0, 255, 0), thickness=2):
    image_with_boxes = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, thickness)

    return image_with_boxes

def compare_detections_from_alg_midas(bounding_boxes_midas, detected_yolo, image):
    boxes_yolo = detected_yolo['boxes'].tolist()
    detected_midas = list()
    for i, bb_midas in enumerate(bounding_boxes_midas):
        found = False
        for j, box_yolo in enumerate(boxes_yolo):
            yolo_box = [box_yolo[0], box_yolo[1], box_yolo[2], box_yolo[3]]
            iou = calculate_intersection_over_union(bb_midas, yolo_box)
            if iou:
                """
                print(f"Match found: Midas box {i+1} and YOLO box {j+1}")
                print(f"Midas box coordinates: {bb_midas}")
                print(f"YOLO box coordinates: {yolo_box}")
                print(f"IoU: {iou}")
                print("-" * 40)
                image_with_boxes = draw_boxes_on_image(image, [bb_midas, yolo_box])
                plt.figure()
                plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
                plt.title(f"Match {i+1}-{j+1}")
                plt.axis('off')
                plt.show()
                """
                found = True
                continue
        if found == False:
            detected_midas.append(bb_midas)
    return detected_midas