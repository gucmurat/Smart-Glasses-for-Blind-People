import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import cv2

def detect_boxes_with_mask_midas(depth_map):

    #depth_map = cv2.resize(depth_map, (cv2.imread(selected_img).shape[1], cv2.imread(selected_img).shape[0]))
    
    depth_map_blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)

    
    binary_mask = cv2.adaptiveThreshold(
        depth_map_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 100  
    
    result = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Object Detection with Boxes (Filtered)')
    plt.axis('off')
    plt.show()

def detect_boxes_with_edges_midas(depth_map):
    threshold = 10 

    edges = cv2.Canny(depth_map, threshold, threshold * 1.8)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes_image = depth_map.copy()
    min_box_area = 10
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_box_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(bounding_boxes_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

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