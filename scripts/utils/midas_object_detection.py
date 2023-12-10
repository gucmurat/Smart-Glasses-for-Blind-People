import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import cv2
"""
NOTE:
detect_boxes_with_mask_midas function gives better results than detect_boxes_with_edges_midas function in the test cases.
"""
def detect_boxes_with_mask_midas(depth_map,input_image):

    depth_map_blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)

    
    binary_mask = cv2.adaptiveThreshold(
        depth_map_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 100  
    bounding_boxes = []
    result = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            x = int((x / depth_map.shape[1]) * input_image.shape[1])
            y = int((y / depth_map.shape[0]) * input_image.shape[0])
            w = int((w / depth_map.shape[1]) * input_image.shape[1])
            h = int((h / depth_map.shape[0]) * input_image.shape[0])
            bounding_boxes.append((x, y, x + w, y + h))
            """
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
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
    
    bounding_boxes_image = input_image.copy()
    min_box_area = 10
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_box_area:
            x, y, w, h = cv2.boundingRect(contour)
            x = int((x / depth_map.shape[1]) * input_image.shape[1])
            y = int((y / depth_map.shape[0]) * input_image.shape[0])
            w = int((w / depth_map.shape[1]) * input_image.shape[1])
            h = int((h / depth_map.shape[0]) * input_image.shape[0])
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