import torch
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def divide_and_show(img, box, left_or_right):
    img_array = np.array(img)
    box = box[0]
    height, width, _ = img_array.shape
    if left_or_right == "l":
        percentages = [0.1, 0.3, 0.4, 0.2]
    else:
        percentages = [0.2, 0.4, 0.3, 0.1]
    img = Image.fromarray(np.uint8(img_array))
    starts = [0] + list(np.cumsum(percentages[:-1]))
    ends = list(np.cumsum(percentages))
    result_dict = {}
    
    if box is not None:
        x1, y1, x2, y2 = box
        box_center = 0.5 * (x1 + x2)
        percentage_area = None
        for i, (start, end) in enumerate(zip(starts, ends)):
            if start * width <= box_center <= end * width:
                percentage_area = i
                break
        for i, (start, end) in enumerate(zip(starts, ends)):
            section = img_array[:, int(start * width):int(end * width), :]
            section_width = int(end * width) - int(start * width)

            overlap_start = max(x1, int(start * width))
            overlap_end = min(x2, int(end * width))
            overlap_width = max(0, overlap_end - overlap_start)
            overlap_area = overlap_width * (y2 - y1)

            box_area = (x2 - x1) * (y2 - y1)
            box_percentage_in_section = (overlap_area / box_area) * 100

            if left_or_right == "l":
                result_dict[i] = box_percentage_in_section
            elif left_or_right == "r":
                result_dict[i+1] = box_percentage_in_section
                
    return result_dict

def get_object_direction(image_left, image_right, box):
    left_angle = divide_and_show(image_left, box[0], "l")
    right_angle = divide_and_show(image_right, box[1], "r")
    direction_count = 5
    final_direction = 2
    max_percentage = 0
    for cur_direction in range(direction_count):
        cur_percentage = 0
        if cur_direction == 0:
            cur_percentage = left_angle[cur_direction]
        elif cur_direction == 4:
            cur_percentage = right_angle[cur_direction]
        else:
            cur_percentage = right_angle[cur_direction] + left_angle[cur_direction]
        
        if cur_percentage >= max_percentage:
            max_percentage = cur_percentage
            final_direction = cur_direction
    return direction_num_to_clock_direction(final_direction)

def direction_num_to_clock_direction(direction_num):
    if direction_num == 0:
        return 10
    elif direction_num == 1:
        return 11
    elif direction_num == 2:
        return 12
    elif direction_num == 3:
        return 1
    elif direction_num == 4:
        return 2
    else:
        return 12