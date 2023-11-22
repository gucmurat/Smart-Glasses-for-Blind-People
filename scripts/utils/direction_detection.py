import torch
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def calculate_direction(box, image):
    x_min, y_min, x_max, y_max = box
    center_x = (x_min + x_max) / 2

    image_width = image.shape[1]  
    horizontal_position = (center_x / image_width - 0.5) * 100  
    direction = 0
    angle = 0
    if -15 <= horizontal_position < 15:
        direction = 0
        #angle = 12
        angle = 0
    elif 15 <= horizontal_position < 45:
        direction = 1
        angle = 1
    elif 45 <= horizontal_position < 75:
        direction = 1
        angle = 2
    elif 75 <= horizontal_position < 105:
        direction = 1
        angle = 3
    elif -45 <= horizontal_position < -15:
        direction = -1
        #angle = 11
        angle = -1
    elif -75 <= horizontal_position < -45:
        direction = -1
        #angle = 10
        angle = -2
    elif -105 <= horizontal_position < -75:
        direction = -1
        #angle = 9
        angle = -3
    else:
        direction = 0
        angle = 0
    return direction, angle


import torch

def angle_to_vertical_3d(box, image_size, depth):
    """
    Calculate the angle between the center of each 3D box and the vertical axis
    passing through the center of the image.

    Parameters:
    - boxes: A tensor representing the boxes in the format (top-left x, top-left y, bottom-right x, bottom-right y)
    - image_size: A tuple (width, height) representing the size of the image
    - depth: A tensor representing the depth of each box

    Returns:
    - angles: A list of angles between each box center and the vertical axis
    """

    # Ensure the input is a PyTorch tensor
    if not isinstance(box, torch.Tensor):
        box = torch.tensor(boxes)

    if not isinstance(depth, torch.Tensor):
        depth = torch.tensor(depth)

    # Extract box coordinates
    x1, y1, x2, y2 = box[0]

    # Calculate the center of the box
    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2

    # Calculate the center of the image
    image_center_x = image_size[0] / 2
    image_center_y = image_size[1] / 2

    # Get the depth of the box
    box_depth = depth

    # Calculate the angle between the box center and the vertical axis
    angle = torch.atan2((box_center_x - image_center_x), (box_center_y - image_center_y) / box_depth)

    # Convert the angle to degrees
    angle_degrees = torch.rad2deg(angle).item()

    return angle_degrees


def direction_to_camera(box_coordinates, image_width, image_height):
    # Extracting coordinates from the tensor
    x_min, y_min, x_max, y_max = box_coordinates[0]

    # Calculating the center of the bounding box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    # Calculating the normalized coordinates (-1 to 1) relative to the image dimensions
    normalized_x = (2 * center_x / image_width) - 1

    # Adjusting the normalized y to be in the range [-1, 1]
    normalized_y = 1 - (2 * center_y / image_height)

    # Calculating the angle in radians
    angle = np.arctan2(normalized_y, normalized_x)

    # Converting the angle to degrees and scaling to be in the range [-90, 90]
    direction_degrees = np.degrees(angle)

    return direction_degrees
"""
def divide_and_show(img, percentages, boxes):
    # Convert the image to a numpy array
    img_array = np.array(img)

    # Get the height and width of the image
    height, width, _ = img_array.shape

    # Convert the numpy array back to an Image object
    img = Image.fromarray(np.uint8(img_array))

    # Calculate the starting and ending indices for each section based on the percentages
    starts = [0] + list(np.cumsum(percentages[:-1]))
    ends = list(np.cumsum(percentages))

    # Display the divided sections
    fig, axes = plt.subplots(1, len(percentages), figsize=(15, 5))

    # Draw boxes on the image if provided
    if boxes is not None:
        print("Not none")
        draw = ImageDraw.Draw(img)

        for box in boxes:
            x1, y1, x2, y2 = box
            box_width = x2 - x1
            box_center = x1 + 0.5 * box_width

            # Determine the percentage area the box belongs to
            percentage_area = None
            for i, (start, end) in enumerate(zip(starts, ends)):
                if start <= box_center <= end:
                    percentage_area = i
                    break

            # Display the divided sections and draw boxes
            for i, (start, end) in enumerate(zip(starts, ends)):
                section = img_array[:, int(start * width):int(end * width), :]
                axes[i].imshow(section)
                axes[i].axis('off')

                # Draw boxes on the image
                draw.rectangle([start * width, y1, end * width, y2], outline="red", width=2)

            # If more than 50% of the box is in the specific percentage area, print the area
            if percentage_area is not None and box_width * 0.5 > (ends[percentage_area] - starts[percentage_area]) * width:
                print(f"Box {box} belongs to the {percentage_area}-th area.")
    
    else:
        print("none")
        # Display the divided sections without boxes
        for i, (start, end) in enumerate(zip(starts, ends)):
            section = img_array[:, int(start * width):int(end * width), :]
            axes[i].imshow(section)
            axes[i].axis('off')
    
    print("jkjsdjkd")
    plt.show()
"""
"""   
def divide_and_show(img, percentages, boxes):
    # Convert the image to a numpy array
    img_array = np.array(img)

    # Get the height and width of the image
    height, width, _ = img_array.shape

    # Convert the numpy array back to an Image object
    img = Image.fromarray(np.uint8(img_array))

    # Calculate the starting and ending indices for each section based on the percentages
    starts = [0] + list(np.cumsum(percentages[:-1]))
    ends = list(np.cumsum(percentages))

    # Display the divided sections
    fig, axes = plt.subplots(1, len(percentages), figsize=(15, 5))

    # Draw boxes on the image if provided
    if boxes is not None:
        draw = ImageDraw.Draw(img)

        for box in boxes:
            x1, y1, x2, y2 = box
            box_center = 0.5 * (x1 + x2)

            # Determine the percentage area the box belongs to
            percentage_area = None
            for i, (start, end) in enumerate(zip(starts, ends)):
                print(start)
                print(end)
                print(box_center)
                if start <= box_center <= end:
                    percentage_area = i
                    break

            # Display the divided sections and draw boxes
            for i, (start, end) in enumerate(zip(starts, ends)):
                section = img_array[:, int(start * width):int(end * width), :]
                axes[i].imshow(section)
                axes[i].axis('off')

                # Draw boxes on the image if the box belongs to the current percentage area
                if percentage_area == i:
                    draw.rectangle([start * width, y1, end * width, y2], outline="red", width=2)
            print(f"Box {box} belongs to the {percentage_area}-th area.")
            # If the box center is within the specific percentage area, print the area
            if percentage_area is not None:
                print(f"Box {box} belongs to the {percentage_area}-th area.")

    else:
        # Display the divided sections without boxes
        for i, (start, end) in enumerate(zip(starts, ends)):
            section = img_array[:, int(start * width):int(end * width), :]
            axes[i].imshow(section)
            axes[i].axis('off')

    plt.show()
    
"""

def divide_and_show(img, percentages, box):
    img_array = np.array(img)
    box = box[0]
    height, width, _ = img_array.shape

    img = Image.fromarray(np.uint8(img_array))

    starts = [0] + list(np.cumsum(percentages[:-1]))
    ends = list(np.cumsum(percentages))

    #fig, axes = plt.subplots(1, len(percentages), figsize=(15, 5))
    
    if box is not None:
        #draw = ImageDraw.Draw(img)

        x1, y1, x2, y2 = box
        box_center = 0.5 * (x1 + x2)
        percentage_area = None
        for i, (start, end) in enumerate(zip(starts, ends)):
            if start * width <= box_center <= end * width:
                percentage_area = i
                break

        
        for i, (start, end) in enumerate(zip(starts, ends)):
            section = img_array[:, int(start * width):int(end * width), :]
            """
            axes[i].imshow(section)
            axes[i].axis('off')
            """
            """
            if percentage_area == i:
                draw.rectangle([start * width, y1, end * width, y2], outline="red", width=2)
            """
        
        if percentage_area is not None:
            print(f"Box {box} belongs to the {percentage_area}-th area.")
    """
    else:
        # Display the divided sections without the box
        for i, (start, end) in enumerate(zip(starts, ends)):
            section = img_array[:, int(start * width):int(end * width), :]
            axes[i].imshow(section)
            axes[i].axis('off')
    """
    #plt.show()
    return percentage_area
