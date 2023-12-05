from numpysocket import NumpySocket
import cv2
import time
import logging
import numpy as np

HOST = 'localhost'
PORT = 9999

cap_right = cv2.VideoCapture(1) # right(reversed)
cap_left = cv2.VideoCapture(0) # left

logger = logging.getLogger("OpenCV server")
logger.setLevel(logging.INFO)

with NumpySocket() as s:
    s.connect((HOST, PORT))
    logger.info("connected to the server.")
    while cap_right.isOpened() and cap_left.isOpened():
        ret_right, frame_right = cap_right.read()
        
        frame_right = cv2.rotate(frame_right, cv2.ROTATE_180)
        #ref_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret_left, frame_left = cap_left.read()

        frame_right_resize = frame_right[::2, ::2]
        frame_left_resize = frame_left[::2, ::2]
        
        if ret_left and ret_right:
            try:
                array_of_frames = np.array([frame_right_resize, frame_left_resize])
                s.sendall(array_of_frames)
            except Exception as e:
                print(e)
                break
        else:
            break
        