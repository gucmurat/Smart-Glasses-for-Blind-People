from numpysocket import NumpySocket
import cv2
import time
import logging
import numpy as np
import urllib.request as urllib

HOST = '192.168.1.172'
PORT = 9999

cap_left = cv2.VideoCapture(2) # left
cap_left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap_right = cv2.VideoCapture(0) # right
cap_left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

logger = logging.getLogger("OpenCV server")
logger.setLevel(logging.INFO)

with NumpySocket() as s:
    s.connect((HOST, PORT))
    logger.info("connected to the server.")
    m=0
    list = [None,None]
    while  cap_left.isOpened():
        if m==0:
            ret, frame = cap_left.read()
            frame_left_resize = frame[::2, ::2]
            list[0] = frame_left_resize
        elif m==1:
            ret, frame = cap_right.read()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            frame_right_resize = frame[::2, ::2]
            list[1] = frame_right_resize
            s.sendall(np.array(list))
        else:
            break
        m = (m+1)%2
        time.sleep(0.05)