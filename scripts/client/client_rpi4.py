from numpysocket import NumpySocket
import cv2
import time
import logging
import numpy as np
import urllib.request as urllib

HOST = '192.168.1.172'
PORT = 9999

cap_left = cv2.VideoCapture(2)
cap_left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1080);
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080);
cap_right = cv2.VideoCapture(0)
cap_left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1080);
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080);

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
            #cv2.imwrite("./images/image_right.jpg", list[1]) 
            #cv2.imwrite("./images/image_left.jpg", list[0])
            s.sendall(np.array(list))
        else:
            break
        m = (m+1)%2
        time.sleep(0.05)