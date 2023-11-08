from numpysocket import NumpySocket
import cv2
import time
import logging

HOST = '192.168.0.1'
PORT = 65432

cap = cv2.VideoCapture(1)

with NumpySocket() as s:
    s.connect(("localhost", 9999))
    logging.info("connected to the server.")
    while cap.isOpened():
        ret, frame = cap.read()
        #ref_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resize = frame[::2, ::2]
        if ret is True:
            try:
                s.sendall(frame_resize)
                logging.info("frame is sent to the server.")
            except Exception:
                break
        else:
            break