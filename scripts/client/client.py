from numpysocket import NumpySocket
import cv2
import time
import logging

HOST = 'localhost'
PORT = 9999

cap = cv2.VideoCapture(1)

logger = logging.getLogger("OpenCV server")
logger.setLevel(logging.INFO)

with NumpySocket() as s:
    s.connect((HOST, PORT))
    logger.info("connected to the server.")
    while cap.isOpened():
        ret, frame = cap.read()
        #ref_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resize = frame[::2, ::2]
        if ret is True:
            try:
                s.sendall(frame_resize)
                logger.info("frame is sent to the server.")
            except Exception:
                break
        else:
            break