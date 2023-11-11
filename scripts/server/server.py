from numpysocket import NumpySocket
import evaluation
import cv2
import logging

evaluation.initialize_models()

HOST = ''
PORT = 9999

logger = logging.getLogger("OpenCV server")
logger.setLevel(logging.INFO)

with NumpySocket() as s:
    s.bind((HOST, PORT))

    while True:
        try:
            s.listen()
            logger.info("Started to listen port.")
            conn, addr = s.accept()

            logger.info(f"connection approved: {addr}")
            while conn:
                array_of_frames = conn.recv()
                if len(array_of_frames) == 0:
                    break
                frame_right = array_of_frames[0]
                frame_left = array_of_frames[1]
                
                cv2.imshow("output", frame_right)
                cv2.imshow("output", frame_left)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            logger.info(f"disconnection:: {addr}")
        except ConnectionResetError:
            pass
