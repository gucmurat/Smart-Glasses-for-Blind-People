from numpysocket import NumpySocket
import evaluation
import cv2
import logging

HOST = '127.0.0.1'
PORT = 65432

logger = logging.getLogger("OpenCV server")
logger.setLevel(logging.INFO)

with NumpySocket() as s:
    s.bind(("", 9999))

    while True:
        try:
            s.listen()
            conn, addr = s.accept()

            logger.info(f"connected: {addr}")
            while conn:
                frame = conn.recv()
                if len(frame) == 0:
                    break
                result = evaluation.depth_map_result(frame)
                cv2.imshow("Depth Map", result)
                #cv2.imshow("Frame", frame)

                # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    exit(1)
            logger.info(f"disconnected: {addr}")
        except ConnectionResetError:
            pass
