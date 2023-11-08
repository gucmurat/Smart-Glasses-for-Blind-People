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
                frame = conn.recv()
                if len(frame) == 0:
                    break
                logging.info("frame is obtained.")
                result = evaluation.depth_map_result(frame)
                cv2.imshow("Depth Map", result)
                #cv2.imshow("Frame", frame)

                # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    exit(1)
            logger.info(f"disconnection:: {addr}")
        except ConnectionResetError:
            pass
