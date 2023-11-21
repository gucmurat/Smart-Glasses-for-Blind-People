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
                try:
                    result = evaluation.stereo_vision_distance_result(frame_left, 
                                                            frame_right, 
                              evaluation.detected_labels_and_boxes_result(evaluation.model_yolov8,frame_left), 
                              evaluation.detected_labels_and_boxes_result(evaluation.model_yolov8,frame_right))
                    print(result)
                except Exception as e:
                    print(e)
                    continue
                
            logger.info(f"disconnection:: {addr}")
        except ConnectionResetError:
            pass
