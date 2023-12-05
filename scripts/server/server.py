from numpysocket import NumpySocket
import evaluation
import cv2
import logging
import requests

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
                #################################################
                cv2.imshow("Camera 1", frame_right)
                cv2.imshow("Camera 2", frame_left)
                
                key = cv2.waitKey(1) & 0xFF  # Capture the key event once
                
                obj_result = evaluation.get_distance_values_from_objects(frame_left, frame_right)
                obj_sentence = evaluation.result_to_sentence(obj_result)
                print(obj_sentence)
                capt_result = evaluation.image_captioning_result(frame_left)[0]['generated_text']
                print(capt_result)
                
                
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    try:
                        pass
                        #url = "http://127.0.0.1:8000/send_text"
                        #data = {"text": obj_sentence}
                        #response = requests.post(url, json=data)
                    except Exception as e:
                        print(e)
                        continue
                elif key == ord('s'):
                    try:
                        pass
                        #url = "http://127.0.0.1:8000/send_text"
                        #data = {"text": capt_result}
                        #response = requests.post(url, json=data)
                    except Exception as e:
                        print(e)
                        continue
                #################################################
                
                
            logger.info(f"disconnection:: {addr}")
        except ConnectionResetError:
            pass
