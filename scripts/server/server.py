from numpysocket import NumpySocket
import evaluation
import cv2
import logging
import requests

evaluation.initialize_models()

HOST = ''
PORT = 9999

url = "http://127.0.0.1:8002/send_text"

logger = logging.getLogger("OpenCV server")
logger.setLevel(logging.INFO)

with NumpySocket() as s:
    s.bind((HOST, PORT))
    i = 0
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
                
                key = cv2.waitKey(1) & 0xFF
                
                image_capt_flag = 0                
                
                if i%4!=0 or i==0:    
                    obj_result = evaluation.get_distance_values_from_objects(frame_left, frame_right)
                    obj_sentence = evaluation.result_to_sentence(obj_result)
                    sentence_list = obj_sentence.split(". ")
                    
                    if len(sentence_list)>4:
                        obj_sentence = ". ".join(sentence_list[:3])
                        image_capt_flag = 1
                    if obj_sentence != "":
                        print("------------------------------")
                        print(obj_sentence)
                        print("------------------------------")
                        data = {"text": obj_sentence}
                        response = requests.post(url, json=data)
                        i+=1
                                          
                elif (i%4==0 and i!=0) or image_capt_flag==1:
                    try:
                        capt_result = evaluation.image_captioning_result(frame_left)[0]['generated_text']
                        print("------------------------------")
                        print(capt_result)
                        print("------------------------------")
                        data = {"text": capt_result}
                        response = requests.post(url, json=data)
                        if image_capt_flag==0:
                            i+=1    
                    except Exception as e:
                        print(e)
                        continue 
                    
                if key == ord('q'):
                    break                    
                #################################################
                
                
            logger.info(f"disconnection:: {addr}")
        except ConnectionResetError:
            pass
