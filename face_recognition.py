
import cv2
import numpy as np
from face_train import Model
from load_data import labels_to_num

def face(path_model):

    model = Model()
    model.load_model(file_path=path_model)

    color = (0, 255, 0)

    cap = cv2.VideoCapture(0)

    cascade_path = "F:/app_project/web_face_identification/face_identification/face_project/haarcascade_frontalface_alt2.xml"

    while True:
        ret, frame = cap.read()  

        if ret is True:

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue
        cascade = cv2.CascadeClassifier(cascade_path)

        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect

                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                face_res = model.face_predict(image)
                faceID = np.argsort(face_res[0])[-1]
                if face_res[0][faceID] >= 0.8:  
                    face_num = labels_to_num[faceID]  

                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)

                    cv2.putText(frame, f'{face_num}:{face_res[0][faceID]*100}%',
                                (x + 30, y + 30),  
                                cv2.FONT_HERSHEY_SIMPLEX,  
                                1,  
                                (255, 0, 255),  
                                2)  
                else:
                    pass

        cv2.imshow("识别朕", frame)

        k = cv2.waitKey(10)

        if k & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    face("F:/app_project/web_face_identification/face_identification/face_project/model/face_model_03.keras")