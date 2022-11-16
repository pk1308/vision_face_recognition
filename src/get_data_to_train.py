from facenet_pytorch import MTCNN
import numpy as np
import torch
import cv2
import os
from datetime import datetime


class TrainingDataCollector:

    def __init__(self, args):
        self.args = args
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(image_size=112, margin=16 , device=device , select_largest=True)

    def collectImagesFromCamera(self):
        # initialize video stream
        cap = cv2.VideoCapture(0)

        # Setup some useful var
        faces = 0
        frames = 0
        max_faces = int(self.args["faces"])
        data_dir =   self.args["output"] 
        if not (os.path.exists(data_dir)):
            os.makedirs(data_dir)

        while faces < max_faces:
            ret, frame = cap.read()
            frames += 1

            dtString = str(datetime.now().microsecond)
            # Get all faces on current frame
            boxes, probs, landmarks = self.detector.detect(frame, landmarks=True)

            if boxes is not None:
                file_path = os.path.join(data_dir, f'{dtString}.jpg')
                print(file_path)
                cv2.imwrite(file_path, frame)
                message = f'Face {faces + 1} of {max_faces} saved'
                print(message)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, message, (50, 50), 1,  font, (255, 0, 0), 2 , cv2.LINE_AA)
                faces += 1
            cv2.imshow("Face detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
