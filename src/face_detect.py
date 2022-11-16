from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2


class FaceDetect:

    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))
        self.mtcnn = MTCNN(image_size=112, margin=0, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        saved_data = torch.load('./faceEmbeddingModels/embeddings.pt')
        self.embeddings = saved_data['embeddings']
        self.names = saved_data['names']

    def getFaceEmbedding(self, img):
        face, prob = self.mtcnn(img, return_prob=True)
        boxes , probs = self.mtcnn.detect(img)
        if face is not None and prob > 0.70:
            boxes, probs, landmarks = self.mtcnn.detect(img, landmarks=True)
            embedding = self.resnet(face.unsqueeze(0).to(self.device)).detach()
            return embedding , boxes
        else:
            return None , None
    def get_prediction(self, embedding):
        dist_list = []
        for idx, emb_db in enumerate(self.embeddings):
            dist = torch.dist(embedding, emb_db).item()
            dist_list.append(dist)
        idx_min = dist_list.index(min(dist_list))
        result_name = self.names[idx_min]
        return result_name
    def detect_face(self):
        cv2.namedWindow("Predicted Name")
        vc = cv2.VideoCapture(0)

        while True:
            rval, frame = vc.read()
            if frame is not None:
                embedding , boxes = self.getFaceEmbedding(frame)
                if embedding is not None:
                    result_name = self.get_prediction(embedding)
                    frame = self.draw_boxes(frame, boxes , result_name)

            cv2.imshow("preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.destroyWindow("preview")
        vc.release()
    @staticmethod
    def draw_boxes(frame, boxes, result_name):
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, result_name, (int(x1), int(y1)-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame