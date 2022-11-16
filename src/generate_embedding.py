from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import os 


class GenerateEmbedding:
    def __init__(self , args):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))
        self.mtcnn = MTCNN(image_size=112, margin=0, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.args = args

    def getFaceEmbedding(self):
        dataset = datasets.ImageFolder("./datasets/train")
        idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
        loader = DataLoader(dataset, collate_fn=lambda x: x[0])

        knownEmbeddings = []
        knownNames = []

        for img, idx in loader:
            face, prob = self.mtcnn(img, return_prob=True) 
            if face is not None and prob > 0.90:
                embedding = self.resnet(face.unsqueeze(0).to(self.device)).detach()
                print(embedding.shape)
                knownEmbeddings.append(embedding.numpy())
                knownNames.append(idx_to_class[idx])
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        embedding_path = "./faceEmbeddingModels/embeddings.pt"
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        torch.save(data, embedding_path)
class TrainModel:
    def __init__(self, args):
        self.args = args

    def train(self):
        embedding_path = "./faceEmbeddingModels/embeddings.pt"
        data = torch.load(embedding_path)

