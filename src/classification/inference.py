import torchvision.models as models
import torch
from dataset import TacoTrashDataset, transform
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F

model = None

def loadModel():
    global model
    model = models.resnet50()
    num_custom_classes = 5
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_custom_classes)

    checkpoint = torch.load('D:/Work/My Projects/LitterDetection/Litter-Voxel51-Hackathon/src/classification/classification_resnet.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']

    model.eval()
    return model


def predictClass(image, bbox):
    global model
    if model is None:
        model = loadModel()

    image = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
    image = transform(image)
    output = model(image.unsqueeze_(0))
    probabilities = F.softmax(output[0], dim=0)
    predicted_class_idx = torch.argmax(probabilities)
    max_prob = probabilities[predicted_class_idx]
    
    return predicted_class_idx, max_prob

