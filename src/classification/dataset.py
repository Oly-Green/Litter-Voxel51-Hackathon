import torchvision.transforms as transforms
import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
import random

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

category_dict = {'glass': [9, 26], 'plastic': [4, 5, 29, 42, 47, 48, 49], 'paper': [33, 35], 'styrofoam': [57]}
class_to_category_idx = {6: 0, 9: 0, 23: 0, 26: 0, 21: 3, 24: 3, 7: 3, 4: 3, 5: 3,  29: 3, 42: 3, 47: 3, 48: 3, 49: 3, 33: 1, 35: 1, 57: 2}
idx_to_category = {0: 'glass', 1: 'paper', 2: 'styrofoam', 3: 'plastic', 4: 'other'}

class TacoTrashDataset(Dataset):
    def __init__(self, data_dir, isTrain=True):
        self.data_dir = data_dir
        self.transform = transform
        self.isTrain = isTrain

        with open(data_dir + '/annotations.json') as f:
            self.data_info = json.load(f)
        
        self.images_info = self.data_info['images']
        self.all_annotation_info = self.data_info['annotations']

        self.restrictAnnotations()
        self.splitTrainTest()
    
    def restrictAnnotations(self):
        self.annotation_info  = []
        for annotation in self.all_annotation_info:
            if annotation['category_id'] in [4, 5, 6, 7, 9, 26, 29, 42, 47, 48, 49, 33, 35, 57]:
                self.annotation_info.append(annotation)
    
    def splitTrainTest(self):
        random.shuffle(self.annotation_info)
        self.train_annotation_info = self.annotation_info[:int(0.8*len(self.annotation_info))]
        self.test_annotation_info = self.annotation_info[int(0.8*len(self.annotation_info)):]
        
    
    def __len__(self):
        if self.isTrain:
            return len(self.train_annotation_info)
        else:
            return len(self.test_annotation_info)
    
    def __getitem__(self, idx):
        if self.isTrain:
            annotation = self.train_annotation_info[idx]
        else:            
            annotation = self.test_annotation_info[idx]
        
        img_idx = annotation['image_id']
        img_path = os.path.join(self.data_dir, self.images_info[img_idx]['file_name'])
        if annotation['category_id'] in [9, 26, 29, 42, 47, 48, 49, 33, 35, 57]:
            label = class_to_category_idx[annotation['category_id']]
        else:
            label = 4
        
        image = Image.open(img_path).convert("RGB")
        bbox = annotation['bbox']
        image = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
        labels = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)

        return image, labels