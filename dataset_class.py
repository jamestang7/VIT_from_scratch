import torch
import random 
from torch.utils.data import DataLoader, Dataset 
from torchvision import transforms 
import numpy as np 

class MINISTTrainDataset(Dataset):
    def __init__(self, images, labels, indicies):
        self.images = images 
        self.labels = labels
        self.indices = indicies 
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]
        )

    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index].reshape((28, 28)).astype(np.uint8)
        label = self.labels[index]
        idx = self.indices[index]
        image = self.transform(image)

        return {"image": image, "label": label, "index": idx}
    

class MINISTValDataset(Dataset):
    def __init__(self, images, labels, indicies):
        self.images = images 
        self.labels = labels
        self.indices = indicies 
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]
        )

    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index].reshape((28, 28)).astype(np.uint8)
        label = self.labels[index]
        idx = self.indices[index]
        image = self.transform(image)

        return {"image": image, "label": label, "index": idx}
