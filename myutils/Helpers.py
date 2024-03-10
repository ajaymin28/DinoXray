import torch.utils.data as data
from PIL import Image
import os
import torch
import torchvision.transforms as transforms 
import numpy as np
from torchvision.io import read_image
# import clip
import glob

# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

class ChestXRayDataloader(data.Dataset):
    """ HAR dataloader """

    def __init__(self, img_preprocessing_fn, labels={ "NORMAL": 0 , "PNEUMONIA": 1}, rootPath="./chest_xray/train/", seed=123):
        self.labels = []
        self.images = []

        self.transform = transforms.Compose([ 
            # transforms.PILToTensor(),
            transforms.Resize((224,224), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        
        for label, labelInt in labels.items():
            print(f"Looking for images in : {rootPath}/{label}/*.jpeg")

            # images_list = glob.glob(f"{rootPath}/{label}/*.JPEG")
            images_list = glob.glob(f"{rootPath}/{label}/*.jpeg")
            # images_list += glob.glob(f"{rootPath}/{label}/*.png")
            # images_list += glob.glob(f"{rootPath}/{label}/*.PNG")

            print(f"{label} Images found: {len(images_list)}")

            for imagePath in images_list:
                self.labels.append(labelInt)
                self.images.append(imagePath)


        self.img_preprocessing_fn = img_preprocessing_fn

        
        self.classes = list(set(self.labels))
        self.classes_zeros_list = [0 for k in self.classes]

    def create_one_hot_encoding(self, label):
        index_of_label = self.classes.index(label)
        zerosList = self.classes_zeros_list.copy()
        zerosList[index_of_label] = 1
        return zerosList

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target)
        """

        image, caption = None, None

        if os.path.exists(self.images[index]):
            # self.create_one_hot_encoding(self.labels[index])
            Img = Image.open(f"{self.images[index]}").convert('RGB') 
            image = self.img_preprocessing_fn(Img)
            caption = self.labels[index]
            return image, caption 
        return None,None

    def __len__(self):
        return len(self.images)