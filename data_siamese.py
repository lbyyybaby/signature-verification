import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# Transformation for training, validation, and testing data
def get_train_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
            A.ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
            A.CoarseDropout(p=0.2),
            A.Transpose(p=0.5)
        ]),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

def get_test_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

# Create datasets suitable for Siamese Network
class SiameseDataset(Dataset):
    def __init__(self, images_1, images_2, labels, transform=None):
        self.images_1 = images_1
        self.images_2 = images_2
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img1 = Image.open(self.images_1[idx]).convert('RGB')
        img2 = Image.open(self.images_2[idx]).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            img1 = np.array(img1) / 255.0
            img2 = np.array(img2) / 255.0
            img1 = self.transform(image=img1)['image']
            img2 = self.transform(image=img2)['image']

        return img1, img2, label
    
