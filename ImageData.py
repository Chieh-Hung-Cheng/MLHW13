import os
import random

import cv2
import matplotlib.pyplot as plt
import torch.utils.data
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.data_path = r"D:\ML_Dataset\HW13\Food-11"
        if self.split == "train":
            self.split_path = os.path.join(self.data_path, "training")
        elif self.split == "valid":
            self.split_path = os.path.join(self.data_path, "validation")
        elif self.split == "test":
            self.split_path = os.path.join(self.data_path, "evaluation")
        else:
            raise ValueError("Split must be either 'train', 'valid' or 'test'")

        self.image_dirs = [i for i in os.listdir(self.split_path)]
        self.train_transform = transforms.Compose([
            transforms.Resize((260, 260), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=60)], p=0.5),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, item):
        image_dir = self.image_dirs[item]
        image_path = os.path.join(self.split_path, image_dir)
        image_pil = Image.open(image_path)

        if self.split == "test":
            return self.test_transform(image_pil)

        category = int(image_dir.removesuffix(".jpg").split("_")[0])
        if self.split == "train":
            return self.train_transform(image_pil), category
        if self.split == "valid":
            return self.test_transform(image_pil), category
        raise ValueError("Split must be either 'train', 'valid' or 'test'")


def show_image_tensor(image_tensor):
    plt.imshow(image_tensor.permute(1, 2, 0).to(int))
    plt.show()


if __name__ == "__main__":
    validset = ImageDataset(split="valid")
