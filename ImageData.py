import os
import random

import cv2
import matplotlib.pyplot as plt
import torch.utils.data
from torchvision import transforms


class ImageDataset(torch.utils.data.Dataset):
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

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, item):
        image_dir = self.image_dirs[item]
        image_path = os.path.join(self.split_path, image_dir)
        image_numpy = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image_tensor = torch.FloatTensor(image_numpy).permute(2,0,1)
        if self.split == "train" or self.split == "valid":
            category = int(image_dir.removesuffix(".jpg").split("_")[0])
            train_transform = transforms.Compose([
                transforms.Resize((260, 260), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                transforms.RandomApply([transforms.RandomRotation(degrees=60)], p=0.5),
                transforms.CenterCrop((224, 224))
            ])
            return train_transform(image_tensor), category

        elif self.split == "test":
            test_transform = transforms.Compose([
                transforms.Resize((224, 224), antialias=True),
            ])
            return test_transform(image_tensor)
        else:

            raise ValueError("Split must be either 'train', 'valid' or 'test'")


def show_image_tensor(image_tensor):
    plt.imshow(image_tensor.permute(1, 2, 0).to(int))
    plt.show()


if __name__ == "__main__":
    image_dataset = ImageDataset("train")
    for _ in range(10):
        x = random.randint(0, len(image_dataset))
        show_image_tensor(image_dataset[x][0])
