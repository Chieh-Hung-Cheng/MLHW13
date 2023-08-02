import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from Utils import Utils
from ImageData import ImageDataset
from ImageNetwork import ImageNetwork, get_teacher_network


class Trainer:
    def __init__(self):
        # Configurations
        Utils.fix_randomness(3105432)
        self.save_path = os.path.join(os.getcwd(), "models")
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        self.time_string = time.strftime("H%M%m%d")
        self.num_epochs = 50
        batch_size = 64
        # Data Related
        self.trainset = ImageDataset("train")
        self.validset = ImageDataset("valid")
        self.train_loader = DataLoader(self.trainset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       pin_memory=True)
        self.valid_loader = DataLoader(self.validset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       pin_memory=True)
        # Model related
        self.teacher = get_teacher_network()
        self.student = ImageNetwork().cuda()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=0.001)
        # Stats
        self.best_accuracy = np.NINF
        self.best_loss = np.PINF

    def train_loop(self):
        for epoch in range(self.num_epochs):
            mean_train_loss = self.train_one_epoch()
            mean_valid_loss, valid_accuracy = self.valid_one_epoch()
            self.summarize(valid_accuracy)

    def train_one_epoch(self):
        self.student.train()
        self.teacher.eval()
        loss_accumulate = 0
        idx = 0
        train_pbar = tqdm(self.train_loader)
        for idx, (image_b, label_b) in enumerate(train_pbar):
            # Forward pass
            image_b, label_b = image_b.cuda(), label_b.cuda()
            pred_example = self.teacher(image_b)
            pred_practice = self.student(image_b)
            # Backward Pass
            self.optimizer.zero_grad()
            pred_example_softmax = torch.softmax(pred_example, dim=1)
            loss = self.criterion(pred_example_softmax, pred_practice)
            loss.backward()
            self.optimizer.step()
            # Statistics
            loss_accumulate += loss.item()
            # Display
            train_pbar.set_postfix({"mean train loss": f"{loss_accumulate / (idx + 1):.5f}",
                                    "best_accuracy": f"{self.best_accuracy:.1%}"})

        return loss_accumulate / (idx + 1)

    def valid_one_epoch(self):
        self.student.eval()
        loss_accumulate = 0
        hit_count = 0
        item_count = 0
        idx = 0
        valid_pbar = tqdm(self.valid_loader)
        for idx, (image_b, label_b) in enumerate(valid_pbar):
            # Forward Pass
            image_b, label_b = image_b.cuda(), label_b.cuda()
            pred_b = self.student(image_b)
            # Backward Pass
            loss = self.criterion(pred_b, label_b)
            # Statistics
            hit_count += (pred_b.argmax(1) == label_b).sum().item()
            item_count += len(label_b)
            loss_accumulate += loss.item()
            # Display
            valid_pbar.set_postfix({"mean valid loss": f"{loss_accumulate / (idx + 1):.5f}",
                                    "valid accuracy": f"{hit_count / item_count:.1%}",
                                    "best_accuracy": f"{self.best_accuracy:.1%}"})

        return loss_accumulate / (idx + 1), hit_count / item_count

    def summarize(self, valid_accuracy):
        if valid_accuracy > self.best_accuracy:
            self.best_accuracy = valid_accuracy
            torch.save(self.student.state_dict(), os.path.join(self.save_path, f"model_{self.time_string}.ckpt"))


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_loop()
