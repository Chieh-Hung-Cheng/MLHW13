import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time

from Utils import Utils
from ImageData import ImageDataset
from ImageNetwork import ImageNetwork, get_teacher_network


def knowledge_distillation_loss(student_answer, labels, teacher_answer, alpha=0.5, temperature=1.0):
    """
    Implement knowledge distillation loss
    :param student_answer: [batch_size, num_classes]
    :param labels: [batch_size, ]
    :param teacher_answer: [batch_size, num_classes]
    :param alpha: float
    :param temperature: float
    :return:
    """
    # Original Cross Entropy Loss
    csl_criterion = nn.CrossEntropyLoss()
    csl_loss = csl_criterion(student_answer, labels)
    assert(csl_loss > 0)
    # KL Divergence Loss
    p_symbol = student_answer / temperature
    q_symbol = teacher_answer / temperature
    p_log_softmax = F.log_softmax(p_symbol, dim=1)
    q_log_softmax = F.log_softmax(q_symbol, dim=1)
    kl_criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
    kl_loss = kl_criterion(p_log_softmax, q_log_softmax)

    return (alpha * (temperature**2) * kl_loss) + ((1 - alpha) * csl_loss)




class Trainer:
    def __init__(self):
        # Configurations
        Utils.fix_randomness(3105432)
        self.save_path = os.path.join(os.getcwd(), "models")
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        self.time_string = time.strftime("%H%M%m%d")
        self.num_epochs = 100
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
                                       shuffle=True,
                                       pin_memory=True)
        # Model related
        self.teacher = get_teacher_network()
        self.student = ImageNetwork().cuda()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=1e-3, weight_decay=1e-5)
        # Stats
        self.best_accuracy = np.NINF
        self.best_loss = np.PINF
        self.stale = 0
        self.epoch_now = 0

    def train_loop(self):
        for self.epoch_now in range(self.num_epochs):
            mean_train_loss = self.train_one_epoch()
            mean_valid_loss, valid_accuracy = self.valid_one_epoch()
            self.summarize(valid_accuracy)

    def train_one_epoch(self):
        self.student.train()
        self.teacher.eval()
        loss_accumulate = 0
        idx = 0
        train_pbar = tqdm(self.train_loader)
        train_pbar.set_description("Train {}/{}".format(self.epoch_now + 1, self.num_epochs))
        for idx, (image_b, label_b) in enumerate(train_pbar):
            # Forward pass
            image_b, label_b = image_b.cuda(), label_b.cuda()
            pred_teacher = self.teacher(image_b)
            pred_student = self.student(image_b)
            # Backward Pass
            self.optimizer.zero_grad()
            loss = knowledge_distillation_loss(pred_student, label_b, pred_teacher, alpha=0.5, temperature=1.5)
            loss.backward()
            self.optimizer.step()
            # Statistics
            loss_accumulate += loss.item()
            # Display
            train_pbar.set_postfix({"mean train loss": f"{loss_accumulate / (idx + 1):.5f}",
                                    "best_accuracy": f"{self.best_accuracy:.1%}",
                                    "stale": f"{self.stale}"})

        return loss_accumulate / (idx + 1)

    def valid_one_epoch(self):
        self.student.eval()
        loss_accumulate = 0
        hit_count = 0
        item_count = 0
        idx = 0
        valid_pbar = tqdm(self.valid_loader)
        valid_pbar.set_description("Valid {}/{}".format(self.epoch_now + 1, self.num_epochs))
        for idx, (image_b, label_b) in enumerate(valid_pbar):
            # Forward Pass
            image_b, label_b = image_b.cuda(), label_b.cuda()
            pred_b = self.student(image_b)
            # Backward Pass
            loss = self.criterion(pred_b, label_b)
            # Statistics
            hit_count += (pred_b.argmax(dim=1) == label_b).sum().item()
            item_count += len(label_b)
            loss_accumulate += loss.item()
            # Display
            valid_pbar.set_postfix({"mean valid loss": f"{loss_accumulate / (idx + 1):.5f}",
                                    "valid accuracy": f"{hit_count / item_count:.1%}",
                                    "best_accuracy": f"{self.best_accuracy:.1%}",
                                    "stale": f"{self.stale}"})

        return loss_accumulate / (idx + 1), hit_count / item_count

    def valid_teacher(self):
        self.teacher.eval()
        with torch.no_grad():
            loss_accumulate = 0
            hit_count = 0
            item_count = 0
            valid_pbar = tqdm(self.valid_loader)
            for idx, (image_b, label_b) in enumerate(valid_pbar):
                # Forward Pass
                image_b, label_b = image_b.cuda(), label_b.cuda()
                pred_b = self.teacher(image_b)
                # Backward Pass
                loss = self.criterion(pred_b, label_b)
                # Statistics
                loss_accumulate += loss.item()
                hit_count += (pred_b.argmax(dim=1) == label_b).sum()
                item_count += len(label_b)
                # Display
                valid_pbar.set_postfix({"mean valid loss": f"{loss_accumulate/(idx+1):.5f}",
                                        "valid accuracy": f"{hit_count / item_count:.1%}"})

    def summarize(self, valid_accuracy):
        if valid_accuracy > self.best_accuracy:
            self.best_accuracy = valid_accuracy
            self.stale = 0
            torch.save(self.student.state_dict(), os.path.join(self.save_path, f"model_{self.time_string}.ckpt"))
        else:
            self.stale += 1


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_loop()
