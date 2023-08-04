import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ImageNetwork import ImageNetwork
from ImageData import ImageDataset

class Tester:
    def __init__(self):
        # Configurations
        self.time_string = time.strftime("%H%M%m%d")
        # Data
        testset = ImageDataset("test")
        self.test_loader = DataLoader(testset, batch_size=64, shuffle=False, pin_memory=True)
        # Model
        self.output_path = os.path.join(os.getcwd(), "output")
        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)
        model_path = os.path.join(os.getcwd(), "models")
        load_name = "15360804"
        self.student = ImageNetwork().cuda()
        self.student.load_state_dict(torch.load(os.path.join(model_path, f"model_{load_name}.ckpt")))

    def infer(self):
        self.student.eval()
        with torch.no_grad():
            predictions = list()
            for idx, image_b in enumerate(tqdm(self.test_loader)):
                image_b = image_b.cuda()
                pred_b = self.student(image_b)
                predictions.append(pred_b.argmax(dim=1))
            predictions = torch.cat(predictions, dim=0)
            self.save_result_csv(predictions)


    def save_result_csv(self, predictions):
        with open(os.path.join(self.output_path, f"predictions_{self.time_string}.csv"), "w") as f:
            f.write("id,Category\n")
            for idx, prediction in enumerate(predictions):
                f.write(f"{idx},{prediction}\n")


if __name__ == "__main__":
    tester = Tester()
    tester.infer()
