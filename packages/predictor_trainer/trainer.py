from packages.predictor.net import Predictor
from packages.predictor_trainer.regularized_loss import RegularizedLoss
from packages.predictor_trainer.loader import TrainingDataLoader
from packages.predictor.net import Predictor
from packages.osu_file_analyzer.analyzer import OsuFileAnalyzer
from packages.training_data_generator.generator import TrainingDataGenerator
from packages.training_data_generator.saver import DataSaver


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from tqdm import tqdm

class Trainer:
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.predict_model = Predictor().to(self.device)
        self.loss_function = RegularizedLoss(model=self.predict_model, lambda_reg=1e-3)
        self.optimizer = torch.optim.SGD(self.predict_model.parameters(), lr=5e-3)

        self.batch_size = 32
        self.epoch_num = 12000

        self.model_saving_path = "./data/model/model.pth"

        self.training_data_loader = TrainingDataLoader()
        self.data_saver = DataSaver()


    def save_model(self) -> None:
        torch.save(self.predict_model, self.model_saving_path)

    def load_model(self) -> None:
        self.predict_model = torch.load(self.model_saving_path)
        self.predict_model.to(self.device)


    def train(self) -> None:
        epoch_x, loss_val = [], []
        for epoch_cnt in tqdm(range(self.epoch_num)):
            batch_inputs_np, batch_targets_np = self.training_data_loader.get_a_batch()
            batch_inputs_tensor = torch.tensor(batch_inputs_np, dtype=torch.float32).to(self.device)
            batch_targets_tensor = torch.tensor(batch_targets_np, dtype=torch.float32).to(self.device)

            self.optimizer.zero_grad()

            loss = self.loss_function(self.device, batch_inputs_tensor, batch_targets_tensor)
            loss.backward()
            self.optimizer.step()

            self.data_saver.randomly_update_training_input(self.batch_size // 2)

            # if (epoch_cnt + 1) % 20 == 1:
            #     print(f'Epoch [{epoch_cnt + 1} / {self.epoch_num}], Loss: {loss.item():.4f}')
            epoch_x.append(epoch_cnt)
            loss_val.append(loss.item())

        plt.plot(epoch_x, loss_val)
        plt.show() 


    def evaluate(self, data_point_index: int) -> float:
        self.predict_model.eval()

        input_x_np, target_np = self.training_data_loader.load_a_data_point(data_point_index)
        input_x_np = input_x_np.reshape((1, input_x_np.shape[0], input_x_np.shape[1]))
        
        real_sr = target_np[0]

        input_tensor = torch.tensor(input_x_np, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            out_tensor = self.predict_model(input_tensor)
        
        sr_predicted = out_tensor.detach().cpu().numpy()[0, 0]
        return real_sr, sr_predicted, abs(real_sr - sr_predicted)
    

    def validate(self, file_path: str) -> None:
        analyzer = OsuFileAnalyzer()
        generator = TrainingDataGenerator()

        circle_lists, hold_lists = analyzer.analyze_map(file_path)
        generator.update_with_a_new_beatmap(circle_lists, hold_lists)
        begin_indices = generator.get_random_sections_begin_indices()

        if begin_indices == []:
            return
        # [dim_num, interval_num]
        mask = generator.get_circle_and_holds_mask(begin_indices)

        mask = mask.reshape((1, mask.shape[0], mask.shape[1]))
        input_tensor = torch.tensor(mask, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out_tensor = self.predict_model(input_tensor)
        sr_predicted = out_tensor.detach().cpu().numpy()[0, 0]
        print(f"predicted: {sr_predicted}")