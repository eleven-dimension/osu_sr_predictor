from typing import Tuple

import random
import numpy as np
import os

import torch

class TrainingDataLoader:
    def __init__(
            self,
            batch_size=32,
            track_num=4,
            section_num=100,
            interval_num=1000,
    ) -> None:
        self.batch_size = batch_size
        self.input_data_path = "./data/training_input/"
        
        self.input_data_indices = []
        # add index to input_data_indices
        items_in_directory = os.listdir(self.input_data_path)
        for item in items_in_directory:
            item_path = os.path.join(self.input_data_path, item)
            if os.path.isdir(item_path):
                self.input_data_indices.append(int(item))

        self.dim_num = track_num * 2 * section_num
        self.interval_num = interval_num


    def load_a_data_point(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        data_point_path = self.input_data_path + str(index) + "/" 
        input_x_path = data_point_path + "input.npy"
        target_path = data_point_path + "target.txt"

        input_x = np.load(input_x_path)
        target = np.loadtxt(target_path)

        input_x = input_x.reshape((self.dim_num, self.interval_num))
        target = target.reshape((1))
        # print(input_x.shape)
        # print(target.shape)
        return input_x, target


    def get_a_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        indices_chosen = random.sample(self.input_data_indices, self.batch_size)
        inputs = np.zeros((self.batch_size, self.dim_num, self.interval_num))
        targets = np.zeros((self.batch_size, 1))

        # print(indices_chosen)
        for _, index in enumerate(indices_chosen):
            input_x, target = self.load_a_data_point(index)
            inputs[_] = input_x
            targets[_] = target
        
        # print(inputs.shape)
        # print(targets.shape)
        return inputs, targets