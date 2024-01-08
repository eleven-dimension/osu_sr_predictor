from tqdm import tqdm
from random import sample

from packages.training_data_generator.generator import TrainingDataGenerator
from packages.osu_file_analyzer.analyzer import OsuFileAnalyzer

import numpy as np
import os
import json

class DataSaver:
    def __init__(self) -> None:
        self.analyzer = OsuFileAnalyzer()
        self.generator = TrainingDataGenerator()

        self.saving_path = "./data/training_input/"
        self.sr_dict_path = "./data/sr.json"
        with open(self.sr_dict_path, 'r') as file:
            self.data = json.load(file)
        self.data_in_list = list(self.data.items())

    def save_a_beatmap(
            self, 
            file_path: str, 
            difficulty: float, 
            folder_index: int
    ) -> None:
        circle_lists, hold_lists = self.analyzer.analyze_map(file_path)
        self.generator.update_with_a_new_beatmap(circle_lists, hold_lists)
        begin_indices = self.generator.get_random_sections_begin_indices()
        
        if begin_indices == []:
            return
        # [dim_num, interval_num]
        mask = self.generator.get_circle_and_holds_mask(begin_indices)
        # print(mask.dtype)
        new_folder_path = self.saving_path + str(folder_index) + "/"
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        np.save(new_folder_path + 'input.npy', mask)
        np.savetxt(
            new_folder_path + 'target.txt', 
            np.array([difficulty], dtype=np.float32),
            fmt='%.3f'
        )


    def save_all(self) -> None:
        for index, (beatmap_path, sr) in tqdm(enumerate(self.data.items())):
            self.save_a_beatmap(beatmap_path, sr, index)


    def randomly_update_training_input(self, update_size) -> None:
        update_index_list = sample([_ for _ in range(len(self.data))], update_size)
        
        for index in update_index_list:
            beatmap_path, sr = self.data_in_list[index]
            self.save_a_beatmap(beatmap_path, sr, index)