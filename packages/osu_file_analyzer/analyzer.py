from typing import List, Tuple
from math import floor
import re


# (448,192,4,128,0,1228) = (x, y, time, type)
# 448,192,2064,1,0,0:0:0:40:clap.wav     6][:
# 192,192,10038,128,0,10117:0:0:0:0:     6][:
class OsuFileAnalyzer:
    def __init__(self) -> None:
        pass
    

    def is_circle(self, note_type: int) -> bool:
        return bool(note_type & 1)
    

    def is_hold(self, note_type: int) -> bool:
        return bool((note_type >> 7) & 1)

    
    def get_beatmap_id(self, file_path: str) -> int:
        with open(file_path, 'r', encoding='UTF-8') as file:
            beatmap_id_pattern = r'^BeatmapID:\s*(\d+)'
            for line in file:
                raw_str = line.strip()
                match = re.match(beatmap_id_pattern, raw_str)
                if match:
                    beatmap_id = int(match.group(1))
                    return beatmap_id
        return -1


    def is_mania_4k(self, file_path: str) -> bool:
        hit_obj_pattern = r'^([^:]+):'
        mode_pattern = r'^Mode:\s*(\d+)'
        four_key_possible_x = [64, 192, 320, 448]

        with open(file_path, 'r', encoding='UTF-8') as file:
            before_hit_objects = True
            for line in file:
                raw_str = line.strip()
                if before_hit_objects:
                    match = re.match(mode_pattern, raw_str)
                    if match:
                        mode = int(match.group(1))
                        if mode != 3:
                            return False
                
                if raw_str == "[HitObjects]":
                    before_hit_objects = False
                    continue
                if not before_hit_objects:
                    match = re.match(hit_obj_pattern, raw_str)
                    first_six_numbers = match.group(1)
                    hit_object_param_list = first_six_numbers.split(',')
                    hit_object_param_list = [int(x) for x in hit_object_param_list]
                    # deal with list
                    [x, _, begin_time, note_type, __, end_time] = hit_object_param_list

                    if x not in four_key_possible_x:
                        return False
        
        return True


    def analyze_map(self, file_path: str) -> Tuple[List[List[int]], List[List[Tuple[int, int]]]]:
        pattern = r'^([^:]+):'

        circle_lists, hold_lists = [[] for _ in range(4)], [[] for _ in range(4)]

        with open(file_path, 'r', encoding='UTF-8') as file:
            before_hit_objects = True
            for line in file:
                raw_str = line.strip()
                if raw_str == "[HitObjects]":
                    before_hit_objects = False
                    continue
                if not before_hit_objects:
                    match = re.match(pattern, raw_str)
                    first_six_numbers = match.group(1)
                    hit_object_param_list = first_six_numbers.split(',')
                    hit_object_param_list = [int(x) for x in hit_object_param_list]
                    # deal with list
                    [x, _, begin_time, note_type, __, end_time] = hit_object_param_list
                    track_id = floor(x * 4 / 512)
                    if self.is_circle(note_type):
                        circle_lists[track_id].append(begin_time)
                    else:
                        hold_lists[track_id].append([begin_time, end_time])

        return circle_lists, hold_lists