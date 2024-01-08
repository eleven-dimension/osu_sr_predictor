from typing import List, Tuple
from math import ceil
from random import randint

import numpy as np

class BinarySearch:
    @classmethod
    def binary_search_range(cls, arr: List[int], left: int, right: int) -> Tuple[int, int]:
        # first >=
        def left_bound(arr, target):
            left, right = 0, len(arr)
            while left < right:
                mid = left + (right - left) // 2
                if arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid
            return left

        # first >
        def right_bound(arr, target):
            left, right = 0, len(arr)
            while left < right:
                mid = left + (right - left) // 2
                if arr[mid] <= target:
                    left = mid + 1
                else:
                    right = mid
            return left

        left_index = left_bound(arr, left)
        right_index = right_bound(arr, right)

        return left_index, right_index


class TrainingDataGenerator:
    def __init__(
            self,
            interval_num=1000,
            interval_length=1000,
            section_length=10,
            section_num=100,
            track_num=4
    ) -> None:
        self.interval_num = interval_num
        self.interval_length = interval_length
        self.section_length = section_length
        self.section_num = section_num

        self.track_num = track_num

        self.holds_difference_map = None
        self.circles_map = None
    

    def map_left_to_index(self, t):
        return 1 + t // 10

    def map_right_to_index(self, t):
        if t % 10 == 0:
            return t // 10
        return 1 + t // 10


    def set_circles_map(self, circle_lists: List[List[int]]) -> None:
        last_circle_time = -1
        for track_id in range(self.track_num):
            if len(circle_lists[track_id]) > 0:
                last_circle_time = max(last_circle_time, circle_lists[track_id][-1])
        
        self.circles_map = [
            [0] * (2 + ceil(last_circle_time / self.section_length)) for _ in range(
                self.track_num
            )
        ]

        for track_id in range(self.track_num):
            for circle in circle_lists[track_id]:
                circle_index = self.map_left_to_index(circle)
                self.circles_map[track_id][circle_index] = 1

         
    def zero_patching_at_tail(self) -> None:
        max_len = 0
        for track_id in range(self.track_num):
            max_len = max(max_len, len(self.circles_map[track_id]))
            max_len = max(max_len, len(self.holds_difference_map[track_id]))
        
        self.circles_map = [
            track_list + [0] * (max_len - len(track_list)) for track_list in self.circles_map
        ]
        self.holds_difference_map = [
            track_list + [0] * (max_len - len(track_list)) for track_list in self.holds_difference_map
        ]

    
    def set_holds_difference_map(self, hold_lists: List[List[Tuple[int, int]]]) -> None:
        last_hold_end_time = -1
        for track_id in range(self.track_num):
            if len(hold_lists[track_id]) > 0:
                last_hold_end_time = max(last_hold_end_time, hold_lists[track_id][-1][1])
        
        self.holds_difference_map = [
            [0] * (2 + ceil(last_hold_end_time / self.section_length)) for _ in range(
                self.track_num
            )
        ]
        
        # print(len(self.holds_difference_map[0]))

        for track_id in range(self.track_num):
            for hold in hold_lists[track_id]:
                [begin_time, end_time] = hold
                # print(1 + (begin_time // self.section_length))
                begin_index = self.map_left_to_index(begin_time)
                self.holds_difference_map[track_id][
                    begin_index
                ] += 1
                
                end_index = self.map_right_to_index(end_time)
                self.holds_difference_map[track_id][
                    end_index + 1
                ] -= 1
        
        for track_id in range(self.track_num):
            for index in range(1, len(self.holds_difference_map[track_id])):
                self.holds_difference_map[track_id][index] += self.holds_difference_map[track_id][index - 1]

        # print(self.holds_difference_map)


    def update_with_a_new_beatmap(
            self, circle_lists: List[List[int]], 
            hold_lists: List[List[Tuple[int, int]]]
    ) -> None:
        self.set_holds_difference_map(hold_lists)
        self.set_circles_map(circle_lists)
        self.zero_patching_at_tail()


    def get_random_sections_begin_indices(self) -> List[int]:
        begin_indices = [0] * self.interval_num
        max_begin_index = len(self.circles_map[0]) - self.section_num
        if max_begin_index <= 0:
            return []
        for section_index in range(self.interval_num):
            begin_indices[section_index] = randint(0, max_begin_index)

        return begin_indices

    
    # [800, 1000] [dim_num, interval_num]
    def get_circle_and_holds_mask(self, begin_indices: List[int]) -> np.ndarray:
        dim_num = self.track_num * 2 * self.section_num
        # print(f"dim_num: {dim_num}")
        masks = np.zeros((self.interval_num, dim_num), dtype=np.bool_) # [1000, 800]
        for interval_index, begin_index in enumerate(begin_indices):
            end_index = begin_index + self.section_num - 1 # [begin_index, end_index]
            
            mask = []
            for track_circles_map in self.circles_map:
                mask.extend(track_circles_map[begin_index:end_index + 1])
            for track_holds_map in self.holds_difference_map:
                mask.extend(track_holds_map[begin_index:end_index + 1])
            # print(len(mask))
            masks[interval_index] = np.array(mask, dtype=np.bool_)

        # (800, 1000)
        return masks.T