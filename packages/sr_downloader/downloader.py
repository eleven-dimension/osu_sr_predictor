import os
import requests
import json
from tqdm import tqdm

from typing import List, Tuple

from packages.osu_file_analyzer.analyzer import OsuFileAnalyzer

class DifficultyAPIFetcher:
    def __init__(self) -> None:
        self.API_URL_PREFIX = "http://osu.ppy.sh/api/get_beatmaps?k=b03936ca56264f6a08f27637229aa18ef04fa707&b="


    def fetch_difficulty(self, beatmap_id: int) -> float:
        try:
            # print(self.API_URL_PREFIX + str(beatmap_id))
            response = requests.get(
                self.API_URL_PREFIX + str(beatmap_id),
            )
            if response.status_code == 200:
                data = response.json()

                # unranked or std
                if len(data) == 0 or int(data[0]["approved"]) != 1 or int(data[0]["mode"]) != 3:
                    return -1

                difficulty_rating = float(data[0]["difficultyrating"])

                return difficulty_rating
            else:
                print(f"error code: {response.status_code}")
        except requests.RequestException as e:
            print(f"error: {e}")


class Downloader:
    def __init__(self) -> None:
        self.song_directory = "D:/Software/osu/Songs/"
        # self.song_directory = "./data/songs_test/"
        self.osu_analyzer = OsuFileAnalyzer()
        self.sr_fetcher = DifficultyAPIFetcher()


    def get_all_osu_list(self, song_path: str) -> List[str]:
        items_in_directory = os.listdir(song_path)
        osu_files = []
        for item in items_in_directory:
            if item.endswith(".osu"):
                osu_files.append(os.path.join(song_path, item))
        
        return osu_files


    def get_all_songs_list(self) -> List:
        items_in_directory = os.listdir(self.song_directory)
        songs_list = []
        for item in items_in_directory:
            item_path = os.path.join(self.song_directory, item)
            if os.path.isdir(item_path):
                songs_list.append(item_path)
        return songs_list
    

    def get_osu_sr(self):
        song_list = self.get_all_songs_list()
        dict_from_file_to_sr = {}
        for song_path in tqdm(song_list):
            all_osu_list = self.get_all_osu_list(song_path)
            for osu_file in all_osu_list:
                if not self.osu_analyzer.is_mania_4k(osu_file):
                    continue
                beatmap_id = self.osu_analyzer.get_beatmap_id(osu_file)
                sr = self.sr_fetcher.fetch_difficulty(beatmap_id)
                if sr == -1:
                    continue
                dict_from_file_to_sr[osu_file] = sr

        saved_dict_path = './data/sr.json'
        with open(saved_dict_path, 'w') as file:
            json.dump(dict_from_file_to_sr, file, default=str)