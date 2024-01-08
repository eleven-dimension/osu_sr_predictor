from packages.training_data_generator.saver import DataSaver

if __name__ == "__main__":
    s = DataSaver()
    s.save_a_beatmap(
        "D:/Software/osu/Songs/1819173 Team Grimoire - Dantalion/Team Grimoire - Dantalion (minhtien20079) [Conflict].osu",
        difficulty=3.2,
        folder_index=0
    )
    
