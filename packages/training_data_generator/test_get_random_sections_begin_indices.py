from packages.training_data_generator.generator import TrainingDataGenerator

if __name__ == "__main__":
    g = TrainingDataGenerator(
        interval_num=5,
        section_num=3
    )
    g.update_with_a_new_beatmap(
        [
            [5, 10, 13, 30, 39, 40, 50],
            [9, 46, 76, 89, 90, 105, 118],
            [5, 10, 13, 30],
            [9, 46, 76, 89, 90, ]      
        ],
        [
            [[30, 55], [90, 175]],
            [[30, 55], [90, 180]],
            [[30, 55], [90, 170]],
            [[35, 55], [90, 175], [200, 250]]        
        ]
    )

    print(g.get_random_sections_begin_indices())