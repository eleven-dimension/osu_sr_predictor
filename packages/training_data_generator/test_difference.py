from packages.training_data_generator.generator import TrainingDataGenerator

if __name__ == "__main__":
    g = TrainingDataGenerator()

    g.set_holds_difference_map([
        [[30, 55], [90, 175]],
        [[30, 55], [90, 180]],
        [[30, 55], [90, 170]],
        [[35, 55], [90, 175]]        
    ])
    '''
        [
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], 
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], 
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        ]
    '''
