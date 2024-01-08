from packages.training_data_generator.generator import TrainingDataGenerator

if __name__ == "__main__":
    g = TrainingDataGenerator()

    g.set_circles_map([
        [5, 10, 13, 30, 39, 40, 50],
        [9, 46, 76, 89, 90, 105, 118],
        [5, 10, 13, 30, 39, 40, 50],
        [9, 46, 76, 89, 90, 105, 118]      
    ])

    print(g.circles_map)
    '''
        [
            [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0], 
            [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        ]
    '''
