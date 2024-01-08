from packages.predictor_trainer.loader import TrainingDataLoader

if __name__ == "__main__":
    loader = TrainingDataLoader()
    loader.get_a_batch()