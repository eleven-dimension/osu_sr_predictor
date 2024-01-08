from packages.predictor_trainer.trainer import Trainer

from tqdm import tqdm

if __name__ == "__main__":
    trainer = Trainer()
    trainer.load_model()

    delta_sum = 0
    for index in tqdm(trainer.training_data_loader.input_data_indices):
        real_sr, predicted, delta = trainer.evaluate(index)
        delta_sum += delta
    
    print(f"avg delta: {delta_sum / len(trainer.training_data_loader.input_data_indices)}")