from packages.predictor_trainer.trainer import Trainer

from tqdm import tqdm

if __name__ == "__main__":
    trainer = Trainer()
    trainer.load_model()
    trainer.validate("./data/validate/90.osu")