from packages.predictor_trainer.trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    trainer.save_model()