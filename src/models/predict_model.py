import logging
from pathlib import Path

import hydra
import os

from dotenv import find_dotenv, load_dotenv
from model import AwesomeSpamClassificationModel
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from src.data.data import SpamDatasetDataModule


@hydra.main(
    version_base=None, config_path="../../config", config_name="default_config.yaml"
)
def main(config: DictConfig):
    print("Evaluating until hitting the ceiling")

    model = AwesomeSpamClassificationModel(config=config)
    model = model.load_from_checkpoint(
        os.path.join(config.model.model_output_dir, config.model.model_name_local)
    )
    model.eval()

    data_module = SpamDatasetDataModule(config=config)
    data_module.setup()

    trainer = Trainer()
    trainer.test(model, dataloaders=data_module.test_dataloader())


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
