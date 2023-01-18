import logging
import os
from pathlib import Path

import hydra
import model as mymodel
import torch
from dotenv import find_dotenv, load_dotenv
from model import AwesomeSpamClassificationModel
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from src import _PATH_DATA
from src.data.data import SpamDatasetDataModule


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = AwesomeSpamClassificationModel(mymodel.OUTPUT_SIZE)
    model.load_state_dict(checkpoint)
    return model


# @click.command()
# @click.argument("model_checkpoint", type=click.Path(exists=True))
@hydra.main(
    version_base=None, config_path="../../config", config_name="config_all.yaml"
)
def main(config: DictConfig):
    print("Evaluating until hitting the ceiling")

    path_checkpoint = os.path.join(
        hydra.utils.get_original_cwd(),
        config.predict.model_output_dir,
        "trained_model.ckpt",
    )

    model = AwesomeSpamClassificationModel.load_from_checkpoint(path_checkpoint)

    data_module = SpamDatasetDataModule(data_path=_PATH_DATA, config=config)
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
