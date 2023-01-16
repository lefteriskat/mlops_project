import logging
from pathlib import Path

from src.models.model import AwesomeSpamClassificationModel
import torch
from torch import cuda
from src.data.data import SpamDatasetDataModule

import hydra
from omegaconf import OmegaConf

import wandb
from pytorch_lightning.loggers import WandbLogger

from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig
from src import _PATH_DATA
from pytorch_lightning import Trainer
import warnings

warnings.filterwarnings("ignore")

device = "cuda" if cuda.is_available() else "cpu"


@hydra.main(version_base=None, config_path="../../config", config_name="config_all.yaml")
def main(config: DictConfig):
    # logger = logging.getLogger(__name__)
    # logger.info("Start Training...")

    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    torch.manual_seed(config.train.seed)

    wandb.init(project="test-project", entity="mlops_project_dtu", config=config)
    wandb_logger = WandbLogger(project="test-project", config=config)

    model = AwesomeSpamClassificationModel(config=config)
    model.to(device)

    data_module = SpamDatasetDataModule(data_path=_PATH_DATA, config=config)
    data_module.setup()

    trainer = Trainer(
        max_epochs=config.train.epochs,
        # gpus=gpus,
        logger=wandb_logger,
        # val_check_interval=1.0,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
    )
    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )
    trainer.save_checkpoint("models/trained_model.ckpt")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
