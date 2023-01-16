import logging
import warnings
from pathlib import Path

import hydra
import torch
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch import cuda

import wandb
from src import _PATH_DATA
from src.data.data import SpamDatasetDataModule
from src.models.model import AwesomeSpamClassificationModel
# from google.cloud import secretmanager
# import os


warnings.filterwarnings("ignore")

device = "cuda" if cuda.is_available() else "cpu"


@hydra.main(version_base=None, config_path="../../config", config_name="config_all.yaml")
def main(config: DictConfig):
    # logger = logging.getLogger(__name__)
    # logger.info("Start Training...")

    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    torch.manual_seed(config.train.seed)

    # client = secretmanager.SecretManagerServiceClient()
    # PROJECT_ID = "dtumlops-374515"

    # secret_id = "WANDB_API_KEY"
    # resource_name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    # response = client.access_secret_version(name=resource_name)
    # api_key = response.payload.data.decode("UTF-8")
    # os.environ["WANDB_API_KEY"] = api_key

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
