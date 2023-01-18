import logging

# from google.cloud import secretmanager
import os
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
import pickle

warnings.filterwarnings("ignore")

device = "cuda" if cuda.is_available() else "cpu"


@hydra.main(
    version_base=None, config_path="../../config", config_name="config_all.yaml"
)
def main(config: DictConfig):
    # logger = logging.getLogger(__name__)
    # logger.info("Start Training...")

    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    torch.manual_seed(config.train.seed)

    wandb_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_key)
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
    # trainer.save_checkpoint("models/trained_model.ckpt")
    torch.save(model, "models/trained_model.pt")
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save('model_scripted.pt') # Save


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
