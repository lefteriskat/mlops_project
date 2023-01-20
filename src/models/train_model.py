import logging
import os
import warnings
from pathlib import Path

import hydra
import torch
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import wandb
from src.data.data import SpamDatasetDataModule
from src.models.model import AwesomeSpamClassificationModel

warnings.filterwarnings("ignore")


@hydra.main(
    version_base=None, config_path="../../config", config_name="default_config.yaml"
)
def main(config: DictConfig):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for training.")

    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    torch.manual_seed(config.train.seed)

    wandb_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_key)
    wandb.init(project="test-project", entity="mlops_project_dtu", config=config)
    wandb_logger = WandbLogger(project="test-project", config=config)

    model = AwesomeSpamClassificationModel(config=config)
    model.to(device)

    data_module = SpamDatasetDataModule(config=config)
    data_module.setup()

    trainer = Trainer(
        max_epochs=config.train.epochs, logger=wandb_logger, check_val_every_n_epoch=1
    )
    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )
    trainer.save_checkpoint(
        os.path.join(config.model.model_output_dir, config.model.model_name_local)
    )
    model.save_model_cloud()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
