import os
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer


class SpamDataset(Dataset):
    def __init__(self, config: DictConfig, type: str, data_path: str):

        if type == "train":
            dataset_path = os.path.join(data_path, "spam_train.csv")
        elif type == "validation":
            dataset_path = os.path.join(data_path, "spam_validation.csv")
        elif type == "test":
            dataset_path = os.path.join(data_path, "spam_test.csv")
        else:
            raise Exception(f"Unknown Dataset type: {type}")

        dataset = pd.read_csv(dataset_path, encoding="latin-1")

        self.data = dataset["original_message"]
        self.targets = dataset["message_type"]
        self.config = config

        if config.data.tokanizer == "DistilBertTokenizer":
            tokanizer = DistilBertTokenizer
        self.tokenizer = tokanizer.from_pretrained(
            config.data.model_tokanizer, do_lower_case=True
        )

    def __getitem__(self, index):
        content = self.data[index]
        inputs = self.tokenizer.encode_plus(
            content,
            None,
            add_special_tokens=True,
            max_length=self.config.data.tokanizer_max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(self.targets[index], dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)


class SpamDatasetDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.data_path_processed = os.path.join(config.data.path, "processed")
        self.config = config

    def prepare_data(self) -> None:
        if not os.path.isdir(self.data_path):
            raise Exception("data is not prepared")

    def setup(self) -> None:
        self.train_set = SpamDataset(
            config=self.config, type="train", data_path=self.data_path_processed
        )
        self.test_set = SpamDataset(
            config=self.config, type="test", data_path=self.data_path_processed
        )
        self.vallidation_set = SpamDataset(
            config=self.config, type="validation", data_path=self.data_path_processed
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.config.train.batch_size,
            num_workers=self.config.train.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.config.train.batch_size,
            num_workers=self.config.train.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.vallidation_set,
            batch_size=self.config.train.batch_size,
            num_workers=self.config.train.num_workers,
        )
