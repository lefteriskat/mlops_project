import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer

from src import _PATH_DATA


class Custom_Dataset(Dataset):
    def __init__(
        self,
        type,
        tokenizer=DistilBertTokenizer.from_pretrained(
            "distilbert-base-cased", do_lower_case=True
        ),
        max_len=10,
    ):

        path = os.path.normpath(os.path.join(_PATH_DATA, "processed"))
        if type == "train":
            dataset_path = os.path.join(path, "spam_train.csv")
        elif type == "validation":
            dataset_path = os.path.join(path, "spam_validation.csv")
        elif type == "test":
            dataset_path = os.path.join(path, "spam_test.csv")
        else:
            raise Exception(f"Unknown Dataset type: {type}")

        dataset = pd.read_csv(dataset_path, encoding="latin-1")

        self.data = dataset["original_message"]
        self.targets = dataset["message_type"]
        self.len = len(dataset)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        content = self.data[index]
        inputs = self.tokenizer.encode_plus(
            content,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
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
        return self.len


if __name__ == "__main__":
    dataset_train = Custom_Dataset(type="train")
    dataset_validation = Custom_Dataset(type="validation")
    dataset_test = Custom_Dataset(type="test")

    print(dataset_train.data.shape)
    print(dataset_train.targets.shape)
    print(dataset_validation.data.shape)
    print(dataset_validation.targets.shape)
    print(dataset_test.data.shape)
    print(dataset_test.targets.shape)
