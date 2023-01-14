# -*- coding: utf-8 -*-
import logging
import os
import zipfile
from pathlib import Path

import click
import numpy as np
import pandas
import requests
import torch
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("interim_filepath", type=click.Path())
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, interim_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    name_file = "spam.csv"
    path_csv = os.path.normpath(os.path.join(input_filepath, name_file))
    dataset = pandas.read_csv(path_csv, encoding="latin-1")

    # Dropping unwanted columns
    dataset.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True, axis=1)
    # Chaning the labels for convinience
    dataset["v1"].replace({"ham": 0, "spam": 1}, inplace=True)
    # Changing the column names for better
    dataset.rename({"v1": "message_type", "v2": "original_message"}, axis=1, inplace=True)

    orig_message = dataset[dataset.columns[1]]
    logger.info(orig_message)

    path_csv_save = os.path.normpath(os.path.join(interim_filepath, name_file))
    dataset.to_csv(path_csv_save)

    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.5
    RANDOM_STATE = 2022

    train, validation = train_test_split(
        dataset,
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE,
        stratify=dataset["message_type"],
    )
    test, validation = train_test_split(
        validation,
        random_state=RANDOM_STATE,
        test_size=VALIDATION_SIZE,
        stratify=validation["message_type"],
    )

    only_name = name_file.split(".")[0]
    path_csv_train = os.path.normpath(
        os.path.join(output_filepath, "".join([only_name, "_train.csv"]))
    )
    path_csv_validation = os.path.normpath(
        os.path.join(output_filepath, "".join([only_name, "_validation.csv"]))
    )
    path_csv_test = os.path.normpath(
        os.path.join(output_filepath, "".join([only_name, "_test.csv"]))
    )

    train.to_csv(path_csv_train)
    validation.to_csv(path_csv_validation)
    test.to_csv(path_csv_test)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
