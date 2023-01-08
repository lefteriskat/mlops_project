# -*- coding: utf-8 -*-
import logging
import os
import zipfile
from pathlib import Path

import click
import numpy as np
import pandas
import torch
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('dataset_path', type=click.Path())
def main(dataset_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Downloading dataset from kaggle')
    path = os.path.join(dataset_path, "raw")

    try:
        import kaggle
    except:
        logger.warning("kaggle no imported")
    
    try:
        kaggle.api.competition_download_files("sms-spam-collection-dataset", path=path)
    except Exception:
        logger.warning("Downloaded failed")

    out_folder_interim = os.path.join(dataset_path, "interim")
    os.makedirs(out_folder_interim, exist_ok=True)
    with zipfile.ZipFile(os.path.join(path, "sms_spam.zip"), 'r') as zip_ref:
        zip_ref.extractall(out_folder_interim)
    
    path_csv = os.path.join(out_folder_interim, "spam.csv")
    dataset = pandas.read_csv(path_csv, encoding='latin-1')

    # Dropping unwanted columns
    dataset.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True, axis=1)
    # Chaning the labels for convinience
    dataset["v1"].replace({"ham": 0, "spam":1}, inplace=True)
    # Changing the column names for better 
    dataset.rename({"v1": "message_type", "v2": "original_message"},axis=1, inplace=True)
   
    message_type = dataset[dataset.columns[0]]
    orig_message = dataset[dataset.columns[1]]
    logger.info(orig_message)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
