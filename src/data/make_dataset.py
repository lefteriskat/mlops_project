# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import zipfile
import pandas
import torch
import numpy as np
import requests 


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    name_file = "spam.csv"
    path_csv = os.path.normpath(os.path.join(input_filepath, name_file))
    dataset = pandas.read_csv(path_csv, encoding='latin-1')

    # Dropping unwanted columns
    dataset.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True, axis=1)
    # Chaning the labels for convinience
    dataset["v1"].replace({"ham": 0, "spam":1}, inplace=True)
    # Changing the column names for better 
    dataset.rename({"v1": "message_type", "v2": "original_message"}, axis=1, inplace=True)
   
    orig_message = dataset[dataset.columns[1]]
    logger.info(orig_message)
 
    name_file_processed = "spam_processed.csv"
    path_csv_save = os.path.normpath(os.path.join(output_filepath, name_file_processed))
    dataset.to_csv(path_csv_save)  

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
