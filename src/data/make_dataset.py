import os
import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split


@hydra.main(
    version_base=None, config_path="../../config", config_name="default_config.yaml"
)
def main(config: DictConfig):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    path_csv = os.path.normpath(
        os.path.join(config.data.path, config.data.path_input, config.data.name_file)
    )
    dataset = pd.read_csv(path_csv, encoding="latin-1")

    # Dropping unwanted columns
    dataset.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True, axis=1)
    # Chaning the labels for convinience
    dataset["v1"].replace({"ham": 0, "spam": 1}, inplace=True)
    # Changing the column names for better
    dataset.rename(
        {"v1": "message_type", "v2": "original_message"}, axis=1, inplace=True
    )

    path_csv_save = os.path.normpath(
        os.path.join(config.data.path, config.data.path_interim, config.data.name_file)
    )
    dataset.to_csv(path_csv_save)

    validation_size = config.data.test_size + config.data.validation_size
    validation_test_percentage = config.data.test_size / validation_size

    train, validation = train_test_split(
        dataset,
        random_state=config.data.random_state,
        test_size=validation_size,
        stratify=dataset["message_type"],
    )
    test, validation = train_test_split(
        validation,
        random_state=config.data.random_state,
        test_size=validation_test_percentage,
        stratify=validation["message_type"],
    )

    only_name = config.data.name_file.split(".")[0]
    path_csv_train = os.path.normpath(
        os.path.join(
            config.data.path,
            config.data.path_processed,
            "".join([only_name, "_train.csv"]),
        )
    )
    path_csv_validation = os.path.normpath(
        os.path.join(
            config.data.path,
            config.data.path_processed,
            "".join([only_name, "_validation.csv"]),
        )
    )
    path_csv_test = os.path.normpath(
        os.path.join(
            config.data.path,
            config.data.path_processed,
            "".join([only_name, "_test.csv"]),
        )
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
