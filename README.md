Natural Language Processing - SMS Spam Classification
==============================

Final Project for DTU Machine Learning Operations Course January 2023

[![Test flake8](https://https://github.com/lefteriskat/mlops_project/actions/workflows/flake8.yml/badge.svg)](https://https://github.com/lefteriskat/mlops_project/actions/workflows/flake8.yml)
[![Run tests](https://https://github.com/lefteriskat/mlops_project/actions/workflows/tests.yml/badge.svg?branch=main)](https://https://github.com/lefteriskat/mlops_project/actions/workflows/tests.yml)
[![Test isort](https://https://github.com/lefteriskat/mlops_project/actions/workflows/isort.yml/badge.svg)](https://https://github.com/lefteriskat/mlops_project/actions/workflows/isort.yml)
## Project Description

### Overall goal of the project
The goal of this project is to use Natural Language processing and train a neural network wchich successfully classifies sms text as spam or no spam. 

### What framework are you going to use
We plan to use [Transformers](https://github.com/huggingface/transformers) framework.

### How to you intend to include the framework into your project
As a strating point, we are going to use some of the pretrained models offered by transformers. 
Afterward, we are going to fine tune them according to our specific case.
In addition we may include some preprocesssing modules offered by this framework.

### What data are you going to run on (initially, may change)
We are going to use the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
The dataset consists of SMS tagged messages that have been collected for SMS Spam research. It contains 5,574 SMS messages in English, tagged acording being ham (legitimate) or spam.

### What deep learning models do you expect to use 
We are planing to use the [DistilBERT base model](https://huggingface.co/distilbert-base-uncased)  to achieve our goal.We are going to check also [BERT-TINY](https://huggingface.co/prajjwal1/bert-tiny) since our problem is relatively easy and it may also do the job.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
