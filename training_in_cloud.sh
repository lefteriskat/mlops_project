#!/bin/bash
python -u src/models/train_model.py
gsutil -m cp models/trained_model.pkl gs://mlops_trained_model_05