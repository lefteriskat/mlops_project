#!/bin/bash
python -u src/models/train_model.py
gsutil -m cp models/trained_model.pkl gs://mlops_trained_model_05
gsutil -m cp models/trained_model.pt gs://mlops_trained_model_05