#!/bin/bash
python -u src/models/train_model.py
gsutil -m cp models/model_scripted.pt gs://mlops_trained_model_05