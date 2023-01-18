# Base image
FROM python:3.10.0-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


COPY . /mlops_project

WORKDIR /mlops_project
RUN pip install -r requirements.txt --no-cache-dir

RUN dvc pull

ARG WANDB_API_KEY=local
ENV WANDB_API_KEY ${WANDB_API_KEY}


ENV PYTHONPATH /
ENTRYPOINT ["./training_in_cloud.sh"]

