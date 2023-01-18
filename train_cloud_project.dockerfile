# Base image
FROM python:3.10.0-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc wget curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

ENV CLOUDSDK_INSTALL_DIR /usr/local/gcloud/
RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

COPY . /mlops_project

WORKDIR /mlops_project
RUN pip install -r requirements.txt --no-cache-dir

RUN dvc pull

ARG WANDB_API_KEY=local
ENV WANDB_API_KEY ${WANDB_API_KEY}


ENV PYTHONPATH /
RUN chmod u+x training_in_cloud.sh
ENTRYPOINT ["./training_in_cloud.sh"]

