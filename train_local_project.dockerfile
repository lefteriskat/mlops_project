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
COPY data/ data/

ENV PYTHONPATH /
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]

