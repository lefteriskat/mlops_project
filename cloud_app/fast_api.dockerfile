FROM python:3.10-slim

EXPOSE $PORT

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt --no-cache-dir

COPY cloud_app.py cloud_app.py

CMD exec uvicorn cloud_app:app --port $PORT --host 0.0.0.0 --workers 1