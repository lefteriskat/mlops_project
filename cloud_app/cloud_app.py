from fastapi import FastAPI
from http import HTTPStatus

from google.cloud import storage
from transformers import DistilBertTokenizer
import torch
import io

BUCKET_NAME = "mlops_trained_model_05"
MODEL_FILE = "model_scripted.pt"
TOKENIZER_MAX_LEN = 10

app = FastAPI()

client = storage.Client().create_anonymous_client()
bucket = client.bucket(BUCKET_NAME)
blob = bucket.blob(MODEL_FILE).download_as_bytes()
buffer = io.BytesIO(blob)
my_model = torch.jit.load(buffer)
my_model.eval()


def prediction(request):
    """will to stuff to your request"""
    data = request
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-cased", do_lower_case=True, max_len=TOKENIZER_MAX_LEN
    )
    input_data = tokenizer.encode_plus(
        data,
        None,
        add_special_tokens=True,
        max_length=TOKENIZER_MAX_LEN,
        pad_to_max_length=True,
        return_token_type_ids=True,
        truncation=True,
    )

    ids = input_data["input_ids"]
    mask = input_data["attention_mask"]
    data_tokenized = (
        torch.tensor(ids, dtype=torch.long).unsqueeze(0),
        torch.tensor(mask, dtype=torch.long).unsqueeze(0),
        torch.tensor(0, dtype=torch.long).unsqueeze(0),
    )

    (logits,) = my_model(data_tokenized)
    preds = torch.argmax(logits, dim=1)
    if preds.item() == 0:
        return "No"
    else:
        return "Yes"


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.get("/check_if_spam")
def read_item(sms_text: str):

    x = prediction(sms_text)
    return {"is_spam": x}
