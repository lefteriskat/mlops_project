from google.cloud import storage
from transformers import DistilBertTokenizer
import torch
import pickle

BUCKET_NAME = "gs://mlops_trained_model_05"
MODEL_FILE = "trained_model.ckpt"
TOKENIZER_MAX_LEN = 10

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
my_model = pickle.loads(blob.download_as_string())


def prediction(request):
    """will to stuff to your request"""
    request_json = request.get_json()
    if request_json and "input_data" in request_json:
        data = request_json["input_data"]
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

        (logits,) = model(data_tokenized)
        preds = torch.argmax(logits, dim=1)
        if preds.item() == 0:
            print(f'Belongs to class: {"ham"}')
        else:
            print(f'Belongs to class: {"spam"}')
    else:
        return "No input data received"
