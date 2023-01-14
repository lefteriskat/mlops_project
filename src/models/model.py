# Importing the libraries needed
import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, DistilBertModel, DistilBertTokenizer

# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")


OUTPUT_SIZE = 2
INPUT_SIZE = 768
DROP_P = 0.2


class AwesomeSpamClassificationModel(torch.nn.Module):
    def __init__(self, input_size, output_size, drop_p=0.2):
        super(AwesomeSpamClassificationModel, self).__init__()
        # self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # self.pre_classifier = torch.nn.Linear(input_size, input_size)
        # self.dropout = torch.nn.Dropout(drop_p)
        # self.classifier = torch.nn.Linear(input_size, output_size)
        self.model = BertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            torchscript=True,
            num_labels=output_size,
        )

    def forward(self, input_ids, attention_mask):
        # output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        # hidden_state = output_1[0]
        # pooler = hidden_state[:, 0]
        # pooler = self.pre_classifier(pooler)
        # pooler = torch.nn.ReLU()(pooler)
        # pooler = self.dropout(pooler)
        # output = self.classifier(pooler)

        return self.model(input_ids=input_ids, attention_mask=attention_mask)


def calculate_accuracy(output, targets):
    n_correct = (output.max(1)[1] == targets).sum().item()
    return n_correct


def train(model, trainloader, loss_function, optimizer=None, epochs=2, print_every=2):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    steps = 0

    for epoch in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for data, mask, targets in trainloader:
            steps += 1
            outputs = model.forward(data, mask)
            outputs = outputs[0]
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()

            n_correct += calculate_accuracy(outputs, targets)
            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if steps % print_every == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                print(f"Training Loss per {print_every} steps: {loss_step}")
                print(f"Training Accuracy per {print_every} steps: {accu_step}")

            optimizer.zero_grad()
            loss.backward()
            # # When using GPU
            optimizer.step()

    print(f"The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}")
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return


def validate(model, loss_function, testloader):
    model.eval()
    n_correct = 0
    n_wrong = 0
    total = 0
    steps = 0
    with torch.no_grad():
        for data, mask, targets in testloader:
            steps += 1
            outputs = model.forward(data, mask)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            n_correct += calculate_accuracy(outputs, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if steps % 100 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                # print(f"Validation Loss per 100 steps: {loss_step}")
                # print(f"Validation Accuracy per 100 steps: {accu_step}")
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    # print(f"Validation Loss Epoch: {epoch_loss}")
    # print(f"Validation Accuracy Epoch: {epoch_accu}")

    return epoch_accu
