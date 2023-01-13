import torch
from model import  AwesomeSpamClassificationModel
from torch.utils.data import DataLoader
from model import validate
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = AwesomeSpamClassificationModel(
        checkpoint["input_size"], checkpoint["output_size"],
    )
    model.load_state_dict(checkpoint["state_dict"])

    return model

def predict(model_checkpoint, test_data):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)
    print(test_data)

    # TODO: Implement evaluation logic here
    # model = torch.load(model_checkpoint)
    model = load_checkpoint(model_checkpoint)
    test_set = DataLoader(SmsSpam(train=False), batch_size=64, shuffle=True)

    loss_function = torch.nn.CrossEntropyLoss()
    accuracy = validate(model,loss_function, test_set)

    print(f"Accuracy: {accuracy.item()*100}%")



if __name__ == "__main__":
    predict()