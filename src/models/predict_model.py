import click
import torch
from torch import cuda
from model import AwesomeSpamClassificationModel, validate
import model as mymodel
from src.data.data import Custom_Dataset
from torch.utils.data import DataLoader


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = AwesomeSpamClassificationModel(mymodel.OUTPUT_SIZE)
    model.load_state_dict(checkpoint)
    return model


@click.command()
@click.argument("model_checkpoint", type=click.Path(exists=True))
def predict(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    # model = torch.load(model_checkpoint)
    model = load_checkpoint(model_checkpoint)
    test_set = DataLoader(Custom_Dataset(type="test"), batch_size=64, shuffle=True)
    loss_function = torch.nn.CrossEntropyLoss()
    accuracy = validate(model, loss_function, test_set)


if __name__ == "__main__":
    predict()
