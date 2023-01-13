import model as mymodel
from torch.utils.data import DataLoader
import torch
from src.data.data import Custom_Dataset
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

def train_model():
    model = mymodel.AwesomeSpamClassificationModel(mymodel.INPUT_SIZE,mymodel.OUTPUT_SIZE)
    model.to(device)

    trainloader = DataLoader(Custom_Dataset(type="train"), batch_size=64, shuffle=True)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # Creating the loss function and optimizer
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=mymodel.LEARNING_RATE)

    mymodel.train(model, trainloader, loss_function, optimizer=optimizer, epochs=20, print_every=100)


if __name__ == "__main__":
    train_model()