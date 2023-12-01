import torch
import os
from torch import nn
from torch.utils.data import DataLoader


class NN(nn.Module):

    """
    Implementation of a Neural Network
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, a):
        a = self.flatten(a)
        logits = self.linear_relu_stack(a)
        return logits

    def find_device(self):
        """This is the device to train on."""
        device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
        print(f"Using {device} device")
        return "cpu"


    


if __name__ == '__main__':
    print('Starting NN')
    model = NN()
    print(f'Device Found: {model.find_device()}')
    device = model.find_device()

    x = torch.rand(1, 28, 28, device=device)
    logits = model(x)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f'Predicted Class: {y_pred}')



