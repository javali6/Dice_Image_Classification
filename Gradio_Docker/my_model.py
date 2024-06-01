import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceClassifier(nn.Module):
    model_parameters = {
        "dense_units": 64,
        "kernel_size": 3,
        "num_filters": 64,
        "pool_size": 2,
        "padding": 1,
    }
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            1,
            32,
            kernel_size=self.model_parameters["kernel_size"],
            padding=self.model_parameters["padding"],
        )
        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=self.model_parameters["kernel_size"],
            padding=self.model_parameters["padding"],
        )
        self.pool = nn.MaxPool2d(self.model_parameters["pool_size"], 2)
        self.fc1 = nn.Linear(
            self.model_parameters["num_filters"] * 7 * 7, self.model_parameters["dense_units"]
        )
        self.fc2 = nn.Linear(self.model_parameters["dense_units"], 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x