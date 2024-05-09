import numpy as np
import pandas as pd
import torch


def set_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(path):
    data = pd.read_csv(path, header=None)
    images = data.iloc[:, 1:]  # pixels
    labels = data.iloc[:, 0].values  # dice values
    return np.array(images), np.array(labels)
