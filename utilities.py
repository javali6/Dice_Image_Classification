import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt


def set_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(path):
    data = pd.read_csv(path, header=None)
    images = data.iloc[:, 1:]  # pixels
    labels = data.iloc[:, 0].values  # dice values
    return np.array(images), np.array(labels)


def plot_confusion_matrix(conf_matrix, labels, title, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(labels),
        yticklabels=np.unique(labels),
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.savefig(filename)
