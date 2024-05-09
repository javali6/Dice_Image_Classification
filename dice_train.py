import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import utilities


def main():
    DICE_PATH = 'data/dice.csv'
    # Create a Neptune run object
    # run = neptune.init_run(
    #     project="jamal-workspace/dice-classification",
    #     api_token="==",
    # )
    parameters = {
        "dense_units": 64,
        "kernel_size": 3,
        "num_filters": 64,
        "pool_size": 2,
        "learning_rate": 0.001,
        "batch_size": 36,
        "n_epochs": 20,
        "padding": 1
    }
    # run["model/parameters"] = parameters
    # Load the data
    data, labels = utilities.load_data(DICE_PATH)
    data = data.reshape((60000, 28, -1))
    data_normalized = torch.tensor(data / 255.0, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    train_images, test_images, train_labels, test_labels = train_test_split(
        data_normalized,
        labels_tensor,
        test_size=0.2,
        random_state=42,
        stratify=labels_tensor)

    train_images = train_images.unsqueeze(1)
    test_images = data_normalized.unsqueeze(1)
    train_labels = train_labels - 1
    test_labels = labels_tensor - 1
    # Create a DataLoader
    train_data = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_data, batch_size=parameters['batch_size'], shuffle=True)
    test_data = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_data, batch_size=parameters['batch_size'], shuffle=True)

    # Define the model
    class DiceClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=parameters['kernel_size']
                                   , padding=parameters['padding'])
            self.conv2 = nn.Conv2d(32, 64, kernel_size=parameters['kernel_size']
                                   , padding=parameters['padding'])
            self.pool = nn.MaxPool2d(parameters['pool_size'], 2)
            self.fc1 = nn.Linear(parameters['num_filters'] * 7 * 7, parameters['dense_units'])
            self.fc2 = nn.Linear(parameters['dense_units'], 6)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 7 * 7)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    dice_classifier = DiceClassifier()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dice_classifier.parameters(), lr=parameters['learning_rate'])
    # run["training/loss_function"] = str(criterion)
    # run["training/optimizer"] = str(optimizer)

    conf_matrix_train = None
    train_y_true = []
    train_y_pred = []

    # Train the model
    for epoch in range(parameters['n_epochs']):
        train_loss = 0.0
        train_correct = 0
        train_samples = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = dice_classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_samples += labels.size(0)
            train_loss += loss.item()
            train_y_true.extend(labels.cpu().numpy())
            train_y_pred.extend(predicted.cpu().numpy())

        train_accuracy = 100 * train_correct / train_samples
        train_precision = 100 * precision_score(train_y_true, train_y_pred, average='weighted')
        conf_matrix_train = confusion_matrix(train_y_true, train_y_pred)

        print(
            f'Epoch {epoch + 1}/{parameters["n_epochs"]}, Loss: {train_loss / len(train_loader):.4f}'
            f', Accuracy: {train_accuracy:.2f}%', f', Precision: {train_precision:.2f}%')
        # run["training/precision"].log(train_precision)
        # run["training/accuracy"].log(train_accuracy)
        # run["training/loss"].log(train_loss / len(train_loader))

    # Test the model
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = dice_classifier(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    test_accuracy = 100 * correct / total
    test_precision = 100 * precision_score(y_true, y_pred, average='weighted')
    conf_matrix_test = confusion_matrix(y_true, y_pred)

    print(f'Accuracy on test set: {test_accuracy:.2f}%', f'Precision on test set: {test_precision:.2f}%')
    # run["testing/test_accuracy"] = test_accuracy
    # run["testing/test_precision"] = test_precision

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(train_y_true),
                yticklabels=np.unique(train_y_true))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - Train Set')
    plt.savefig('confusion_matrix_train.png')
    # run["training/confusion_matrix"].upload('confusion_matrix_train.png')

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true),
                yticklabels=np.unique(y_true))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - Test Set')
    plt.savefig('confusion_matrix_test.png')
    # run["testing/confusion_matrix"].upload('confusion_matrix_test.png')

    # Save the model
    torch.save(dice_classifier.state_dict(), './dice_classifier.pth')
    # run.stop()


if __name__ == "__main__":
    main()
