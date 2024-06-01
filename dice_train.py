import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import neptune

import utilities 
import  my_model


def main():
    DICE_PATH = "data/dice.csv"
    # Create a Neptune run object
    run = neptune.init_run(
        project="jamal-workspace/dice-classification",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MDRkOGU3YS0zYjlmLTRiNWMtODQ1NS0yYmY4OGQ5YjA2YTkifQ==",
    )
    parameters = {
        "learning_rate": 0.001,
        "batch_size": 36,
        "n_epochs": 20,
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
        stratify=labels_tensor,
    )

    train_images = train_images.unsqueeze(1)
    test_images = data_normalized.unsqueeze(1)
    train_labels = train_labels - 1
    test_labels = labels_tensor - 1
    # Create a DataLoader
    train_data = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(
        train_data, batch_size=parameters["batch_size"], shuffle=True
    )
    test_data = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(
        test_data, batch_size=parameters["batch_size"], shuffle=True
    )

    # Define the model

    dice_classifier = my_model.DiceClassifier()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dice_classifier.parameters(), lr=parameters["learning_rate"])
    # run["training/loss_function"] = str(criterion)
    # run["training/optimizer"] = str(optimizer)

    conf_matrix_train = None
    train_y_true = []
    train_y_pred = []

    # Train the model
    for epoch in range(parameters["n_epochs"]):
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
        train_precision = 100 * precision_score(
            train_y_true, train_y_pred, average="weighted"
        )

        print(
            f'Epoch {epoch + 1}/{parameters["n_epochs"]}, Loss: {train_loss / len(train_loader):.4f}'
            f", Accuracy: {train_accuracy:.2f}%",
            f", Precision: {train_precision:.2f}%",
        )
        # run["training/precision"].log(train_precision)
        # run["training/accuracy"].log(train_accuracy)
        # run["training/loss"].log(train_loss / len(train_loader))
    conf_matrix_train = confusion_matrix(train_y_true, train_y_pred)

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
    test_precision = 100 * precision_score(y_true, y_pred, average="weighted")
    conf_matrix_test = confusion_matrix(y_true, y_pred)

    print(
        f"Accuracy on test set: {test_accuracy:.2f}%",
        f"Precision on test set: {test_precision:.2f}%",
    )
    # run["testing/test_accuracy"] = test_accuracy
    # run["testing/test_precision"] = test_precision

    utilities.plot_confusion_matrix(
        conf_matrix_train,
        y_true,
        "Confusion Matrix - Test Set",
        "confusion_matrix_test.png",
    )
    run["training/confusion_matrix"].upload('confusion_matrix_train.png')

    utilities.plot_confusion_matrix(
        conf_matrix_test,
        y_true,
        "Confusion Matrix - Test Set",
        "confusion_matrix_test.png",
    )
    run["testing/confusion_matrix"].upload('confusion_matrix_test.png')

    # Save the model
    torch.save(dice_classifier.state_dict(), "/app/output/dice_classifier.pth")
    # run.stop()


if __name__ == "__main__":
    main()
