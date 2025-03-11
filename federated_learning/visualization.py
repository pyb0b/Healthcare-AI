import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from neural_network import NeuralNetwork
from data_handling import DataPreprocessor, DiabetesDataset
from torch.utils.data import DataLoader
import pandas as pd


def plot_training_metrics(losses, accuracies):
    """Plots training loss and accuracy trends."""
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color='tab:red')
    ax1.plot(losses, color='tab:red', label="Loss")
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color='tab:blue')
    ax2.plot(accuracies, color='tab:blue', label="Accuracy")
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()
    plt.title("Training Loss & Accuracy")
    plt.show()


def plot_confusion_matrix(y_true1, y_pred1):
    """Plots the confusion matrix using real predictions."""
    cm = confusion_matrix(y_true1, y_pred1)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Diabetes", "Diabetes"],
                yticklabels=["No Diabetes", "Diabetes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


# Example usage (for real model evaluation)
if __name__ == "__main__":
    df = pd.read_csv("datalocal2.csv")
    preprocessor = DataPreprocessor(df)
    preprocessor.replace_zeros_with_median()
    preprocessor.normalize_features()
    _, X_test, _, y_test = preprocessor.split_data()

    test_dataset = DiabetesDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = 8
    model = NeuralNetwork(input_size)
    model.load_state_dict(torch.load("model2.pth"))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.36).float()
            y_true.extend(labels.numpy().flatten())
            y_pred.extend(predicted.numpy().flatten())

    plot_confusion_matrix(y_true, y_pred)
