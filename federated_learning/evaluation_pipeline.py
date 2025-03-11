import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
import pandas as pd
from neural_network import NeuralNetwork
from data_handling import DataPreprocessor, DiabetesDataset


def evaluate_model(model1, test_loader1):
    """Evaluates the model on test data and computes key performance metrics."""
    model1.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader1:
            outputs = model1(inputs)
            predicted = (outputs > 0.36).float()
            y_true.extend(labels.numpy().flatten())
            y_pred.extend(predicted.numpy().flatten())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return accuracy, precision, recall, f1


# Example usage (for testing purposes)
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

    # Load trained model
    model.load_state_dict(torch.load("federated_model.pth"))
    model.eval()

    evaluate_model(model, test_loader)

