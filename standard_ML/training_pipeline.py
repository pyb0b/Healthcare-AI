import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def train_model(model1, train_loader1, test_loader1, num_epochs=50, learning_rate=0.001):
    """Trains the neural network model and evaluates it on the test set."""
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for classification
    optimizer = optim.Adam(model1.parameters(), lr=learning_rate)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(num_epochs):
        # Training phase
        model1.train()
        epoch_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader1:
            optimizer.zero_grad()
            outputs = model1(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = 100 * correct / total
        train_losses.append(epoch_loss / len(train_loader1))
        train_accuracies.append(train_accuracy)

        # Validation phase
        model1.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader1:
                outputs = model1(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        test_accuracy = 100 * correct / total
        test_losses.append(test_loss / len(test_loader1))
        test_accuracies.append(test_accuracy)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

    # Plot loss trends
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Test Loss")
    plt.show()

    return model1


# Example usage (for testing purposes)
if __name__ == "__main__":
    from neural_network import NeuralNetwork
    from data_handling import DataPreprocessor, DiabetesDataset
    from torch.utils.data import DataLoader
    import pandas as pd

    df = pd.read_csv("dataset.csv")
    preprocessor = DataPreprocessor(df)
    preprocessor.replace_zeros_with_median()
    preprocessor.normalize_features()
    X_train, X_test, y_train, y_test = preprocessor.split_data()

    train_dataset = DiabetesDataset(X_train, y_train)
    test_dataset = DiabetesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = 8
    model = NeuralNetwork(input_size)
    trained_model = train_model(model, train_loader, test_loader, 3)
    torch.save(model.state_dict(), "trained_model.pth")

