import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetwork(nn.Module):
    """Defines a fully connected neural network."""
    def __init__(self, input_size1, hidden_size=60):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size1, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# Example usage (for testing purposes)
if __name__ == "__main__":
    input_size = 8  # Number of features
    model = NeuralNetwork(input_size)
    sample_input = torch.randn(1, input_size)
    output = model(sample_input)
    print(output)
