import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# Load CSVs
df1 = pd.read_csv('datalocal1.csv')
df2 = pd.read_csv('datalocal2.csv')

# Combine and prepare data
df = pd.concat([df1, df2], ignore_index=True)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.long)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.long)

# Split training into two clients
client_data = [
    (X_train[:len(X_train)//2], y_train[:len(y_train)//2]),
    (X_train[len(X_train)//2:], y_train[len(y_train)//2:])
]
input_dim = X_train.shape[1]

# Define model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)

# Encryption placeholders
def encrypt_model_weights(weights): return weights
def decrypt_model_weights(weights): return weights

# Federated training
def federated_train(X_test_tensor, y_test_tensor):
    global_model = LogisticRegression(input_dim)
    rounds = 10

    for rnd in range(rounds):
        client_weights = []
        for X, y in client_data:
            local_model = LogisticRegression(input_dim)
            local_model.load_state_dict(decrypt_model_weights(global_model.state_dict()))  ####### decrypt
            optimizer = optim.SGD(local_model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            local_model.train()
            for _ in range(20):
                optimizer.zero_grad()
                output = local_model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
            encrypted_weights = encrypt_model_weights(local_model.state_dict())
            client_weights.append(encrypted_weights)

        # Averaging model weights
        avg_weights = {}
        for key in client_weights[0].keys():
            avg_weights[key] = sum(client[key] for client in client_weights) / len(client_weights)
        global_model.load_state_dict(avg_weights)

    # Evaluation
    global_model.eval()
    with torch.no_grad():
        output = global_model(X_test_tensor)
        preds = torch.argmax(output, dim=1)
        acc = accuracy_score(y_test_tensor.numpy(), preds.numpy())
        cm = confusion_matrix(y_test_tensor.numpy(), preds.numpy())

    return global_model, acc, cm

# Run
trained_model, test_accuracy, conf_matrix = federated_train(X_test, y_test)
print("✅ Test Accuracy:", test_accuracy)
print("🧮 Confusion Matrix:\n", conf_matrix)

# --- Load new test CSV ---
df_test = pd.read_csv("data_test.csv")

# Assume last column is the label
X_new = df_test.iloc[:, :-1].values
y_new = df_test.iloc[:, -1].values

# Normalize using same scaler
X_new_scaled = scaler.transform(X_new)

# Convert to torch tensors
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)
y_new_tensor = torch.tensor(y_new, dtype=torch.long)

# --- Evaluate on new test data ---
trained_model.eval()
with torch.no_grad():
    output = trained_model(X_new_tensor)
    preds = torch.argmax(output, dim=1)
    acc = accuracy_score(y_new_tensor.numpy(), preds.numpy())
    cm = confusion_matrix(y_new_tensor.numpy(), preds.numpy())

print("📊 New Test Accuracy:", acc)
print("📉 New Confusion Matrix:\n", cm)
