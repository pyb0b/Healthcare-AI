import torch

# Number of training samples per model
n1 = 537
n2 = 231
total_samples = n1 + n2  # Total samples

# Compute weights
w1 = n1 / total_samples  # Weight for model1
w2 = n2 / total_samples  # Weight for model2

# Load model parameters
model1 = torch.load("model1.pth")
model2 = torch.load("model2.pth")

# Ensure both models have the same architecture
assert model1.keys() == model2.keys(), "Models must have the same architecture"

# Perform weighted averaging
fed_avg_model = {}
for key in model1.keys():
    fed_avg_model[key] = w1 * model1[key] + w2 * model2[key]  # Weighted sum

# Save the federated model
torch.save(fed_avg_model, "federated_model.pth")

print("Federated model saved successfully as federated_model.pth")
