import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# ---- CNN Model ----
def build_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 32 * 32, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )

# ---- Utilities ----
def flatten_model(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def unflatten_model(model, flat_params):
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        p.data = flat_params[pointer:pointer + numel].view_as(p.data).clone()
        pointer += numel

def get_dataloader(path, batch_size=int(32)):
    transform = transforms.Compose([
        transforms.Grayscale(),  # Ensure single channel
        transforms.Resize((int(128), int(128))),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)  # , pin_memory=False, num_workers=0)

# ---- Local Training ----
def train_one_local_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.float().unsqueeze(int(1)).to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

def train_local_model(model, data_path, epochs, device):
    dataloader = get_dataloader(data_path, int(32))
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(0.001))
    for _ in range(epochs):
        train_one_local_epoch(model, dataloader, criterion, optimizer, device)
    return flatten_model(model)

# ---- Global Testing ----
def test_model(model, test_loader, device):
    model.eval()
    correct = total = int(0)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = (outputs >= float(0.5)).float().squeeze()
            correct += (predicted == labels).sum().item()
            total += labels.size(int(0))
    return correct / total

# ---- Main Federated Logic ----
def federated_learning():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    local_paths = ["datasets/Local1/train", "datasets/Local2/train"]
    local_sizes = [len(os.listdir(os.path.join(p, 'normal'))) + len(os.listdir(os.path.join(p, 'opacity'))) for p in local_paths]
    total = sum(local_sizes)
    weights = [s / total for s in local_sizes]

    global_model = build_cnn()
    rounds = 2
    local_epochs = int(10)

    for r in range(rounds):
        print(f"\nRound {r+1}")
        local_weights = []
        for i, path in enumerate(local_paths):
            local_model = build_cnn()
            unflatten_model(local_model, flatten_model(global_model))
            updated_weights = train_local_model(local_model, path, local_epochs, device)
            local_weights.append(updated_weights * weights[i])
        
        new_flat = sum(local_weights)
        unflatten_model(global_model, new_flat)
    
    print("\nEvaluating global model on test set...")
    print("\n model parameters:", new_flat)
    test_loader = get_dataloader("datasets/test")
    acc = test_model(global_model, test_loader, device)
    print(f"Test Accuracy: {round(acc * 100, 2)}%")

# ---- Run ----
federated_learning()
