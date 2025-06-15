import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


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


def flatten_model(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def unflatten_model(model, flat_params):
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        p.data = flat_params[pointer:pointer + numel].view_as(p.data).clone()
        pointer += numel


def get_dataloader(path, batch_size=32, shuffle=True):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_local_model(model, dataloader, epochs, device):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for _ in range(epochs):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()


def compute_accuracy(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            pred = (output >= 0.5).float().squeeze()
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total


def federated_learning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    d1_train = "datasets/Local1/train"
    d1_val = "datasets/Local1/val"
    d2_train = "datasets/Local2/train"
    d2_val = "datasets/Local2/val"
    dtest = "datasets/test"

    # Count samples
    c1_count = len(os.listdir(os.path.join(d1_train, "normal"))) + len(os.listdir(os.path.join(d1_train, "opacity")))
    c2_count = len(os.listdir(os.path.join(d2_train, "normal"))) + len(os.listdir(os.path.join(d2_train, "opacity")))
    total = c1_count + c2_count
    c1 = c1_count / total
    c2 = c2_count / total

    # Global model
    global_model = build_cnn().to(device)
    w_global = flatten_model(global_model)
    b_global = torch.zeros_like(global_model[9].bias.data)

    rounds = 5
    local_epochs = 10

    for r in range(rounds):
        print(f"\nRound {r+1}")

        # -------- Client 1 --------
        model1 = build_cnn().to(device)
        unflatten_model(model1, w_global)
        model1[9].bias.data = b_global.clone()

        train_loader1 = get_dataloader(d1_train)
        val_loader1 = get_dataloader(d1_val, shuffle=False)
        train_local_model(model1, train_loader1, local_epochs, device)
        w1 = flatten_model(model1)
        b1 = model1[9].bias.data.clone()
        acc1 = compute_accuracy(model1, val_loader1, device)

        # -------- Client 2 --------
        model2 = build_cnn().to(device)
        unflatten_model(model2, w_global)
        model2[9].bias.data = b_global.clone()
        train_loader2 = get_dataloader(d2_train)
        val_loader2 = get_dataloader(d2_val, shuffle=False)
        train_local_model(model2, train_loader2, local_epochs, device)
        w2 = flatten_model(model2)
        b2 = model2[9].bias.data.clone()
        acc2 = compute_accuracy(model2, val_loader2, device)

        # -------- Weighted Aggregation --------
        w_avg = w1 * c1 + w2 * c2
        b_avg = b1 * c1 + b2 * c2
        w_global = w_avg
        b_global = b_avg

        print(f"Local1 Val Accuracy: {round(acc1 * 100, 2)}%")
        print(f"Local2 Val Accuracy: {round(acc2 * 100, 2)}%")
        print("w:", w_avg[:5], "...")
        print("b:", b_avg)

    # -------- Final Test --------
    print("\nFinal Evaluation on Test Set:")
    test_loader = get_dataloader(dtest, shuffle=False)
    global_final_model = build_cnn().to(device)
    unflatten_model(global_final_model, w_global)
    global_final_model[9].bias.data = b_global.clone()

    acc = compute_accuracy(global_final_model, test_loader, device)
    print(f"Test Accuracy: {round(acc * 100, 2)}%")
    torch.save(global_final_model.state_dict(), "federated_cnn.pth")


federated_learning()
