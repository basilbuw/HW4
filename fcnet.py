import torch
from torch import nn
import numpy as np
import random
import torchvision
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import pickle
import json
from collections import defaultdict
from tqdm import tqdm
import torch.optim as optim
from math import floor

torch.manual_seed(0)
DEVICE = torch.device("mps")
FILENAME = "fc_results.json"
print(DEVICE)

def load_data(batch_size):
    train_dataset = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
    train_dataset, val_dataset = random_split(train_dataset, [int(0.9 * len(train_dataset)), int( 0.1 * len(train_dataset))])

    # Create separate dataloaders for the train, test, and validation set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


class FCNet(nn.Module):
    def __init__(self, M=1000):
        super(FCNet, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.05),
            nn.Linear(3072, M, bias=True),
            nn.ReLU(),
            nn.Linear(M, 10, bias=True)
        )

    def forward(self, x):
        return self.model(x)


def compute_accuracy(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return round(correct / total, 3)


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):

    model = model.to(DEVICE)

    train_acc_list = []
    val_acc_list = []

    for e in tqdm(range(num_epochs)):
        running_loss = 0.0
        for batch, labels in train_loader:
            batch, labels = batch.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {e + 1}, loss: {running_loss / len(train_loader)}')
        # Compute and print accuracy
        train_acc = compute_accuracy(model, train_loader)
        val_acc = compute_accuracy(model, val_loader)
        print(f'Training accuracy: {train_acc}, Validation accuracy: {val_acc}')
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
    print('Finished Training')
    return train_acc_list, val_acc_list, model

def random_search(

        lr_values=(-3, -5),
        batch_size_values=(4, 8),
        M_values=(800, 2000),
        num_epochs=25
):

    # If results file exists, load it. Otherwise, create an empty results dict.
    if os.path.exists(FILENAME):
        with open(FILENAME, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    lr = 10**(random.uniform(lr_values[0], lr_values[1]))
    batch_size = int(2**(random.uniform(batch_size_values[0], batch_size_values[1])))
    M = random.randint(M_values[0], M_values[1])

    # Iterate over each combination of hyperparameters
    train_loader, val_loader, test_loader = load_data(batch_size=batch_size)
    hyperparameters = str(("fullyconnected", lr, batch_size, M))

    if hyperparameters not in results:
        # If this set of hyperparameters hasn't been trained yet, do it now
        print(f'Training with hyperparameters: {hyperparameters}')
        # Create and train the model
        model = FCNet(M=M)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        train_acc, val_acc, model = train(model, train_loader, val_loader, criterion, optimizer, num_epochs)

        #### save current results ###
        results[hyperparameters] = {}
        results[hyperparameters]['training_accuracy'] = train_acc
        results[hyperparameters]['validation_accuracy'] = val_acc
        results[hyperparameters]['max_validation_accuracy'] = max(val_acc)
        results[hyperparameters]['final_test_accuracy'] = compute_accuracy(model, test_loader)
        with open(FILENAME, 'w') as f:
            print("saved results")
            json.dump(results, f)

    # After all models have been trained, find the model with the highest validation accuracy
    best_hyperparameters = max(results, key=lambda x: results[x]['max_validation_accuracy'])
    print(f'Best hyperparameters: {best_hyperparameters}, with validation accuracy: {results[best_hyperparameters]["max_validation_accuracy"]}')
    return results


print(DEVICE)
for _ in range(5):
    random_search()
