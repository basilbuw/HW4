import torch
from torch import nn
import numpy as np
from typing import Tuple, Union, List, Callable
from torch.optim import SGD
import torchvision
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm
import os
import pickle
import json
from collections import defaultdict
from tqdm import tqdm
import torch.optim as optim
from math import floor

def load_data(batch_size):
    train_dataset = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
    train_dataset, val_dataset = random_split(train_dataset, [int(0.9 * len(train_dataset)), int( 0.1 * len(train_dataset))])

    # Create separate dataloaders for the train, test, and validation set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

class FullyConnectedNet(nn.Module):
    def __init__(self, M):
        # here M is the number of hidden layer neurons in the fully connected network
        super(FullyConnectedNet, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, M, bias=True),
            nn.ReLU(),
            nn.Linear(M, 10, bias=True)
        )

    def forward(self, x):
        return self.model(x)


class ConvNet(nn.Module):
    def __init__(self, M=100, k=5, N=14):
        super(ConvNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, M, k),
            nn.ReLU(),
            nn.MaxPool2d(N),
            nn.Flatten(),
            nn.Linear(M * floor((32 - k + 1) // N) ** 2, 10)
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
    return train_acc_list, val_acc_list

def random_search(

        lr_values=[1e-2],
        batch_size_values=[128],
        M_values=[100],
        k_values=[5],
        N_values=[14],
        num_epochs=25
):

    # If results file exists, load it. Otherwise, create an empty results dict.
    if os.path.exists('results.json'):
        with open('results.json', 'r') as f:
            results = json.load(f)
    else:
        results = {}

    # Iterate over each combination of hyperparameters
    for lr in lr_values:
        for batch_size in batch_size_values:
            train_loader, val_loader, test_loader = load_data(batch_size=batch_size)
            for M in M_values:
                for k in k_values:
                    for N in N_values:

                        hyperparameters = str((lr, batch_size, M, k, N))

                        if hyperparameters not in results:
                            # If this set of hyperparameters hasn't been trained yet, do it now
                            print(f'Training with hyperparameters: {hyperparameters}')
                            # Create and train the model
                            model = ConvNet(M=M, k=k, N=N)
                            optimizer = optim.Adam(model.parameters(), lr=lr)
                            criterion = nn.CrossEntropyLoss()
                            train_acc, val_acc = train(model, train_loader, val_loader, criterion, optimizer, num_epochs)

                            #### save current results ###
                            results[hyperparameters] = {}
                            results[hyperparameters]['training_accuracy'] = train_acc
                            results[hyperparameters]['validation_accuracy'] = val_acc
                            results[hyperparameters]['max_validation_accuracy'] = max(val_acc)
                            with open('results.json', 'w') as f:
                                json.dump(results, f)

    # After all models have been trained, find the model with the highest validation accuracy
    best_hyperparameters = max(results, key=lambda x: results[x]['max_validation_accuracy'])
    print(f'Best hyperparameters: {best_hyperparameters}, with validation accuracy: {results[best_hyperparameters]["max_validation_accuracy"]}')
    return results


def simple_model():
    train_loader, val_loader, test_loader = load_data(batch_size=128)
    #fcmodel = FullyConnectedNet(1000)
    convnet = ConvNet(M=400, k=5, N=14)
    optimizer = optim.Adam(convnet.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    train(convnet, train_loader, val_loader, criterion, optimizer, num_epochs=20)


if __name__ == '__main__' :

    # DEVICE = torch.device("cpu")
    DEVICE = torch.device("mps")
    print(DEVICE)
    random_search(num_epochs=2)



