import os
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets.mnist import MNIST


from vit.model import ViT

device = 'cuda'

def main():
    #Prepare Dataset
    transform = transforms.ToTensor()
    train_set = MNIST(root='data/', train=True, download=True, transform=transform)
    val_set = MNIST(root='data/', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=32)
    val_loader = DataLoader(train_set, shuffle=False, batch_size=32)

    #Define Model
    model = ViT(img_size=28, patch_size=2, in_channels=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 5
    #Train
    for epoch in range(epochs):
        epoch_loss = 0.
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss = loss_fn(pred, label)

            epoch_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch} loss: {epoch_loss:.2f}')

        #Test
        with torch.no_grad():
            correct, total = 0, 0
            val_loss = 0.
            for img, label in val_loader:
                img, label = img.to(device), label.to(device)
                pred = model(img)
                loss = loss_fn(pred, label)

                val_loss += loss.detach().cpu().item() / len(val_loader)

                correct += torch.sum(torch.argmax(pred, dim=1) == label).detach().cpu().item()
                total += len(label)
            
            print(f'Validation Loss: {val_loss:2f}')
            print(f'Validation Accuracy: {correct / total * 100:2f}%')
            
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, 'checkpoint/mnist_5.pt')

if __name__ == '__main__':
    main()