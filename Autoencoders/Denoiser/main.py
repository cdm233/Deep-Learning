from cProfile import label
from ctypes import util
import time
import os
from turtle import forward
import numpy as np
from sklearn import linear_model
import torch
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random


class autoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 20 * 20),
            nn.ReLU(),

            nn.Linear(20 * 20, 15 * 15),
            nn.ReLU(),

            nn.Linear(15 * 15, 11 * 11),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(11 * 11, 15 * 15),
            nn.ReLU(),

            nn.Linear(15 * 15, 20 * 20),
            nn.ReLU(),

            nn.Linear(20 * 20, 28 * 28),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def loadModel(noiseStrength, validationBatch, path):
    model = autoEncoder()
    model.load_state_dict(torch.load(path))
    model.eval()

    for i in range(4):
        validationBatch[0][i][0] = validationBatch[0][i][0] +  noiseStrength* torch.randn(*validationBatch[0][i][0].shape)
        validationBatch[0][i][0] = np.clip(validationBatch[0][i][0], 0., 1.)

    for i in range(4):
        plt.subplot(2, 4, i+1)
        plt.imshow(validationBatch[0][i][0])

    model.cpu()
    model.eval()

    for j in range(4):
        plt.subplot(2, 4, 4 + j + 1)
        valF = validationBatch[0][j][0].view(-1, 28*28)
        result = model(valF)
        result = result.reshape(28, 28)
        print(result.size())
        plt.imshow(result.detach().numpy())

    plt.savefig("deoiser sample.png")
    plt.show()
    return

def trainModel(model, train, epochs, noiseStrength):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    lossArr = []
    bestLoss = None
    for epoch in range(epochs):
        loss = 0
        sizeee = len(train)
        print(f"Training on Epoch {epoch+1}/{epochs}")
        for i, (batch_features, label) in enumerate(train, 0):
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device

            origin = batch_features
            origin = origin.view(-1, 784).cuda()

            batch_features = batch_features.view(-1, 784).cuda()
            batch_features = batch_features + noiseStrength* torch.randn(*batch_features.shape).cuda()
            batch_features = np.clip(batch_features.cpu(), 0., 1.)
            batch_features = batch_features.cuda()

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, origin)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

            print(f"  Currently on: {i + 1}/{sizeee}", end="\r")
        
        print(" ")

        # compute the epoch training loss
        loss = loss / len(train)
        lossArr.append(loss)
        bestLoss = loss
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
        path = "denoiser_loss{:.6f}".format(loss)
        if(loss < 0.014):
            torch.save(model.state_dict(), path)

    plt.plot(range(epochs), lossArr)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Denoiser Training Error, Loss: {bestLoss:.6f}")
    plt.savefig(f"Denoiser Training Error, Loss_{bestLoss:.6f}.png")

def main():
    noiseStrength = 0.4
    epochs = 30
    batchSize = 48

    mnistData = datasets.MNIST('data', train=True, download=True)
    img_to_tensor = transforms.ToTensor()

    train = DataLoader(datasets.MNIST('data', train=True, download=True,
                       transform=img_to_tensor), batch_size=batchSize, shuffle=True)
    val = DataLoader(datasets.MNIST('data', train=False, download=True,
                     transform=img_to_tensor), batch_size=batchSize, shuffle=True)

    validationBatch = next(iter(val))

    model = autoEncoder()
    model.cuda()


    loadModel(noiseStrength, validationBatch, "./denoiser_loss0.012320")

    return

    trainModel(model, train, epochs, noiseStrength)


    for i in range(4):
        validationBatch[0][i][0] = validationBatch[0][i][0] +  noiseStrength* torch.randn(*validationBatch[0][i][0].shape)
        validationBatch[0][i][0] = np.clip(validationBatch[0][i][0], 0., 1.)

    for i in range(4):
        plt.subplot(2, 4, i+1)
        plt.imshow(validationBatch[0][i][0])

    model.cpu()
    model.eval()

    for j in range(4):
        plt.subplot(2, 4, 4 + j + 1)
        valF = validationBatch[0][j][0].view(-1, 28*28)
        result = model(valF)
        result = result.reshape(28, 28)
        print(result.size())
        plt.imshow(result.detach().numpy())

    plt.show()

    return


if __name__ == "__main__":
    main()
