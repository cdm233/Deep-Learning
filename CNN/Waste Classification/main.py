import time
import os
from unittest import TestLoader
import numpy as np
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

# classes are folders in each directory with these names
useGPU = True
classes = ['Cloth', 'Glass', 'Hazardous Waste',
           'Metal', 'Organic', 'Other', 'Paper', 'Plastic']

# custom dataset class for loading data from array to create pytorch dataloader
class dataSetLoader(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]
    
    def __len__(self):
        return len(self.data)


def getFolderPath(data_dir):
    # data_dir = '/content/gdrive/MyDrive/Colab Notebooks/APS360/APS360 Project Data/'
    train_dir = os.path.join(data_dir, 'Train Dataset/')
    val_dir = os.path.join(data_dir, 'Validation Dataset/')
    test_dir = os.path.join(data_dir, 'Test Dataset/')

    return (train_dir, val_dir, test_dir)

def evaluate(net, loader, criterion):
    softMax = nn.Softmax(dim=1)
    total_err = 0.0
    total_epoch = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data

        if(useGPU):
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = softMax(net(inputs))
        
        for labelIndex, output in enumerate(outputs, 0):
            result = torch.argmax(output)
            if(result != labels[labelIndex]):
                total_err += 1

        total_epoch += len(labels)
        
    err = float(total_err) / total_epoch
    return err

def loadData(root, batchSize=1):
    print("Start loading data...")
    train_dir, val_dir, test_dir = getFolderPath(root)

    startTime = time.time()
    # load, transform, and augment data through resizing and rotation.
    # resize all images to 224 x 224, and rotate images randomly up to 360 degrees

    # resize all images, randomly rotate image, randomly change histogram of image, convert to tensor
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(360),
        transforms.RandomEqualize(),
        transforms.ToTensor()])

    # Set up numpy arrays to store rotated images
    augmented_train_data = []
    augmented_validation_data = []
    augmented_test_data = []

    # Randomly rotate the images four times and get four images with different angles
    for i in range(1):
        print(f"  Rotation {i}...")
        train_data = datasets.ImageFolder(train_dir, transform=data_transform)
        validation_data = datasets.ImageFolder(val_dir, transform=data_transform)
        test_data = datasets.ImageFolder(test_dir, transform=data_transform)

        for j, item in enumerate(train_data):
            augmented_train_data.append(item)
            print(
                f"    {j + 1}/{len(train_data.samples)} training samples loaded...", end='\r')

        print(" ")
        for j, item in enumerate(validation_data):
            augmented_validation_data.append(item)
            print(
                f"    {j + 1}/{len(validation_data.samples)} validation samples loaded...", end='\r')

        print(" ")
        for j, item in enumerate(test_data):
            augmented_test_data.append(item)
            print(
                f"    {j + 1}/{len(test_data.samples)} testing samples loaded...", end='\r')
        print(" ")

    endTime = time.time()

    wallClock = endTime - startTime
    print(f"Data loading and processing took {wallClock:.2f} seconds")

    return (augmented_train_data, augmented_validation_data, augmented_test_data)

def getTransferModel(gradTrack=False):
    print("Loading resNet50...")
    resNet = torchvision.models.resnet50(pretrained=True)
    resNet.eval()

    if(useGPU):
            resNet.cuda()
            print("  Loaded network to GPU")

    if(not gradTrack):
        for param in resNet.parameters():
            # disable gradient tracking
            param.requires_grad = False

    print("restNet50 Loaded")

    return resNet

def train(net, trainSet, valSet, batchSize, learningRate = 0.005, momentum = 0.1, epochSize = 1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learningRate, momentum=momentum)

    torch.manual_seed(100)

    errorTrainArr = []
    errorValArr = []

    print("Start Training Process")
    for epoch in range(epochSize):
        print(" Training on Epoch", epoch, "/", epochSize - 1)
        sampleIndex = 0
        startTime = time.time()
        for imgs, labels in trainSet:
            print("  Training on Batch", sampleIndex, "/", len(trainSet) - 1, end="\r")

            if(useGPU):
                imgs = imgs.cuda()
                labels = labels.cuda()
                
            optimizer.zero_grad()
            output = net(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            sampleIndex += 1

        net.eval()
        print(" ")
        endTime = time.time()
        # evaluate current model performance on validation set
        print("  Evaluating on Validation Set...")
        valEr = evaluate(net, valSet, criterion)
        print("  Evaluating on Train Set...")
        # trainEr = evaluate(net, trainSet, criterion)
        errorValArr.append(valEr)
        # errorTrainArr.append(trainEr)
        print("  Current Validation Error:", valEr)
        # print("  Current Train Error:", trainEr)
        print("  Epoch Time:", endTime - startTime, "Seconds")
        net.train()

        path = "model_lr{0}_mo{1}_epoch{2}_bs{3}_err{4}".format(learningRate,
                                                   momentum,
                                                   epoch,
                                                   batchSize,
                                                   valEr)
        # save current model
        # torch.save(net.state_dict(), path)
    
    print("Training Finished! ")

    plt.plot(range(epoch + 1), errorValArr, 'r', label="Validation")
    # plt.plot(range(epoch + 1), errorTrainArr, 'b', label="Training")
    plt.title(f"Learning Rate {learningRate}, Momentum {momentum}, Epoch {epochSize}")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc="upper right")
    plt.show()
        
def main():
    torch.manual_seed(100)
    # hyperparameter
    batchSize = 48
    learningRate = 0.009
    momentum = 0.15
    epochSize = 30

    global useGPU
    # if useGPU is enabled, check if cuda is supported
    if(useGPU): 
        useGPU = torch.cuda.is_available()

    print(f"{['Not ', ''][useGPU]}Using GPU...")
    resNet = getTransferModel(False)

        # dummy FC layer for testing training code
    resNet.fc = nn.Sequential(
            nn.Linear(2048, 699),
            nn.Dropout(inplace=True, p=0.04403874256000241),
            nn.ReLU(inplace=True),

            nn.Linear(699, 125),
            nn.Dropout(inplace=True, p=0.04403874256000241),
            nn.ReLU(inplace=True),

            nn.Linear(125, 8)
        )

    trainAugmented, valAugmented, testAugmented = loadData("./APS360 Project Data", batchSize)
   
    criterion = nn.CrossEntropyLoss()

    if(useGPU):
        # move FC layer to GPU
        resNet.fc = resNet.fc.cuda()
    
    # trainLoader = DataLoader(dataSetLoader(trainAugmented), batch_size=batchSize, shuffle=True)
    # valLoader = DataLoader(dataSetLoader(valAugmented), batch_size=batchSize, shuffle=True)
    # testLoader = DataLoader(dataSetLoader(testAugmented), batch_size=batchSize, shuffle=True)

    # resNet.load_state_dict(torch.load("./model_err0.11603188662533215_lr0.0014018943853711103_mo0.05717632630403481_epoch8_bs48"))
    # resNet.eval()
    # resNet.cuda()

    # print("Error:", evaluate(resNet, testLoader, criterion))

    # return

    conTrain = False
    while(True):
        # crate data loaders
        trainLoader = DataLoader(dataSetLoader(trainAugmented), batch_size=batchSize, shuffle=True)
        valLoader = DataLoader(dataSetLoader(valAugmented), batch_size=batchSize, shuffle=True)
        
        # train net
        train(resNet, trainLoader, valLoader, batchSize, learningRate, momentum, epochSize)

        conTrain = int(input("Another round of training with different parameters? (yes=1, no=any other key): ")) == 1
        if(conTrain):
            learningRate = float(input(f"Learning Rate(Current:{learningRate}): "))
            momentum = float(input(f"Momentum(Current:{momentum}): "))
            epochSize = int(input(f"Epoch Size(Current:{epochSize}): "))
            batchSize = int(input(f"Batch Size(Current:{batchSize}): "))
        else:
            break
    
    testLoader = DataLoader(dataSetLoader(testAugmented), batch_size=batchSize, shuffle=True)

    print("Evaluating on Test Set...")
    resNet.eval()
    testEr = evaluate(resNet, testLoader, criterion)
    print("Current Model Test Error: ", testEr)

    return


if __name__ == "__main__":
    main()