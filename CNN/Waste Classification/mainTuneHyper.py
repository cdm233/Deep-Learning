import time
import os
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

def getNewLR(origin, epoch:int):
    return origin*torch.e ** (-epoch/20)

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
        validation_data = datasets.ImageFolder(
            val_dir, transform=data_transform)
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


def train(net, trainSet, valSet, batchSize, learningRate=0.005, momentum=0.1, epochSize=1, dropOut=-1, optimizerO=0):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learningRate, momentum=momentum)
    originLR = learningRate

    errorTrainArr = []
    errorValArr = []

    print("Start Training Process")
    print("Training Parameter: ")
    print(f" Learning Rate: {learningRate}")
    print(f" Momentum: {momentum}")
    print(f" EpochSize: {epochSize}")
    print(f" batchSize: {batchSize}")

    for epoch in range(epochSize):
        # if(epoch == 15):
        #     learningRate = learningRate / 2
        # elif(epoch == 25):
        #     learningRate = learningRate / 1.5
        # elif(epoch == 40):
        #     learningRate = learningRate / 1.5
        learningRate = getNewLR(originLR, epoch)
        
        print(f" Current Learning Rate: {learningRate}")
        if(optimizerO == 0):
            optimizer = optim.Adam(net.parameters(), lr=learningRate, weight_decay=0.0001)
        else:
            optimizer = optim.SGD(net.parameters(), lr=learningRate, momentum=momentum)
            

        print(" Training on Epoch", epoch, "/", epochSize - 1)
        sampleIndex = 0
        startTime = time.time()
        for imgs, labels in trainSet:
            print("  Training on Batch", sampleIndex,
                  "/", len(trainSet) - 1, end="\r")

            if(useGPU):
                imgs = imgs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            output = net(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            sampleIndex += 1

        print(" ")
        endTime = time.time()
        # evaluate current model performance on validation set
        print("  Evaluating on Validation Set...")

        net.eval()
        valEr = evaluate(net, valSet, criterion)
        net.train()
        # print("  Evaluating on Train Set...")
        # trainEr = evaluate(net, trainSet, criterion)
        errorValArr.append(valEr)
        # errorTrainArr.append(trainEr)
        print("  Current Validation Error:", valEr)
        # print("  Current Train Error:", trainEr)
        print("  Epoch Time:", endTime - startTime, "Seconds")

        path = "model_err{4}_lr{0}_mo{1}_epoch{2}_bs{3}_do{5}_op{6}".format(learningRate,
                                                                momentum,
                                                                epoch,
                                                                batchSize,
                                                                valEr,
                                                                dropOut,
                                                                optimizerO)

        # save current model
        if(valEr < 0.11):
            torch.save(net.state_dict(), path)

    print("Training Finished! ")

    bestValErr = min(errorValArr)

    plt.plot(range(epoch + 1), errorValArr, 'r', label="Validation")
    # plt.plot(range(epoch + 1), errorTrainArr, 'b', label="Training")
    plt.title(f"LR {learningRate:.5f}, Momentum {momentum:.5f}, Epoch {epochSize}, BS {batchSize}")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc="upper right")
    plt.savefig(f"valErr{bestValErr:.5f}_LR{learningRate}_Mo{momentum}_Ep{epochSize}_BS{batchSize}_OP{optimizerO}_DO{dropOut}.png")
    plt.clf()


def main():
    torch.manual_seed(100)
    # hyperparameter
    batchSize = 12
    learningRate = 0.00035   
    momentum = 0.06
    epochSize = 70
    dropOutP = 0.08
    optimizerO = 0

    global useGPU
    # if useGPU is enabled, check if cuda is supported
    if(useGPU):
        useGPU = torch.cuda.is_available()

    print(f"{['Not ', ''][useGPU]}Using GPU...")
    trainAugmented, valAugmented, testAugmented = loadData(
        "./APS360 Project Data", batchSize)
    resNet = getTransferModel(False)

    if(useGPU):
        # move FC layer to GPU
        resNet.fc = resNet.fc.cuda()

    criterion = nn.CrossEntropyLoss()

    learningRateSeq = [0.0005, 0.01]
    momentumSeq = [0.01, 0.15]
    batchSizeSeq = [1, 32]
    dropOutSeq = [0.01, 0.3]

    highNode = 710
    lowNode = 130

    seqIndex = 0
    seqMax = 60

    while(seqIndex < seqMax):
        # crate data loaders
        resNet.fc = nn.Sequential(
            nn.Linear(2048, highNode),
            nn.Dropout(inplace=True, p=dropOutP),
            nn.ReLU(inplace=True),

            nn.Linear(highNode, lowNode),
            nn.Dropout(inplace=True, p=dropOutP),
            nn.ReLU(inplace=True),

            nn.Linear(lowNode, 8)
        )
        
        trainLoader = DataLoader(dataSetLoader(
            trainAugmented), batch_size=batchSize, shuffle=True)
        valLoader = DataLoader(dataSetLoader(
            valAugmented), batch_size=batchSize, shuffle=True)
        
        if(useGPU):
            # move FC layer to GPU
            resNet.fc = resNet.fc.cuda()
        
        # train net
        train(resNet, trainLoader, valLoader, batchSize,
              learningRate, momentum, epochSize, dropOutP, optimizerO)

        # batchSize = random.randint(batchSizeSeq[0], batchSizeSeq[1])
        # learningRate = random.uniform(learningRateSeq[0], learningRateSeq[1])
        # momentum = random.uniform(momentumSeq[0], momentumSeq[1])
        # dropOutP = random.uniform(dropOutSeq[0], dropOutSeq[1])
        # optimizerO = random.randint(0, 1)

        seqIndex += 1
       

        conTrain = int(input("Another round of training with different parameters? (yes=1, no=any other key): ")) == 1
        if(conTrain):
            learningRate = float(input(f"Learning Rate(Current:{learningRate}): "))
            momentum = float(input(f"Momentum(Current:{momentum}): "))
            epochSize = int(input(f"Epoch Size(Current:{epochSize}): "))
            batchSize = int(input(f"Batch Size(Current:{batchSize}): "))
            highNode = int(input(f"highNode(Current:{highNode}): "))
            lowNode = int(input(f"lowNode(Current:{lowNode}): "))
        else:
            break



    testLoader = DataLoader(dataSetLoader(testAugmented),
                            batch_size=batchSize, shuffle=True)

    print("Evaluating on Test Set...")
    testEr = evaluate(resNet, testLoader, criterion)
    print("Current Model Test Error: ", testEr)

    return


if __name__ == "__main__":
    main()
