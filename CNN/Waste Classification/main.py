from cProfile import label
import time
import os
from unittest import result
from urllib.request import ProxyBasicAuthHandler
import numpy as np
from sklearn.metrics import confusion_matrix
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
import seaborn as sns

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
    totalLoss = 0.
    for i, data in enumerate(loader, 0):
        inputs, labels = data

        if(useGPU):
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = softMax(net(inputs))
        loss = criterion(outputs, labels)

        for labelIndex, output in enumerate(outputs, 0):
            result = torch.argmax(output)
            if(result != labels[labelIndex]):
                total_err += 1

        total_epoch += len(labels)
        totalLoss += loss.item()

    err = float(total_err) / total_epoch
    loss = float(totalLoss / (i + 1))
    return err, loss


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

    dataTransformNoAug = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Set up numpy arrays to store rotated images
    augmented_train_data = []
    augmented_validation_data = []
    augmented_test_data = []

    # Randomly rotate the images four times and get four images with different angles
    for i in range(1):
        train_data = None
        validation_data = None
        test_data = None
        print(f"  Rotation {i}...")
        if(i == 0):
            train_data = datasets.ImageFolder(train_dir, transform=dataTransformNoAug)
            validation_data = datasets.ImageFolder(val_dir, transform=dataTransformNoAug)
            test_data = datasets.ImageFolder(test_dir, transform=dataTransformNoAug)

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
        else:
            train_data = datasets.ImageFolder(train_dir, transform=data_transform)

        for j, item in enumerate(train_data):
                augmented_train_data.append(item)
                print(
                    f"    {j + 1}/{len(train_data.samples)} training samples loaded...", end='\r')

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


def train(net, trainSet, valSet, batchSize, learningRate=0.005, epochSize=1, dropOut=-1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learningRate, weight_decay=0.0001)    

    errorTrainArr = []
    errorValArr = []
    lossValArr = []
    lossTrainArr = []

    print("Start Training Process")
    print("Training Parameter: ")
    print(f" Learning Rate: {learningRate}")
    print(f" EpochSize: {epochSize}")
    print(f" batchSize: {batchSize}")
    
    net.train()

    for epoch in range(epochSize):
        # learningRate = getNewLR(originLR, epoch)
        
        print(f" Current Learning Rate: {learningRate}")

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
        valEr, valLoss = evaluate(net, valSet, criterion)
        trainEr, trainLoss = evaluate(net, trainSet, criterion)
        net.train()

        print("  Evaluating on Train Set...")
        trainEr = evaluate(net, trainSet, criterion)
        lossValArr.append(valLoss)
        errorValArr.append(valEr)
        lossTrainArr.append(trainLoss)
        errorTrainArr.append(trainEr)
        errorTrainArr.append(trainEr)

        print("  Current Train Error:", trainEr)
        print("  Current Train Error:", trainEr)
        print("  Epoch Time:", endTime - startTime, "Seconds")

        path = "model_err{3}_lr{0}_epoch{1}_bs{2}_do{4}".format(learningRate,
                                                                epoch,
                                                                batchSize,
                                                                trainEr,
                                                                dropOut)

        # save current model
        if(valEr < 0.045):
            torch.save(net.state_dict(), path)

    print("Training Finished! ")

    plt.plot(range(1, epoch + 2), lossTrainArr, 'g', label="Training")
    plt.plot(range(1, epoch + 2), lossValArr, 'r', label="Validation")
    plt.title(f"Model Performance evaluated on Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig(f"ModelPerformanceLoss.png")
    plt.clf()

    plt.plot(range(1, epoch + 2), errorTrainArr, 'g', label="Training")
    plt.plot(range(1, epoch + 2), errorValArr, 'r', label="Validation")
    plt.title(f"Model Performance evaluated on Error")
    plt.title(f"Model Sanity Check")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc="upper right")
    plt.savefig(f"ModelPerformanceError.png")
    plt.clf()

def generateConfusionMatrix(net, valLoader):
    print("Generating Confusion Matrix")
    net.eval()
    labelsArr = []
    predictArr = []
    for i, (img, label) in enumerate(valLoader, 0):
        img = img.cuda()
        output = net(img)
        output = torch.argmax(output)
        output = output.cpu()
        output = output.item()
        label = label.item()
        labelsArr.append(label)

        predictArr.append(output)

    labelsArr = np.asarray(labelsArr)
    predictArr = np.asarray(predictArr)

    matrix = confusion_matrix(labelsArr, predictArr, normalize="true")
    sns.heatmap(matrix.T, square=True, annot=True, cbar=False)
    plt.title("Model Confusion Matrix")
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.xticks(ticks=[0,1,2,3,4,5,6,7], labels=classes, rotation=20)
    plt.yticks(ticks=[0,1,2,3,4,5,6,7], labels=classes, rotation=20)
    plt.savefig('Confusion Matrix: Primary Model.png')
    plt.show()

    return

def calculateParameters(resNet):
    totalP = 0
    classP = 0
    
    for param in resNet.parameters():
        temp = param.size()
        count = 1
        for i in temp:
            count *= i
        totalP += count

    for param in resNet.fc.parameters():
        temp = param.size()
        count = 1
        for i in temp:
            count *= i
        classP += count
    
    return totalP, classP

def randomSearchHyper(resNet, trainLoader, valLoader):
    batchSize = 1
    learningRate = 0.000125
    epochSize = 30
    dropOutP = 0.05

    learningRateSeq = [0.0005, 0.01]
    batchSizeSeq = [1, 128]
    dropOutSeq = [0.01, 0.3]

    seqIndex = 0
    seqMax = 60

    while(seqIndex < seqMax):
        # reinitialize model
        resNet.fc = nn.Sequential(
            nn.Linear(2048, 700),
            nn.Dropout(inplace=True, p=dropOutP),
            nn.ReLU(inplace=True),

            nn.Linear(700, 110),
            nn.Dropout(inplace=True, p=dropOutP),
            nn.ReLU(inplace=True),

            nn.Linear(110, 8)
        )

        if(useGPU):
            # move FC layer to GPU
            resNet.fc = resNet.fc.cuda()

        batchSize = random.randint(batchSizeSeq[0], batchSizeSeq[1])
        learningRate = random.uniform(learningRateSeq[0], learningRateSeq[1])
        dropOutP = random.uniform(dropOutSeq[0], dropOutSeq[1])

        # train net
        train(resNet, trainLoader, valLoader, batchSize,
              learningRate, epochSize, dropOutP)

        seqIndex += 1

def main():
    torch.manual_seed(100)

    # hyperparameter
    batchSize = 1
    learningRate = 0.000125
    epochSize = 20
    dropOutP = 0.05

    global useGPU

    # if useGPU is enabled, check if cuda is supported
    if(useGPU):
        useGPU = torch.cuda.is_available()

    print(f"{['Not ', ''][useGPU]}Using GPU...")
    trainAugmented, valAugmented, testAugmented = loadData(
        "./APS360 Project Data", batchSize)
    resNet = getTransferModel(False)

    resNet.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Dropout(inplace=True, p=dropOutP),
            nn.ReLU(inplace=True),

            nn.Linear(512, 512),
            nn.Dropout(inplace=True, p=dropOutP),
            nn.ReLU(inplace=True),

            nn.Linear(512, 8)
        )

    if(useGPU):
        # move FC layer to GPU
        resNet.fc = resNet.fc.cuda()   

    resNet.load_state_dict(torch.load("model_err0.03631532329495128_lr0.000125_mo0.06_epoch29_bs64_do0.05_op0"))

    totalP, classP = calculateParameters(resNet)
            
    print(totalP, classP, totalP-classP)

    criterion = nn.CrossEntropyLoss()

    trainLoader = DataLoader(dataSetLoader(
        trainAugmented), batch_size=batchSize, shuffle=True)
    valLoader = DataLoader(dataSetLoader(
        valAugmented), batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(dataSetLoader(testAugmented),
                            batch_size=batchSize, shuffle=True)
    

    # train net
    # train(resNet, trainLoader, valLoader, batchSize,
    #         learningRate, epochSize, dropOutP)

    resNet.load_state_dict(torch.load("./model_err0.03985828166519043_lr0.000125_mo0.06_epoch22_bs64_do0.05_op0"))
    
    generateConfusionMatrix(resNet, valLoader)

    print("Evaluating on Test Set...")
    testEr = evaluate(resNet, testLoader, criterion)
    print("Current Model Test Error: ", testEr)

    return


if __name__ == "__main__":
    main()
