from matplotlib.cbook import flatten
from matplotlib.transforms import Transform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt  # for plotting
import torch.optim as optim
import numpy as np
import time
import random

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(3, 3)
        self.conv1 = nn.Conv2d(3, 6, 9)
        self.conv2 = nn.Conv2d(6, 18, 7)
        self.conv3 = nn.Conv2d(18, 30, 4)
        self.fc1 = nn.Linear(30 * 6 * 6, 110)
        self.fc2 = nn.Linear(110, 55)
        self.fc3 = nn.Linear(55, 9)

    def forward(self, input):
        output = self.pool(self.conv1(input))
        output = self.pool(self.conv2(output))
        output = self.pool(self.conv3(output))
        output = output.view(-1, 30 * 6 * 6)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)

        return output

def splitAmount(amount, portion):
    # determines the training amount
    trainAmount = round(amount*portion)
    # half of the remaining will be validation data
    valAmount = round((amount - trainAmount)/2)
    return (trainAmount, valAmount)

def oneHotEncode(label, size):
    result = torch.zeros([1,size])
    for i in range(size):
        result[0][i] = i == label

    return result

def dataLoader(path):
    print("Start Loading Data From", path)
    # below defines what portion of the total data belongs to train, val, and test
    # i give validation and test the same amount of data because i want the validation set to represent the test set
    # as much as possible.
    # i also left most of the data to be training data because i want to have as much training data as possible to increase
    # the model's performance. finally i strike a balance at 0.8, 0.1, and 0.1
    # i will split the data by first extracting the training portion, then half of the remaining data will be validation,
    # and all the remaining will be test data, load data this way guarantees no leakage or error
    trainingPortion = 0.7

    train = []
    validation = []
    test = []

    # normalizes image to [-1, 1]
    normalizer = transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transformer = transforms.Compose([transforms.ToTensor(), normalizer])
    dataSet = datasets.ImageFolder(path, transform=transformer)

    print(f"data set length: {len(dataSet.samples)}")

    # stores all items that belongs to the same class
    classItems = []
    index = 0

    # dataSet is sorted by classIdx
    for item in dataSet:
        if(item[1] == index):
            classItems.append(item)
        else:
            # determines the total amount of images in this class
            amount = len(classItems)
            # split data
            trainAmount, valAmount = splitAmount(amount, trainingPortion)

            # append items to corresponding dataset
            train += classItems[:trainAmount]
            validation += classItems[trainAmount:trainAmount + valAmount]
            test += classItems[trainAmount + valAmount:]

            # clear array for next iteration
            classItems.clear()

            # append current item
            classItems.append(item)
            # go to the next class
            index += 1
            print(" Loaded", index, "/", len(dataSet.classes))

    # push last portion to corresponding set
    trainAmount, valAmount = splitAmount(len(classItems), trainingPortion)
    train += classItems[:trainAmount]
    validation += classItems[trainAmount:trainAmount + valAmount]
    test += classItems[trainAmount + valAmount:]
    print(" Loaded", index+1, "/", len(dataSet.classes))
    print(" Finished Loading data")
    random.shuffle(train)
    random.shuffle(validation)
    random.shuffle(test)
    return (train, validation, test)


def train(net, trainSet, valSet, learningRate = 0.005, momentum = 0.1, epochSize = 10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learningRate, momentum=momentum)
    torch.manual_seed(100)

    errorTrainArr = []
    errorValArr = []

    print("Start Training Process")
    for epoch in range(epochSize):
        print(" Training on Epoch", epoch)
        sampleIndex = 0
        startTime = time.time()
        for img, label in trainSet:
            print("  Training on Sample", sampleIndex, end="\r")
            print(img.size())
            optimizer.zero_grad()
            output = net(img)
            ground = oneHotEncode(label, 9)
            loss = criterion(output, ground)
            loss.backward()
            optimizer.step()
            sampleIndex += 1

        # errorTrainArr.append(evaluate(net, trainSet))
        endTime = time.time()
        errorValArr.append(evaluate(net, valSet))
        print("Epoch Time:", endTime - startTime, "Seconds")
        
    plt.plot(range(epoch + 1), errorValArr)
    plt.title(f"Learning Rate {learningRate}, Momentum {momentum}, Epoch {epochSize}")
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.show()

def evaluate(net, testSet):
    error = 0
    softMax = nn.Softmax(dim=1)
    for image, label in testSet:
        out = softMax(net(image))
        out = torch.argmax(out)
        if(out != label):
            error += 1

    print("Accuracy:", 1 - error/len(testSet))
    return error/len(testSet)

def main():
    trainSet, valSet, testSet = dataLoader("./Data")
    convNet = CNN()
    # device = torch.device("cuda:0")
    # convNet.to(device)
    train(convNet, trainSet, valSet, epochSize=13, learningRate=0.0012, momentum=0.08)
    evaluate(convNet, testSet)

if __name__ == "__main__":
    main()
