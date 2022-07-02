import torchvision.models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt  # for plotting
import torch.optim as optim
import time
import random

trainingAmount = 6000
useGPU = False

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

def evaluate(net, testSet):
    error = 0
    softMax = nn.Softmax(dim=1)
    ing = 0
    for image, label in testSet:
        ing += 1
        print(f"    Evaluating on {ing}", end="\r")
        image = image.cuda()
        out = softMax(net(image))
        out = torch.argmax(out)
        if(out != label):
            error += 1

    print("Accuracy:", 1 - error/len(testSet))
    return error/len(testSet)

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


class Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(256*6*6, 1024)
        self.layer2 = nn.Linear(1024, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 9)      

    def forward(self, x):
        x = x.view(-1, 256*6*6)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x

def train(net, trainSet, ValSet, learningRate = 0.005, momentum = 0.1, epochSize = 10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learningRate, momentum=momentum)
    torch.manual_seed(100)
    errorValArr = []

    print("Start Training Process")
    for epoch in range(epochSize):
        print(" Training on Epoch", epoch)
        sampleIndex = 0
        startTime = time.time()
        for img, label in trainSet:
            print("  Training on Sample", sampleIndex, end="\r")
            optimizer.zero_grad()
            ground = oneHotEncode(label, 9)

            # move data to GPU memory
            img = img.cuda()
            ground = ground.cuda()

            output = net(img)
            loss = criterion(output, ground)
            loss.backward(retain_graph=True)
            optimizer.step()
            sampleIndex += 1

        print(" ")
        endTime = time.time()
        valEr = evaluate(net, ValSet)
        errorValArr.append(valEr)
        print("Epoch Time:", endTime - startTime, "Seconds")
        
        path = "model_lr{0}_mo{1}_epoch{2}_ta{3}_err{4}".format(learningRate,
                                                   momentum,
                                                   epoch,
                                                   trainingAmount,
                                                   valEr)
        # save current model
        torch.save(net.state_dict(), path)
        
    plt.plot(range(epoch + 1), errorValArr)
    plt.title(f"Learning Rate {learningRate}, Momentum {momentum}, Epoch {epochSize}")
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.show()

def main():
    trainSet, valSet, testSet = dataLoader("./Data")
    alexnet = torchvision.models.alexnet(pretrained=True)
    
    alexnet.cuda()

    trainSetFeatures = []
    valSetFeatures = []
    testSetFeatures = []

    i = 0
    with torch.no_grad():
        for img, label in trainSet:
            img = img.cuda()
            item = [alexnet.features(img), label]
            trainSetFeatures.append(item)
            del img
            torch.cuda.empty_cache()
            i += 1
            # load only 850 training sample for plotting training curve over multiple epochs
            # if load full 5000 samples, the model over-fits on the third epoch
            if(i == trainingAmount):
                break
    
    i = 0
    with torch.no_grad():
        for img, label in valSet:
            img = img.cuda()
            item = [alexnet.features(img), label]
            valSetFeatures.append(item)
            del img
            torch.cuda.empty_cache()
            i += 1
            # load 700 validation data to speed training up
            if(i == 1000):
                break

    # load all test data
    with torch.no_grad():
        for img, label in testSet:
            img = img.cuda()
            item = [alexnet.features(img), label]
            testSetFeatures.append(item)
            del img
            torch.cuda.empty_cache()

    classifier = Classifier()
    
    # enable GPU
    classifier.cuda()
    train(classifier, trainSetFeatures, valSetFeatures, epochSize=5, learningRate=0.0010, momentum=0.08)
    print("Finished Training")
    print("Evaluating on TestSet")
    evaluate(classifier, testSetFeatures)

if __name__ == "__main__":    
    main()