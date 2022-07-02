import joblib
import scipy.io
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torchvision
from torchvision import datasets, models, transforms
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
import seaborn as sns; 
sns.set()
from skimage import data
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Helper Functions
def getFolderPath(data_dir):
    train_dir = os.path.join(data_dir, 'Train Dataset/')
    val_dir = os.path.join(data_dir, 'Validation Dataset/')
    test_dir = os.path.join(data_dir, 'Test Dataset/')

    return (train_dir, val_dir, test_dir)


def flattenData(dataloader):
    img_arr, label_arr = [], []

    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        flattened = inputs.view(-1, 3*224*224)
        flattened = flattened.squeeze(0).detach().numpy()
        labels = labels.squeeze(0).detach().numpy()
        img_arr.append(flattened)
        label_arr.append(labels) 
    
    img_arr = np.asarray(img_arr)
    label_arr = np.asarray(label_arr)

    return (img_arr, label_arr)   


def loadAndProcessData(root):
    print("Start loading data...")
    
    train_dir, val_dir, test_dir = getFolderPath(root)

    startTime = time.time()

    # resize all images, randomly rotate image, randomly change histogram of image, convert to tensor
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    # Set up numpy arrays to store rotated images
    augmented_train_data = []
    augmented_val_data = []
    augmented_test_data = []

    # Randomly rotate the images four times and get four images with different angles
    for i in range(1):
        print(f"  Rotation {i}...")
        train_data = datasets.ImageFolder(train_dir, transform=data_transform)
        val_data = datasets.ImageFolder(val_dir, transform=data_transform)
        test_data = datasets.ImageFolder(test_dir, transform=data_transform)

        for j, item in enumerate(train_data):
            augmented_train_data.append(item)
            print(
                f"    {j + 1}/{len(train_data.samples)} training samples loaded...", end='\r')

        print(" ")
        for j, item in enumerate(val_data):
            augmented_val_data.append(item)
            print(
                f"    {j + 1}/{len(val_data.samples)} validation samples loaded...", end='\r')

        print(" ")
        for j, item in enumerate(test_data):
            augmented_test_data.append(item)
            print(
                f"    {j + 1}/{len(test_data.samples)} testing samples loaded...", end='\r')
        print(" ")

    # crate data loaders
    trainLoader = DataLoader(augmented_train_data, batch_size=1, shuffle=True)
    valLoader = DataLoader(augmented_val_data, batch_size=1, shuffle=True)
    testLoader = DataLoader(augmented_test_data, batch_size=1, shuffle=True)

    print("Start to Flatten Data")
    train_imgs, train_labels = flattenData(trainLoader)
    val_imgs, val_labels = flattenData(valLoader)
    test_imgs, test_labels = flattenData(testLoader)

    endTime = time.time()

    wallClock = endTime - startTime
    print(f"Data loading and processing took {wallClock:.2f} seconds")
    print("The size of train data: ", len(train_imgs))
    print("The size of validation data: ", len(val_imgs))
    print("The size of test data: ", len(test_imgs))

    return (train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels)


def evaluate(model, imgs, labels):
    
    predict = model.predict(imgs)
    print("The test accuracy is", metrics.accuracy_score(labels, predict))

    # for good measure, plot the confusion matrix
    matrix = confusion_matrix(labels, predict, normalize='true')
    sns.heatmap(matrix.T, square=True, annot=True, cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.savefig('confusion matrix: random forest')
    plt.show()

    return


def plot_training_curve(accuracy_arr):

    plt.title("Validation Accuracy")
    n = len(accuracy_arr) # number of epochs
    plt.plot(range(1, n+1), accuracy_arr, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()


def trainModel(model, train_imgs, train_labels, val_imgs, val_labels, num_epochs, n_estimators, max_depth, min_samples_leaf):
    print("Start training...")

    val_acc = np.zeros(num_epochs)

    startTime = time.time()

    for epoch in range(num_epochs):
        model.fit(train_imgs, train_labels)
        print("  Evaluating on Validation Data...")
        predict = model.predict(val_imgs)
        accuracy = metrics.accuracy_score(val_labels, predict)
        val_acc[epoch] = accuracy
        print("  Epoch {}: Validation Accuracy: {}".format(epoch + 1, accuracy))
        pathName = "model_acc{4}_ne{0}_md{1}_msl{2}_epoch{3}.joblib".format(n_estimators, max_depth, min_samples_leaf, epoch, accuracy)
        joblib.dump(model, pathName)
    
    endTime = time.time()
    print("Finish training. The training time is", endTime - startTime)  

    plot_training_curve(val_acc)
    return

def main():
    # Initialize hyperparameters: 
    n_estimators = 80  #The number of trees in the forest.
    criterion = "gini"  #The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain
    max_depth = 35  #The maximum depth of the tree.
    min_samples_leaf = 3    #The minimum number of samples required to be at a leaf node.
    epoch = 5

    classes = ['Cloth', 'Glass', 'Hazardous Waste', 'Metal', 'Organic', 'Other', 'Paper', 'Plastic']

    root = './APS360 Project Data'
    train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels = loadAndProcessData(root)

    while(True):
        clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        trainModel(clf, train_imgs, train_labels, val_imgs, val_labels, epoch, n_estimators, max_depth, min_samples_leaf)
        evaluate(clf, test_imgs, test_labels)

        conTrain = int(input("Another round of training with different parameters? (yes=1, no=any other key): ")) == 1
        if(conTrain):
            n_estimators = int(input(f"n_estimators(Current:{n_estimators}): "))
            max_depth = int(input(f"max_depth(Current:{max_depth}): "))
            min_samples_leaf = int(input(f"min_samples_leaf(Current:{min_samples_leaf}): "))
            epoch = int(input(f"epoch(Current:{epoch}): "))
        else:
            break



if __name__ == "__main__":
    main()
