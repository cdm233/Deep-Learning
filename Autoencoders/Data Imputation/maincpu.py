import csv
import enum
import numpy as np
import random
import torch
import torch.utils.data
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
import time
from collections import Counter

batchSize = 64

header = ['age', 'work', 'fnlwgt', 'edu', 'yredu', 'marriage', 'occupation',
            'relationship', 'race', 'sex', 'capgain', 'caploss', 'workhr', 'country']
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    names=header,
    index_col=False)

df["age"] = df["age"]/90
df["yredu"] /= 16
df["capgain"] /= 99999
df["caploss"] /= 4356
df["workhr"] /= 99

contcols = ["age", "yredu", "capgain", "caploss", "workhr"]
catcols = ["work", "marriage", "occupation", "edu", "relationship", "sex"]
features = contcols + catcols
df = df[features]

baselineDic = {}

for feature in features:
    baselineDic[feature] = ""
    tempArr = []
    for data in df[feature]:
        tempArr.append(data)

    c = Counter(tempArr)
    temp = c.most_common(1)[0][0]
    baselineDic[feature] = temp


missing = pd.concat([df[c] == " ?" for c in catcols], axis=1).any(axis=1)
df_with_missing = df[missing]
df_not_missing = df[~missing]

data = pd.get_dummies(df_not_missing)
datanp = data.values.astype(np.float32)

cat_index = {}  # Mapping of feature -> start index of feature in a record
cat_values = {} # Mapping of feature -> list of categorical values the feature can take

# build up the cat_index and cat_values dictionary
for i, header in enumerate(data.keys()):
    if "_" in header: # categorical header
        feature, value = header.split()
        feature = feature[:-1] # remove the last char; it is always an underscore
        if feature not in cat_index:
            cat_index[feature] = i
            cat_values[feature] = [value]
        else:
            cat_values[feature].append(value)

def minMaxAvg(df, name):
    return (np.min(df[name]), np.max(df[name]), np.average(df[name]))

def get_onehot(record, feature):
    """
    Return the portion of `record` that is the one-hot encoding
    of `feature`. For example, since the feature "work" is stored
    in the indices [5:12] in each record, calling `get_range(record, "work")`
    is equivalent to accessing `record[5:12]`.
    
    Args:
        - record: a numpy array representing one record, formatted
                  the same way as a row in `data.np`
        - feature: a string, should be an element of `catcols`
    """
    start_index = cat_index[feature]
    stop_index = cat_index[feature] + len(cat_values[feature])
    return record[start_index:stop_index]

def get_categorical_value(onehot, feature):
    """
    Return the categorical value name of a feature given
    a one-hot vector representing the feature.
    
    Args:
        - onehot: a numpy array one-hot representation of the feature
        - feature: a string, should be an element of `catcols`
        
    Examples:
    
    >>> get_categorical_value(np.array([0., 0., 0., 0., 0., 1., 0.]), "work")
    'State-gov'
    >>> get_categorical_value(np.array([0.1, 0., 1.1, 0.2, 0., 1., 0.]), "work")
    'Private'
    """
    # <----- TODO: WRITE YOUR CODE HERE ----->
    # You may find the variables `cat_index` and `cat_values` 
    # (created above) useful.
    return cat_values[feature][np.argmax(onehot)]

# more useful code, used during training, that depends on the function
# you write above

def get_feature(record, feature):
    """
    Return the categorical feature value of a record
    """
    onehot = get_onehot(record, feature)
    return get_categorical_value(onehot, feature)

def get_features(record):
    """
    Return a dictionary of all categorical feature values of a record
    """
    return { f: get_feature(record, f) for f in catcols }

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(57, 45), # TODO -- FILL OUT THE CODE HERE!
            nn.ReLU(),
            
            nn.Linear(45, 30),
            nn.ReLU(),
            
            nn.Linear(30, 15),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(15, 30),
            nn.Tanh(),
            
            nn.Linear(30, 45),
            nn.Tanh(),

            nn.Linear(45, 57), # TODO -- FILL OUT THE CODE HERE!
            nn.Sigmoid() # get to the range (0, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def zero_out_feature(records, feature):
    """ Set the feature missing in records, by setting the appropriate
    columns of records to 0
    """
    start_index = cat_index[feature]
    stop_index = cat_index[feature] + len(cat_values[feature])
    records[:, start_index:stop_index] = 0
    return records

def zero_out_random_feature(records):
    """ Set one random feature missing in records, by setting the 
    appropriate columns of records to 0
    """
    return zero_out_feature(records, random.choice(catcols))

def get_accuracy(model, data_loader):
    """Return the "accuracy" of the autoencoder model across a data set.
    That is, for each record and for each categorical feature, 
    we determine whether the model can successfully predict the value
    of the categorical feature given all the other features of the 
    record. The returned "accuracy" measure is the percentage of times 
    that our model is successful.
        
    Args:
       - model: the autoencoder model, an instance of nn.Module
       - data_loader: an instance of torch.utils.data.DataLoader

    Example (to illustrate how get_accuracy is intended to be called.
             Depending on your variable naming this code might require
             modification.)

        >>> model = AutoEncoder()
        >>> vdl = torch.utils.data.DataLoader(data_valid, batch_size=256, shuffle=True)
        >>> get_accuracy(model, vdl)
    """
    total = 0
    acc = 0
    for col in catcols:
        for item in data_loader: # minibatches
            inp = item.detach().numpy()
            out = model(zero_out_feature(item.clone(), col)).detach().numpy()
            for i in range(out.shape[0]): # record in minibatch
                acc += int(get_feature(out[i], col) == get_feature(inp[i], col))
                total += 1

    return acc / total

def evaluate(model, loader):
    criterion = nn.MSELoss()
    totalLoss = 0
    epochSize = 0

    for data in loader:
        dataC = data
        datam = zero_out_random_feature(dataC)

        recon = model(datam)
        loss = criterion(recon, data)
        totalLoss += loss.item()
        epochSize += 1

    return totalLoss / epochSize

def loadToBatch(data):
    trainLoader = []
    temp = []
    counter = 0
    for data in trainingData:
        temp.append(data)
        if(counter == batchSize - 1):
            trainLoader.append(temp)
            temp = []
            counter = 0
            continue
        counter += 1

    return torch.from_numpy(np.array(trainLoader))

def train(model, train_loader, valid_loader, num_epochs=5, learning_rate=1e-4):
    print("Start Training...")
    torch.manual_seed(42)

    model.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainLossArr = []
    trainAccArr = []
    
    valLossArr = []
    valAccArr = []


    counter = 0
    segment = 1000
    numDataPoints = 0

    for epoch in range(num_epochs):
        print(f"Training on {epoch}/{num_epochs-1}")
        for i, data in enumerate(train_loader, 0):
            print(f"  {i}/{len(trainLoader)} done...", end="\r")
            datam = zero_out_random_feature(data.clone()) # zero out one categorical feature
            recon = model(datam)
            loss = criterion(recon, data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if(i == len(train_loader) - 1):
                # ground = get_feature(data[0].detach().numpy(), 'work')
                # pred = get_feature(recon[0].detach().numpy(), 'work')
                # print(ground)
                # print(pred)
                
                # print(ground == pred)

                # print(data[2])
                # print(recon[2])
                valAcc = get_accuracy(model, valid_loader)
                print(valAcc)

            # if(counter == segment):
            #     start = time.time()
            #     counter = 0
            #     numDataPoints += 1
                
            #     trainAcc = get_accuracy(model, train_loader)
            #     valAcc = get_accuracy(model, valid_loader)

            #     trainLoss = evaluate(model, train_loader)
            #     valLoss = evaluate(model, valid_loader)

            #     print(trainAcc)
            #     print(valAcc)
                
            #     trainAccArr.append(trainAcc)
            #     trainLossArr.append(trainLoss)

            #     valAccArr.append(valAcc)
            #     valLossArr.append(valLoss)
            #     end = time.time()

            #     print("  Evaluation Time:", end - start)

            # counter += 1
        
        print("")
        # path = "model_acc{1}_epoch{0}".format(epoch, valAccArr[len(valLossArr) - 1])
        # torch.save(model.state_dict(), path)
            

    # numDataPoints += 1
    
    # trainAccArr.append(get_accuracy(model, train_loader))
    # trainLossArr.append(evaluate(model, train_loader))

    # valAccArr.append(get_accuracy(model, valid_loader))
    # valLossArr.append(evaluate(model, valid_loader))

    # plt.plot(range(1, numDataPoints + 1), trainLossArr, 'g', label="Training")
    # plt.plot(range(1, numDataPoints + 1), valLossArr, 'r', label="Validation")
    # plt.title(f"Model Performance evaluated on Loss")
    # plt.xlabel("Cycle")
    # plt.ylabel("Loss")
    # plt.legend(loc="upper right")
    # plt.savefig(f"ModelPerformanceLoss.png")
    # plt.clf()

    # plt.plot(range(1, numDataPoints + 1), trainAccArr, 'g', label="Training")
    # plt.plot(range(1, numDataPoints + 1), valAccArr, 'r', label="Validation")
    # plt.title(f"Model Performance evaluated on Accuracy")
    # plt.xlabel("Cycle")
    # plt.ylabel("Accuracy")
    # plt.legend(loc="upper right")
    # plt.savefig(f"ModelPerformanceAccuracy.png")
    # plt.clf()

      

# set the numpy seed for reproducibility
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.seed.html
np.random.seed(50)

totalData = len(data)

trainingNum = round(totalData * 0.7)
validationNum = round(totalData * 0.15)
testingNum = round(totalData * 0.15)

dataShuffled = datanp
np.random.shuffle(dataShuffled)

# todo
trainingData = dataShuffled[:4000]
validationData = dataShuffled[trainingNum:validationNum + trainingNum]
testingData = dataShuffled[validationNum + trainingNum:]

trainLoader = loadToBatch(trainingData)
valLoader = loadToBatch(validationData)
testLoader = loadToBatch(testingData)

print(trainLoader.shape)

model = AutoEncoder()

# total = 0
# acc = 0
# for col in catcols:
#     for item in validationData: # minibatches
#         inp = item
#         itemC = item
#         out = baselineDic[col].replace(" ", "")
#         acc += int(out == get_feature(inp, col))
#         total += 1
#         # break
# result = acc / total

# print(result)
        
# model.load_state_dict(torch.load("./model_err0.000246150820493507_epoch3"))
# model.eval()

# total = 0
# acc = 0
# for col in catcols:
#     print(col)
#     for item in validationData: # minibatches
#         inp = item
#         itemC = item
#         ground = get_feature(inp, col)
#         predict = model(item)

#         print(predict, ground)

#         acc += int(predict == predict)
#         total += 1
#         break
# result = acc / total

# print(result)

train(model, trainLoader, valLoader, 30, 1e-4)