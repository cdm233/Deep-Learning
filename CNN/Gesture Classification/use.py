import torchvision.models
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


def evaluate(net, testSet):
    dic = {
        0:"A",
        1:"B",
        2:"C",
        3:"D",
        4:"E",
        5:"F",
        6:"G",
        7:"H",
        8:"I"
    }
    error = 0
    softMax = nn.Softmax(dim=1)
    ing = 0
    for image, label in testSet:
        ing += 1
        # print(f"    Evaluating on {ing}", end="\r")
        image = image.cuda()
        out = softMax(net(image))
        out = torch.argmax(out)
        print("Predicted:", dic[out.item()],"Actual:", dic[label])
        if(out != label):
            error += 1

    print("Accuracy:", 1 - error/len(testSet))
    return error/len(testSet)

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

def main():
    model = Classifier()
    # model.load_state_dict(torch.load("./model_lr0.0011_mo0.08_epoch16_err0.08571428571428572"))
    model.load_state_dict(torch.load("./model_lr0.0011_mo0.08_epoch14_ta2000_err0.0557142857142857"))
    # model.load_state_dict(torch.load("./model_lr0.0011_mo0.08_epoch2_ta5000_err0.054"))
    model.eval()
    model.cuda()

    alexnet = torchvision.models.alexnet(pretrained=True)

    normalizer = transforms.Normalize(
    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transformer = transforms.Compose([transforms.ToTensor(), normalizer])
    dataSet = datasets.ImageFolder("./testData/", transform=transformer)

    checkSet = []

    for item in dataSet:
        feature = alexnet.features(item[0])
        checkSet.append([feature, item[1]])

    evaluate(model, checkSet)

    return

if __name__ == "__main__":
    main()