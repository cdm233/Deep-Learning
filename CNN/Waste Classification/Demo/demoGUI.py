from tkinter import filedialog
from sklearn.metrics import confusion_matrix
import torch
from torchvision import transforms
import torch.nn as nn
import torchvision.transforms as transforms
import PIL.Image
import PIL.ImageTk
from tkinter import *

class demoApp():
    def __init__(self, parent) -> None:
        # load app parameters
        print("Application initializing...")
        self.root = parent
        self.pixelSize = 3
        self.canvasSize = 224*self.pixelSize
        self.marginSize = 20
        self.positionAdjust = 2
        self.classes = ['Cloth', 'Glass', 'Hazardous Waste', 'Metal', 'Organic', 'Other', 'Paper', 'Plastic']
        self.predicted = False
        
        # define window geometry
        self.root.geometry("300x200+10+10")
        self.root.title("APS360 Project Demo")
        self.root.minsize(width=224*(self.pixelSize + 2) + 30,
                    height=self.canvasSize + 2 * self.marginSize)
        self.root.resizable(width=False, height=False)

        # create main canvas
        self.canvas = Canvas(self.root, bg="pink", height=self.canvasSize, width=self.canvasSize)
        self.canvas.place(x=self.marginSize, y=self.marginSize - self.positionAdjust)

        # create random image
        # for rowIndex in range(224):
        #     for colIndex in range(224):
        #         pixelColor = "#%02x%02x%02x" % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        #         self.canvas.create_rectangle(self.pixelSize*rowIndex, self.pixelSize*colIndex, self.pixelSize*(rowIndex + 1),
        #                         self.pixelSize*(colIndex + 1), fill=pixelColor, outline=pixelColor)

        # create buttons
        predictButton = Button(parent, text="Select File", fg='black',
                            command=self.selectFile)  # , height=5, width=5
        predictButton.place(x=self.canvasSize + 50, y=self.canvasSize /
                            2 + self.marginSize - self.positionAdjust - 40)

        selectFileButton = Button(parent, text="Predict", fg='black',
                            command=self.modelPredict)
        selectFileButton.place(x=self.canvasSize + 50, y=self.canvasSize /
                            2 + self.marginSize - self.positionAdjust)

        self.displayResult()

        print("Loading saved model...")
        self.resNet = torch.load("./finalModel")
        print("Saved model loaded!")
        
        self.resNet.eval()
        self.resNet.cuda()
        self.initialized = True
        print("Application initialized!")


    def modelPredict(self):
        if(not self.imageLoaded):
            print("Image not loaded! Please click select before predict!")
            return
        else:
            self.classes = ['Cloth', 'Glass', 'Hazardous Waste', 'Metal', 'Organic', 'Other', 'Paper', 'Plastic']
            softMax = nn.Softmax(dim=1)
            self.img_tensor = self.img_tensor.unsqueeze(0)
            self.img_tensor = self.img_tensor.float().cuda()
            self.img_tensor = self.img_tensor/255
            
            self.predictions = self.resNet(self.img_tensor)
            self.predictions = softMax(self.predictions)[0]

            self.predictions = self.predictions.cpu().detach().numpy().tolist()

            percentPred = []
            sortedClasses = []

            # sort self.predictions from lowest to highest
            while(len(percentPred) != 8):
                tempLowest = 2
                index = -1
                tempReal = ""
                for i, pred in enumerate(self.predictions, 0):
                    tempString = "{:.1f}".format(pred * 100)
                    if(pred <= tempLowest):
                        tempReal = tempString
                        index = i
                        tempLowest = pred
                sortedClasses.append(self.classes[index])
                percentPred.append(tempReal)
                self.predictions.pop(index)
                self.classes.pop(index)

            finalPrediction = sortedClasses[-1]

            # make class reference
            self.percentPred = percentPred
            self.sortedClasses = sortedClasses
            self.finalPrediction = finalPrediction
            
            for pred in reversed(range(len(sortedClasses))):
                print(f"    {percentPred[pred]}%: {sortedClasses[pred]}")
            
            print(f"Model's final prediction is: {finalPrediction}")

            self.predicted = True
            self.displayResult()


    def selectFile(self):
        filePath = filedialog.askopenfile().name
        print("File selected:", filePath) 
        imageFile = PIL.Image.open(filePath)

        transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize((224, 224))
        ])

        self.img_tensor = transform(imageFile)

        imageFile = imageFile.resize((224*self.pixelSize + self.positionAdjust, 224*self.pixelSize + self.positionAdjust))
        self.photoImage = PIL.ImageTk.PhotoImage(imageFile)
        self.canvas.create_image(0, 0, anchor=NW, image=self.photoImage)

        self.imageLoaded = True

        return

    def displayResult(self):
        finalPredText = "Unknown"
        if(self.predicted):
            finalPredText = self.finalPrediction
        
        if(not self.predicted):
            self.labels = []
            self.labelTexts = []

        self.finalPredictionLabel = Label(self.root, text="Model's Final Prediction: \n" + "    " + finalPredText, font= ('Aerial', 17))
        self.finalPredictionLabel.place(x=224*3 + 100, y=9*40 + 100)

        modelPredictionLabel = Label(self.root, text="Model's Predictions: ", font= ('Aerial', 17))
        modelPredictionLabel.place(x=224*3 + 100, y=80)

        for labelIndex in range(8):
            if(self.predicted):
                self.labelTexts[labelIndex].set(self.percentPred[7 - labelIndex] + " % ： " +  self.sortedClasses[7 - labelIndex])
            else :
                labelText = StringVar()
                labelText.set("xxx.x %" + " : " + self.classes[labelIndex]) 
                label= Label(self.root, textvariable= labelText, font= ('Aerial', 17))
                label.place(x=224*3 + 150, y=labelIndex*40 + 130)
                
                self.labels.append(label)
                self.labelTexts.append(labelText)
                
        return

def main():
    window = Tk()
    demo = demoApp(window)
    window.mainloop()


if __name__ == "__main__":
    main()
