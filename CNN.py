from PIL import Image
import numpy as np
import scipy.misc as smp
            
def createCol(data):
    out = np.array(data)
    out.reshape([len(data), 1])
    return out

class Convolution:
    def __init__(self, w, listIn=[]):
        self.arr = np.array(listIn)
        self.w = w
        
        if (listIn == []):
            self.arr = np.random.rand(w, w)
    
    def add(self, arrIn):
        self.arr = self.arr + arrIn
    
    def applyToImg(self, image):
#        image = np.pad(image, int(self.w/2), 'constant')
        imgW = image.shape[1]
        imgH = image.shape[0]
        out = np.zeros([imgH - self.w+1, imgW - self.w+1])
        for i in range(0, imgH - self.w+1):    #for every pixel in the image (except an outer layer)
            for j in range(0, imgW - self.w+1):
                newVal = np.sum(np.multiply(image[i:i+self.w,j:j+self.w], self.arr)) / np.sum(self.arr)
                out[i][j] = newVal
        return out
            

#Each layer of the Neural Network has properties, those are stored here    
class Layer:
    def __init__(self, convSize):
        self.conv = Convolution(convSize)
        self.convSize = convSize
        
    def compute(self, inputData):
        return self.conv.applyToImg(inputData)

#The Nerual Network arcitechture, contains functions to train the network and give a guess
#Input assumed to be a 2d array of image brightness values (normalized)
class CNN:
    def __init__(self, convSize, inputShape):
        self.layers = []
        self.layerOut = []
        self.convSize = convSize
        self.inputShape = inputShape
        self.learningRate = 0.05    #default learning rate
        self.initLayers()
        
    def initLayers(self):
        initialInput = np.zeros(self.inputShape)
        index = 0;
        self.layers.append(Layer(self.convSize))
        self.layerOut.append(self.layers[index].compute(initialInput))
        while (self.layerOut[index].size > 1):
            index = index + 1
            self.layers.append(Layer(self.convSize))
            self.layerOut.append(self.layers[index].compute(self.layerOut[index-1]))
        
        
    #set the learning rate
    def setLearningRate(self, newRate):
        self.learningRate = newRate
    def getLearningRate(self):
        return self.learningRate
        
    #make a guess of what an output should be base on inputs
    def feedForward(self, inputData):
        curData = inputData
        index = 0
        for layer in self.layers:
            curData = layer.compute(curData)
            self.layerOut[index] = curData
            index = index + 1
        guess = curData
        return guess
    
    #train the neurons by changing weights and bias based on error from real output
    def train(self, inputData, expectedOutput):
        currentOutput = self.feedForward(inputData)
        layerOutputs = [inputData] + self.layerOut
        errorOutput = np.array(expectedOutput) - np.array(currentOutput)
        
        for i in range(len(self.layerOut)):
            index = len(layerOutputs) - i - 1
            gradient = self.learningRate * errorOutput
            prevInput = layerOutputs[index-1]
            errorOutput = np.zeros(prevInput.shape)
            for i in range(gradient.shape[0]):
                for j in range(gradient.shape[1]):
                    errorOutput[i:i+self.convSize, j:j+self.convSize] += self.layers[index-1].conv.arr * gradient[i,j]
                    self.layers[index-1].conv.add(prevInput[i:i+self.convSize, j:j+self.convSize] * gradient[i,j])
    
    #get the weight change to maximize what the network is trained for
    def getAdjustOutput(self, inputData):
        expectedOutput = np.ones([self.outputNum, 1])
        currentOutput = self.feedForward(inputData)
        layerOutputs = [inputData] + self.layerOut
        errorOutput = np.array(expectedOutput) - np.array(currentOutput)
        
        for i in range(len(self.layerOut)):
            index = len(layerOutputs) - i - 1
            errorOutput = self.layers[index-1].weights.transpose().dot(errorOutput)
        adjustment = self.learningRate * errorOutput * (inputData * (1 - inputData))
        return adjustment
    
    
c = Convolution(5); 

img = Image.open('google.jpg')
brightness = []
pix = img.load()
for i in range(img.size[0]):
    row = []
    for j in range(img.size[1]):
        colour = pix[j,i]
        row.append((colour[0]+colour[1]+colour[2])/3)
    brightness.append(row)
brightness = np.array(brightness)/255
brightness = brightness[:99,:99]


cnn = CNN(5, [99, 99])
output = cnn.feedForward(brightness)

#print("Training network...")
#for i in range(100):
#    print(i)
#    cnn.train(brightness, [1])
#print("Done training")