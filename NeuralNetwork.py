import numpy as np
            
def createCol(data):
    out = np.array(data)
    out.reshape([len(data), 1])
    return out
    

#Each layer of the Neural Network has properties, those are stored here    
class Layer:
    def __init__(self, inputSize, selfSize):
        self.weights = 2 * np.random.rand(selfSize, inputSize) - 1  #giving random weights to start
        self.bias = 2 * np.random.rand(selfSize, 1) - 1 #giving random bias to start
        
    def compute(self, inputData):
        out = self.weights.dot(inputData)  #feeding the given inputs through the current layer
#        out = (out - np.amin(out)) #renormalizing the data so it's evenly distributed
#        out = out / np.amax(out)
        out = out + self.bias
        out = 1 / (1 + np.exp(-out))    #sigmoid activation function
#        out = np.exp(out) / np.sum(np.exp(out))
        return out

#The Nerual Network arcitechture, contains functions to train the network and give a guess
class NeuralNetwork:
    def __init__(self, inputNum, hiddenNum, outputNum):
        #Setting up layers for network
        hiddenLayers = len(hiddenNum)
        hiddenNum = [inputNum] + hiddenNum
        self.layers = [Layer(hiddenNum[i], hiddenNum[i+1]) for i in range(hiddenLayers)]
        self.layers.append(Layer(hiddenNum[hiddenLayers], outputNum))
        self.layerOut = [None for i in range(hiddenLayers + 1)]
        self.learningRate = 0.05    #default learning rate
        self.outputNum = outputNum
        
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
            gradient = self.learningRate * errorOutput * (layerOutputs[index] * (1 - layerOutputs[index]))
            weightChange = gradient.dot(layerOutputs[index-1].transpose())
            self.layers[index-1].weights = self.layers[index-1].weights + weightChange
            self.layers[index-1].bias = self.layers[index-1].bias + gradient
            errorOutput = self.layers[index-1].weights.transpose().dot(errorOutput)
    
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
    
#nn = NeuralNetwork(2, [5], 1)
#
##   np.array(inputData, dtype='f').transpose()
#
#data = []
#data.append(np.array([[0, 1]], dtype='f').transpose())
#data.append(np.array([[1, 0]], dtype='f').transpose())
#data.append(np.array([[1, 1]], dtype='f').transpose())
#data.append(np.array([[0, 0]], dtype='f').transpose())
#out = [[[1]], [[1]], [[0]], [[0]]]
#guess = nn.feedForward(data[0])
#print(guess)
#
#import random as rand
#for i in range(50000):
#    index = rand.randint(0, 3)
#    nn.train(data[index], out[index])
#    
#    
#print('')
#for ar in data:
#    print(nn.feedForward(ar))