"""
Created: July 2018
Updated: December 2018

@author: Andrew Farley


"""

import numpy as np

#Checks if incoming data needs to be turned into column vectors, returns the correctly shaped data
#data ----- a 1D numpy array or python list
def checkReform(data):
    out = np.array(data)
    if (out.ndim < 1):
        return np.array([[]])
    elif (out.ndim == 1):
        shape = out.shape
        out = out.reshape(shape[0], 1)
        return out
    else:
        return out

#Each layer of the Neural Network has properties, those are stored here    
class Layer:
    #inputSize ----- the number of inputs (or neurons) from the previous layer
    #selfSize ----- the number of neurons on this layer
    def __init__(self, inputSize, selfSize):
        self.weights = 2 * np.random.rand(selfSize, inputSize) - 1  #giving random weights to start
        self.bias = 2 * np.random.rand(selfSize, 1) - 1 #giving random bias to start
        
    #inputData ----- a 1D array or list of input values with same length as inputSize
    def compute(self, inputData):
        out = self.weights.dot(inputData)  #feeding the given inputs through the current layerout)
        out = out + self.bias
        out = 1 / (1 + np.exp(-out))    #sigmoid activation function
        return out

#The Nerual Network architechture, contains functions to train the network and give a guess
class NeuralNetwork:
    #Constructor of the Neural Network
    #inputNum ----- the number of inputs to the NN
    #hiddenNum ----- a 1D list containing the number of neurons on each hidden layer, can be any size n where n > 1
    #outputNum ----- the number of outputs from the NN
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
    #inputData ----- a 1D array or list of input values with same length as inputNum
    def feedForward(self, inputData):
        curData = checkReform(inputData)
        index = 0
        for layer in self.layers:
            curData = layer.compute(curData)
            self.layerOut[index] = curData
            index = index + 1
        probabilities = curData
        return probabilities
    
    #train the neurons by changing weights and bias based on error from real output
    #inputData ----- a 1D array or list of input values with same length as inputNum
    #expectedOutput ----- a 1D array or list of expected output values for the input with same length as outputNum
    def train(self, inputData, expectedOutput):
        currentOutput = self.feedForward(inputData)
        layerOutputs = [checkReform(inputData)] + self.layerOut
        errorOutput = checkReform(expectedOutput) - checkReform(currentOutput)
        
        for i in range(len(self.layerOut)):
            index = len(layerOutputs) - i - 1
            gradient = self.learningRate * errorOutput * (layerOutputs[index] * (1 - layerOutputs[index]))
            weightChange = gradient.dot(layerOutputs[index-1].transpose())
            self.layers[index-1].weights = self.layers[index-1].weights + weightChange
            self.layers[index-1].bias = self.layers[index-1].bias + gradient
            errorOutput = self.layers[index-1].weights.transpose().dot(errorOutput)
    
    #returns the index of the highest value in the output vector
    #inputData ----- a 1D array or list of input values with same length as the inputNum
    def guess(self, inputData):
        prob = self.feedForward(inputData)
        prob = np.array(prob)
        return prob.argmax()
    
    
#------------------- TESTING --------------------
#This code tests the Neural Network by training it to compute the XOR binary function (something not possible with a perceptron)

#Initialize the network
nn = NeuralNetwork(2, [5], 2) 

#Setting up the training data
data = []
data.append(np.array([0, 1]))
data.append(np.array([1, 0]))
data.append(np.array([1, 1]))
data.append(np.array([0, 0]))
out = [[0, 1], [0, 1], [1, 0], [1, 0]] #The expected output for every input added above

#Train the network
import random as rand
for i in range(50000):
    index = rand.randint(0, 3)
    nn.train(data[index], out[index])
    
#Print the guess for every input
print('')
for ar in data:
    print(nn.guess(ar))
