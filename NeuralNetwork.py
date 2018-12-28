"""
Created: July 2018
Updated: December 2018

@author: Andrew Farley
"""

import numpy as np
from numpy import array #needed to read numpy arrays from strings with eval()

#------------ HELPER FUNCTIONS -----------------------------

#Checks if incoming data needs to be turned into column vectors, returns the correctly shaped data
#data ----- a 1D numpy array or python list
def checkReform(data):
    out = np.array(data)
    if (out.ndim < 1): #invalid data, give a blank column vector
        return np.array([[]])
    elif (out.ndim == 1): #currently a 1D vector, reshape into column
        shape = out.shape
        out = out.reshape(shape[0], 1)
        return out
    else: #already valid, return input
        return out
    
#Writes a line to a file
#file ----- an open writing file object 
#text ----- the line to write
def writeLine(file, text):
    file.write(text)
    file.write('\n')
    
#Reads a line from a file as an actual value
#file ----- an open reading file object 
def readLine(file):
    return eval(file.readline().strip())

#Loads network settings from a text file and returns it
#fileName ----- the filename to load from
def loadNetwork(fileName):
    file = open(fileName, 'r')
    
    inputNum = readLine(file) #reads the structure settings
    hiddenNum = readLine(file)
    outputNum = readLine(file)
    out = NeuralNetwork(inputNum, hiddenNum, outputNum) #creates a new network
    out.setLearningRate(readLine(file)) #sets the learning rate
    
    weights = []
    biases = []
    numLayers = readLine(file)
    for i in range(numLayers): #reads all weights and biases into a big array to set later
        weights.append(readLine(file))
        biases.append(readLine(file))
    out.setLayers(weights, biases) #sets the weights and biases
    
    file.close()
    return out

#-----------------------------------------------------------    


#Each layer of the Neural Network has properties, those are stored here    
class Layer:
    #Layer constructor
    #inputSize ----- the number of inputs (or neurons) from the previous layer
    #selfSize ----- the number of neurons on this layer
    def __init__(self, inputSize, selfSize):
        self.weights = 2 * np.random.rand(selfSize, inputSize) - 1  #giving random weights to start
        self.bias = 2 * np.random.rand(selfSize, 1) - 1 #giving random bias to start
        
    #Feeds an input vector through the layer to produce an output
    #inputData ----- a 1D array or list of input values with same length as inputSize
    def compute(self, inputData):
        out = self.weights.dot(inputData)  #feeding the given inputs through the current layerout)
        out = out + self.bias
        out = 1 / (1 + np.exp(-out))    #sigmoid activation function
        return out
    
    #String version of the layer
    def stringify(self):
        weightsStr = np.array_repr(self.weights) #get string version of numpy array
        weightsStr = weightsStr.replace('\n', '') #make sure it will be on one line
        biasStr = np.array_repr(self.bias) #same but with bias
        biasStr = biasStr.replace('\n', '')
        return weightsStr, biasStr
    
    #Set current layer's weights and bias 
    def setVals(self, weights, bias):
        self.weights = weights
        self.bias = bias
        

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
        self.inputNum = inputNum
        self.hiddenNum = hiddenNum
        self.outputNum = outputNum
        
    #Sets the learning rate
    def setLearningRate(self, newRate):
        self.learningRate = newRate
    def getLearningRate(self):
        return self.learningRate
        
    #Make a guess of what an output should be based on inputs
    #inputData ----- a 1D array or list of input values with same length as inputNum
    def feedForward(self, inputData):
        curData = checkReform(inputData) #make sure data is column vector
        index = 0
        for layer in self.layers: #feed through every layer
            curData = layer.compute(curData)
            self.layerOut[index] = curData #set the current layer output
            index = index + 1
        probabilities = curData #the final output which is a list of probabilities for certain outputs
        return probabilities
    
    #Train the neurons by changing weights and bias based on error from real output
    #inputData ----- a 1D array or list of input values with same length as inputNum
    #expectedOutput ----- a 1D array or list of expected output values for the input with same length as outputNum
    def train(self, inputData, expectedOutput):
        currentOutput = self.feedForward(inputData) #get what the layer currently outputs
        layerOutputs = [checkReform(inputData)] + self.layerOut 
        errorOutput = checkReform(expectedOutput) - checkReform(currentOutput) #current error between expected and output
        
        for i in range(len(self.layerOut)): #for every layer...
            index = len(layerOutputs) - i - 1
            gradient = self.learningRate * errorOutput * (layerOutputs[index] * (1 - layerOutputs[index])) #get the gradient
            weightChange = gradient.dot(layerOutputs[index-1].transpose()) #dot product it with the previous layer output
            self.layers[index-1].weights = self.layers[index-1].weights + weightChange #adjust the weights
            self.layers[index-1].bias = self.layers[index-1].bias + gradient #adjust the bias
            errorOutput = self.layers[index-1].weights.transpose().dot(errorOutput) #propagate the error backwards through the network
    
    #returns the index of the highest value in the output vector
    #inputData ----- a 1D array or list of input values with same length as the inputNum
    def guess(self, inputData):
        prob = self.feedForward(inputData)
        prob = np.array(prob)
        return prob.argmax() #getting the index of the max value of the probabilities output
    
    #Saves the network's settings to a text file, format is in README
    #fileName ----- the filename to save to
    def save(self, fileName):
        file = open(fileName, 'w')
        
        writeLine(file, str(self.inputNum)) #writing the shape of the network on 3 different lines
        writeLine(file, str(self.hiddenNum[1:]))
        writeLine(file, str(self.outputNum))
        writeLine(file, str(self.learningRate)) #learning rate
        writeLine(file, str(len(self.layers))) #number of layers
        for layer in self.layers: #for every layer, write the weights and bias on seperate lines (2 lines per layer)
            layerStrings = layer.stringify()
            writeLine(file, layerStrings[0])
            writeLine(file, layerStrings[1])
            
        file.close() #close out the file to finish
    
    #Sets all the layers in the network with new values according to input data
    #weights ----- a list of numpy arrays of the weights of each layer
    #biases----- a list of numpy arrays of the bias of each layer
    def setLayers(self, weights, biases):
        if (len(weights) != len(self.layers)): #mismatch between number of weights and number of layers
            print("Error: Length of input arrays does not match number of layers")
            return
        for i in range(len(weights)): #set all layers
            self.layers[i].setVals(weights[i], biases[i])
        
    
    
#------------------- TESTING --------------------
#This code tests the Neural Network by training it to compute the XOR binary function (something not possible with a perceptron)
#Just run testXOR() in the console, it should print:
# 1
# 1
# 0
# 0
#If this doesn't print something is terribly wrong
            
def testXOR():
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
