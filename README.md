# NeuralNetworkPY
A basic, easy-to-use, neural network library built from scratch in python. Only dependency is numpy.

## Examples
I find examples are what I want when I go to a readme so I'm going to start with it. References for everything in this package can be found below.

### Creating a Network to Understand XOR
```python
    #Initialize the network with 2 inputs, 1 hidden layer with 5 neurons, 2 outputs
    nn = NeuralNetwork(2, [5], 2) 
    
    #Setting up the training data
    data = []
    data.append(np.array([0, 1]))
    data.append(np.array([1, 0]))
    data.append(np.array([1, 1]))
    data.append(np.array([0, 0]))
    out = [[0, 1], [0, 1], [1, 0], [1, 0]] #The expected output for every input added above 
    #(i.e. 0 xor 1 = 1  which in array form is [0, 1] because the guess function returns the index of the highest value, 1 in this case)
    
    #Train the network
    import random as rand
    for i in range(50000):
        index = rand.randint(0, 3) #randomizes which value we train with on each iteration
        nn.train(data[index], out[index])
        
    #Print the guess for every input
    print('')
    for ar in data:
        print(str(ar[0]) + " xor " + str(ar[1]) + " = " + str(nn.guess(ar)))
```

### NeuralNetwork and MNIST Example ("The hello world! of machine learning")
```python
#This may take a long time to run, 2-10 minutes, depending on the speed of your cpu
#I know it isn't commented the best, I've been writing documentation for hours at this point lol

import NeuralNetworkPY
import numpy as np
import struct
import random as rand

#Data obtained from http://yann.lecun.com/exdb/mnist/. These files MUST be in same directory

#Function obtained from https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    
def prepImgs(raw_images):
    images = np.array([None for i in range(len(raw_images))])
    for i in range(len(raw_images)):
        images[i] = raw_images[i].flatten()
        images[i] = images[i] / 255
    return images
    
def createOutput(val):
    out = []
    for i in range(10):
        if (i == val):
            out.append(1.0)
        else:
            out.append(0.0)
    return np.array([out]).transpose()

def guessToNum(guess):
    max = guess[0]
    index = 0
    for i in range(len(guess)-1):
        if (guess[i+1] > max):
            max = guess[i+1]
            index = i+1
    return index
    
#Load images and labels from files
raw_images = read_idx('train-images.idx3-ubyte')
labels = read_idx('train-labels.idx1-ubyte')

#Prep the images into a big array (need to normalize all values and flatten the 2D array to a 1D)
images = prepImgs(raw_images)
    
#Setup network with 784 inputs (the images are 28*28 = 784), one hidden layer with 50 neurons, and 10 outputs (there are 10 possible values)
digitRecog = NeuralNetworkPY.NeuralNetwork(784, [50], 10)
digitRecog.setLearningRate(0.05)

for n in range(10): #train the network in 10 epochs
    for i in range(len(images)): #in each epoch, train the network with a random image n times where n is the number of images
        index = rand.randint(0, len(images)-1) #choose random image
        digitRecog.train(images[index], createOutput(labels[index])) #train the network for one iteration
    
#get and setup testing data
test_raw_images = read_idx('t10k-images.idx3-ubyte')
test_images = prepImgs(test_raw_images)
test_labels = read_idx('t10k-labels.idx1-ubyte')
correct = 0
total = len(test_images)

#loop through all test images, count how many the network got right
for i in range(len(test_images)):
    guess = digitRecog.feedForward(test_images[i])
    if (guessToNum(guess) == test_labels[i]):
        correct = correct + 1
        
#print results
print("The Network guessed " + str(correct) + "/" + str(total) + " (" + str(100*correct/total) + "%) correct")
```


## Object References
### NeuralNetwork:
  - Description: The NeuralNetwork object which is what is trained and where inputs are passed in. This is what you'll want
  - Functions:
    - `init(inputNum, hiddenNum, outputNum)` (the constructor):
      - Description: Creates a NeuralNetwork object with the structure given as inputs
      - Inputs:
        - `inputNum`
          - Description: The number of inputs to the network (size of input array)
          - Type Required: int
        - `hiddenNum`
          - Description: The size of every hidden layer in the network
          - Type Required: list of int
        - `outputNum`
          - Description: The number of outputs from the network (size of output array)
          - Type Required: int
    - `feedForward(inputData)`:
      - Description: Computes the feed forwards algorithm for the network. Will propagate a given input array through every layer to produce an output array.
      - Inputs:
        - `inputData`
          - Description: The input array to be propogated through the network.
          - Type Required: 1D python list or numpy array (such as [1, 2, 3])
      - Outputs:
        - `probabilities`
          - Description: The confidence values for every possible output.
          - Type: Column vector in a numpy array. Will have shape (n, 1) where n is the number of elements.
    - `train(inputData, expectedOutput)`:
      - Description: Processes one iteration of network training, will adjust the weights and bias based on the error between the expected output and the current output for a given input.
      - Inputs:
        - `inputData`
          - Description: The input array to be trained with
          - Type Required: 1D python list or numpy array (such as [1, 2, 3])
        - `expectedOutput`
          - Description: What the given input array should produce in the network
          - Type Required: 1D python list or numpy array (such as [1, 2, 3])
    - `guess(inputData)`:
      - Description: Produces a guess using forward propagation for a given input. The guess is the highest confidence value in the output array from feedForward()
      - Inputs:
        - `inputData`
          - Description: The input array to get a guess from
          - Type Required: 1D python list or numpy array (such as [1, 2, 3])
      - Outputs:
        - `output`
          - Description: The index of the highest confidence value in the feedForward output.
          - Type: int
    - `setLearningRate(newRate)`:
      - Description: Sets the learning rate of the network
      - Inputs:
        - `newRate`
          - Description: The new learning rate
          - Type Required: float
    - `getLearningRate()`:
      - Description: Gets the learning rate of the network
      - Outputs:
        - `output`
          - Description: The learning rate of the current network.
          - Type: float
    - `save(fileName)`:
      - Description: Saves the network to a text file for later use relative to the current directory. Format is provided below.
      - Inputs:
        - `fileName`
          - Description: The name of the file (ex. "test.txt").
          - Type Required: str
    - `setLayers(weights, biases)`:
      - Description: Sets the given network based on a list of weights and biases.
      - Inputs:
        - `weights`
          - Description: A python list (must be the same size as self.layers) containing the new weight matricies for each layer.
          - Type Required: Python list of 2D numpy arrays
        - `biases`
          - Description: A python list (must be the same size as self.layers) containing the new bias matricies for each layer.
          - Type Required: Python list of 2D numpy arrays
  
### Layer:
  - Description: Used by the NeuralNetwork object for every layer (the inputs not being a layer). Contains information for every neuron in the layer which includes the weight connections and biases.
  - Functions:
    - `init(inputSize, selfSize)` (the constructor):
      - Description: Creates a Layer object with the structure as given by parameters
      - Inputs:
        - `inputSize`
          - Description: The number of inputs from the previous layer. Could also be thought of as the number of neurons on the previous layer.
          - Type Required: int
        - `selfSize`
          - Description: The number of neurons on the current layer.
          - Type Required: int
    - `compute(inputData)`:
      - Description: Propagates a given array of inputs through the current layer. This involves computing the necessary weighted sums for every neuron and adding the bias. Matrix math is used.
      - Inputs:
        - `inputData`
          - Description: The input array to propagate through this layer
          - Type Required: A column vector of inputs between -1 and 1. The shape of the numpy array is (n, 1) where n is the number of elements.
      - Outputs:
        - `out`
          - Description: The output from the current layer with the given input.
          - Type: A column vector of inputs between -1 and 1. The shape of the numpy array is (n, 1) where n is the number of elements.
    - `stringify()`:
      - Description: Gives a string representation of the layer. Used for saving the network to a text file.
      - Outputs:
        - `weightsStr`
          - Description: The string of the weights matrix, stripped of newline characters.
          - Type: str
        - `biasStr`
          - Description: The string of the bias matrix, stripped of newline characters.
          - Type: str
    - `setVals(weights, bias)`:
      - Description: Sets the weights and bias of the current layer to the given inputs.
      - Inputs:
        - `weights`
          - Description: A matrix of the weights, needs to match the size of the current one.
          - Type Required: 2D numpy array
        - `bias`
          - Description: A matrix of the bias, needs to match the size of the current one.
          - Type Required: 2D numpy array
    
## Helper Functions
### `checkReform(data)`
  - Description: Checks an input data list or array, ensures it is a column vector, reshapes it if neccessary, returns the clean output.
  - Inputs:
    - `data`
      - Description: A list or array that is going to be propagated through the array.
      - Type Required: python list or numpy array
  - Outputs:
    - `out`
      - Description: The cleaned input, is ready for use by a NeuralNetwork object
      - Type: A column vector with the shape of the numpy array being (n, 1) where n is the number of elements.
      
### `writeLine(file, text)`
  - Description: Writes a line to an open file object with write permissions.
  - Inputs:
    - `file`
      - Description: The file object to write to, must have write permissions.
      - Type Required: _io.TextIOWrapper object
    - `text`
      - Description: The text to write to the line
      - Type Required: str
      
### `readLine(file)`
  - Description: Reads the next line from an open file object with read permissions evaluated as a python variable.
  - Inputs:
    - `file`
      - Description: The file object to read from, must have read permissions.
      - Type Required: _io.TextIOWrapper object
  - Outputs:
    - `output`
      - Description: Evaluates the line recieved as a python value. (i.e. the string "array([1, 2, 3])" would evaluate as a numpy array containing [1, 2, 3])
      - Type: depends on the string evaluated
      
### `loadNetwork(fileName)`
  - Description: Reads a network info file with the given name and recreates a Neural Network out of it.
  - Inputs:
    - `fileName`
      - Description: The name of the saved Neural Network data file. (i.e. "test.txt")
      - Type Required: str
  - Outputs:
    - `out`
      - Description: A Neural Network with the same specifications as outlined by the file data.
      - Type: NeuralNetwork object

      
## Save File Format
**line**    **value**
1. inputNum 
2. hiddenNum
3. outputNum
4. learningRate 
5. number of layers (layerNum) 
6. layer[0] Weights 
7. layer[0] Bias 

......... 

(5+2xlayerNum-1). layer[layerNum-1] Weights 

(5+2xlayerNum). layer[layerNum-1] Bias 


## Dev Notes
I am currently working on building a Convolutional Neural Network from scratch modelled after the Neural Network created here. The code that is being worked on can be found here as well.
