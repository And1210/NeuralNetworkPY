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

