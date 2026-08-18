# This is a basic program aimed at solving the MNIST dataset. It is written as simply as I could, as it is my first program
# There are no classes, as I wanted to keep it as simple as possible (even if it sacrifices scalability and code reusability)



import torch
import torchvision ## Contains some utilities for working with the image data
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

# Load training data
mnist = MNIST(root = 'data/', download = True, train = True, transform = transforms.ToTensor())
images = []
labels = []
for image, label in mnist:
    images.append(image.flatten())
    labels.append(label)

print(len(images))

inputSizeLayer1 = 8 * 8 # each MNIST image is 8x8 pixels
outputSizeLayer1 = 256 # intermediate value

# create the weights for layer 1
w1 = torch.rand(inputSizeLayer1, outputSizeLayer1, requires_grad=True)
b1 = torch.zeros(inputSizeLayer1, requires_grad=True)
weights1 = [w1, b1]

inputSizeLayer2 = outputSizeLayer1 # output size of layer 1. 
outputSizeLayer2 = 10 # Numbers 0-9

# create the weights for layer 1
w2 = torch.rand(inputSizeLayer2, outputSizeLayer2, requires_grad=True)
b2 = torch.zeros(inputSizeLayer2, requires_grad=True)
weights2 = [w2, b2]

# create a dense layer (to run the forward pass)
# input is the input data (vector, usually), w is the weigth that is multiplied, and b is the weight that is added
def denseLayer1(input, w, b):
    return (torch.relu(torch.matmul(input, w) + b)) # perform an affine transformation (no information is lost). 

def denseLayer2(input, w, b):
    return (torch.softmax(torch.matmul(input, w) + b, dim=0)) # perform an affine transformation (no information is lost). 

def model(inputs, weights):
    output = torch.zeros(outputSizeLayer2) # list of probabilities
    for input in inputs:
        output = output + denseLayer2( denseLayer1(input, weights[0][0], weights[0][1]), weights[1][0], weights[1][1] )
    return output / len(inputs)

#   inputs is the input images (batch). labels is a list of what the images actually are
def trainingStep(inputs, labels): # takes a patch of inputs and uses them for training
    lossFunction = torch.nn.CrossEntropyLoss()
    weights = [weights1, weights2]
    predictions = model(inputs, weights)
    loss = lossFunction(predictions, labels)
    averageLoss = torch.mean(loss)

    averageLoss.backward() # calculate the gradients

    # update the model weights using the most elegent code I have ever written
    weights[0][0] = weights[0][0] - weights[0][0].grad
    weights[0][1] = weights[0][1] - weights[0][1].grad
    weights[1][0] = weights[1][0] - weights[1][0].grad
    weights[1][1] = weights[1][1] - weights[1][1].grad

    # Reset gradients
    weights[0][0].grad = None
    weights[0][1].grad = None
    weights[1][0].grad = None
    weights[1][1].grad = None

    return averageLoss
    
    

def generateBatch(batchIndex, batchSize):
    return images[batchSize * batchIndex : batchSize * (batchIndex + 1)], labels[batchSize * batchIndex : batchSize * (batchIndex + 1)]

def trainingLoop(inputs, labels):
    for batchCounter in range()

def main():
    var1 = torch.ones(size=(2, 4))
    print(var1.shape)
    var3 = torch.tensor([[1., 2., 3., 4.], 
                         [4., 3., 2., 1.]])
    print(var3.shape)
    var2 = torch.normal(mean=var3, std=var1)

    var1 = torch.ones(size=(8, 2))

    print(var1)
    print(var2)

    print(torch.matmul(var1, var2))

main()