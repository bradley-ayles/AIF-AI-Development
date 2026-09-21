# This is a basic program aimed at solving the EMNIST dataset (this includes letters as well as numbers). It is written as simply as I could, as it is my first program
# There are no classes, as I wanted to keep it as simple as possible (even if it sacrifices scalability and code reusability)

import sys

import torch
import torchvision ## Contains some utilities for working with the image data
from torchvision.datasets import EMNIST
import torchvision.transforms as transforms


# make the model run faster
# import intel_extension_for_pytorch as ipex

# Check if your 11th Gen iGPU is detected
if torch.xpu.is_available():
    device = torch.device("xpu")
    print(f"Success! Using iGPU: {torch.xpu.get_device_name(0)}")
    torch.set_default_device(device)
else:
    device = torch.device("cpu")
    print("iGPU not found. Falling back to CPU.")


# Size of batches for MBGD
BATCH_SIZE = 128
LEARNING_RATE = 0.07

print("Downloading EMNIST training data")

# Load training data
emnist_train = byclass_train_data = EMNIST(
    root="data/",
    split="byclass",
    train=True,
    transform=transforms.ToTensor(),
    # target_transform=None,
    download=False # Change to True to download data
)

emnist_test = byclass_train_data = EMNIST(
    root="data/",
    split="byclass",
    train=False,
    transform=transforms.ToTensor(),
    # target_transform=None,
    download=False # Change to True to download data
)

# when downloading the training data, return right away
# sys.exit()

print("EMNIST training data downloaded")

trainingImages = emnist_train.data.reshape(len(emnist_train), -1).float().div_(255)
trainingLabels = emnist_train.targets

testImages = emnist_test.data.reshape(len(emnist_test), -1).float().div_(255)
testLabels = emnist_test.targets


print(f"Training data loaded\nTraining images count: {len(trainingImages)}, test images count: {len(testImages)}")

# Set the domain for the random values for initialising w1 and w2
randDomainMax = 0.05
randDomainMin = -0.05

inputSizeLayer1 = 28 * 28 # each MNIST image is 28x28 pixels
outputSizeLayer1 = 256 # intermediate value

# create the weights for layer 1
w1 = torch.tensor( (randDomainMax - randDomainMin) * torch.rand(inputSizeLayer1, outputSizeLayer1) + randDomainMin, requires_grad=True )
b1 = torch.zeros(outputSizeLayer1, requires_grad=True)
weights1 = [w1, b1]

inputSizeLayer2 = outputSizeLayer1 # output size of layer 1. 
outputSizeLayer2 = 62 # 26 lowercase, 26 uppercase 10 numbers

# create the weights for layer 1
w2 = torch.tensor( (randDomainMax - randDomainMin) * torch.rand(inputSizeLayer2, outputSizeLayer2, requires_grad=True) + randDomainMin, requires_grad=True )
b2 = torch.zeros(outputSizeLayer2, requires_grad=True)
weights2 = [w2, b2]

# create a dense layer (to run the forward pass)
# input is the input data (vector, usually), w is the weigth that is multiplied, and b is the weight that is added
def denseLayer1(input, w, b):
    return (torch.relu(torch.matmul(input, w) + b)) # perform an affine transformation (no information is lost), then relu ( all values >0)

def denseLayer2(input, w, b):
    return torch.matmul(input, w) + b # perform an affine transformation (no information is lost). 

def model(inputs, weights):
    outputs = torch.zeros(len(inputs), outputSizeLayer2) # list of probabilities
    for inputIndex, input in enumerate(inputs):
        outputs[inputIndex] = denseLayer2( denseLayer1(input, weights[0][0], weights[0][1]), weights[1][0], weights[1][1] )
    return outputs

#   inputs is the input images (batch). labels is a list of what the images actually are
def trainingStep(inputs, labels): # takes a patch of inputs and uses them for training
    # Access the weights
    global w1
    global b1
    global w2
    global b2

    lossFunction = torch.nn.CrossEntropyLoss()
    predictions = model(inputs, [[w1, b1], [w2, b2]])
    loss = lossFunction(predictions, labels) # torch.tensor(labels) is needed if labels is a list

    loss.backward() # calculate the gradients

    assert w1.grad != None
    assert b1.grad != None
    assert w2.grad != None
    assert b2.grad != None

    # update the model weights using the most elegent code I have ever written
    # no_grad prevents pytorch from treating this operation as a part of the forward pass
    with torch.no_grad():
        w1 -= w1.grad * LEARNING_RATE
        b1 -= b1.grad * LEARNING_RATE
        w2 -= w2.grad * LEARNING_RATE
        b2 -= b2.grad * LEARNING_RATE

    # Reset gradients
    w1.grad = None
    b1.grad = None
    w2.grad = None
    b2.grad = None

    return loss
    
    
# generate a batch of tensors
def generateBatch(batchIndex, batchSize):
    return (trainingImages[batchSize * batchIndex : batchSize * (batchIndex + 1)]), (trainingLabels[batchSize * batchIndex : batchSize * (batchIndex + 1)])

def trainingLoop(epochs):
    averageLoss = 0.
    accuracyHistory = []

    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        for batchCounter in range(int(len(trainingImages) / BATCH_SIZE)):
            imagesBatch, labelsBatch = generateBatch(batchCounter, BATCH_SIZE)
            currentLoss = trainingStep(imagesBatch, labelsBatch).item()

            averageLoss += currentLoss

            if (batchCounter + 1) % 100 == 0:
                averageLoss = averageLoss / 100.
                print(f"Average loss at batch {batchCounter}: {averageLoss:.2f}. Learning rate {LEARNING_RATE}")

                averageLoss = 0.
        
        accuracyHistory.append(calculateAccuracy())

    return accuracyHistory

        

# creates human readable characters for convenience
def emnist_to_char(label):
    if 0 <= label <= 9:
        return str(label)
    elif 10 <= label <= 35:
        return chr(label + 55)  # 10 maps to 'A' (ASCII 65)
    elif 36 <= label <= 61:
        return chr(label + 61)  # 36 maps to 'a' (ASCII 97)
    return '?'

def calculateAccuracy():
    cumulativeSuccessValue = 0.

    for i in range(len(testImages)):
        prediction = torch.softmax(model([testImages[i],], [[w1, b1], [w2, b2]])[0], dim=0) # make a prediction

        if int(torch.argmax(prediction).item()) is int(testLabels[i]):
            cumulativeSuccessValue += 1
    
    cumulativeSuccessValue /= len(testImages)
    return cumulativeSuccessValue


def main():
    print("Starting training loop now!")

    # Iterate over training data 7 times
    accuracyHistory = trainingLoop(7)
    # model is now trained!

    cumulativeSuccessValue = 0.

    for i in range(len(testImages)):
        prediction = torch.softmax(model([testImages[i],], [[w1, b1], [w2, b2]])[0], dim=0) # make a prediction

        if int(torch.argmax(prediction).item()) is int(testLabels[i]):
            cumulativeSuccessValue += 1

        if i % 100 == 0:
            print(f"Prediction: {emnist_to_char(torch.argmax(prediction))}, confidence: {(prediction.max().float()):.2f}, actual value: {emnist_to_char(testLabels[i])}")

    print(f"Success rate: {(cumulativeSuccessValue / len(testImages)) * 100}%")

    for i, v in enumerate(accuracyHistory):
        print(f"Epoch {i}, accuracy {v * 100.0}%")

main()