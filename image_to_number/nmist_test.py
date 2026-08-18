import torch
import torchvision ## Contains some utilities for working with the image data
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

mnist = MNIST(root = 'data/', download = True, train = True, transform = transforms.ToTensor())
images = []
labels = []
for image, label in mnist:
    images.append(image)
    labels.append(label)

print((len(images)))