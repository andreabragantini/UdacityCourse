# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:54:49 2020

@author: andre
Fashion-MNIST is a set of 28x28 greyscale images of clothes. It's more complex than MNIST.
Each image is 28x28 which is a total of 784 pixels, and there are 10 classes.
"""

import torch
from torch import nn
from torchvision import datasets, transforms
from torch import optim
import helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Look at one image
image, label = next(iter(trainloader))
helper.imshow(image[0,:]);

### Building your network
'''You should include at least one hidden layer. 
We suggest you use ReLU activations for the layers and to return the logits or 
log-softmax from the forward pass.'''

class NetworkMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784,200)
        self.output = nn.Linear(200,10)
        self.ReLU = nn.ReLU()
        self.LogSoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self,x):
        x = self.hidden(x)
        x = self.ReLU(x)
        x = self.output(x)
        x = self.LogSoftmax(x)
    
        return x

network = NetworkMNIST()
network

### Train the network
criterion = nn.NLLLoss()
''' Since we are using LogSoftmax as the output of our model make sense to use 
the Negative Log Likelihood Loss criterion.'''
optimizer = optim.SGD(network.parameters(), lr=0.01)

''' By adjusting the hyperparameters (hidden units, learning rate, etc), 
you should be able to get the training loss below 0.4.'''

# Training pass
epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # TODO: Training pass
        optimizer.zero_grad()
        
        output = network(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()  
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")


#### Test out the network!

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]

# Convert 2D image to 1D vector
img = img.resize_(1, 784)

# TODO: Calculate the class probabilities (softmax) for img
logps = network(img)
ps = torch.exp(logps)

# Plot the image and probabilities
helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')




