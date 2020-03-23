# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:24:48 2020

@author: andre

Now we're going to build a larger network to identify text in an image. 
Here we'll use the MNIST dataset which consists of greyscale handwritten digits
"""
import numpy as np
import torch
import helper
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

''' trainloader has a batch size of 64 and shuffle=True tells it to shuffle 
the dataset every time we start going through the data loader again.'''

dataiter = iter(trainloader)            # create an iterator to go through the dataset
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

# print one image from dataset
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');

### SIMPLE NETWORK with MATRIX MULTIPLICATION
''' let's try to build a simple network for this dataset using weight matrices and matrix multiplications.
Thinking about sizes, we need to convert the batch of images with shape (64, 1, 28, 28) to a have a shape 
of (64, 784), 784 is 28 times 28. 
This is typically called flattening, we flattened the 2D images into 1D vectors.
Here we need 10 output units, one for each digit.

Exercise: 
Flatten the batch of images images. 
Then build a multi-layer network with 784 input units, 256 hidden units, and 10 output units 
using random tensors for the weights and biases. 
For now, use a sigmoid activation for the hidden layer. 
Leave the output layer without an activation '''

n_input = 784
n_hidden = 256
n_output = 10

W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)

B1 = torch.randn(1, n_hidden)
B2 = torch.randn(1, n_output)

def activation(x):
    """ Sigmoid activation function 
    
        Arguments
        ---------
        x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))

# flatten the input images (change the shape)
inputs = images.view(64, 784)
# or
inputs = images.view(images.shape[0], -1)
print(inputs.shape)

# Define the NN layers
h = activation(torch.mm(inputs, W1) + B1)       # hidden layer
y = torch.mm(h, W2) + B2                        # output layer

# We use the softmax function for calcualting the output probability in MULTI-CLASS problems
def softmax(x):
    output = torch.randn_like(x)
    for i in range(len(x)):
        for j in range(len(x[i])):   
            num = torch.exp(x[i][j])
            den = sum(torch.exp(x[i]))
            output[i][j] = num/den
    return output
# or
def softmax(y):
    return torch.exp(y)/torch.sum(torch.exp(y), dim=1).view(-1, 1)
    
# Here, out should be the output of the network in the previous excercise with shape (64,10)
probabilities = softmax(y)

# Does it have the right shape? Should be (64, 10)
print(probabilities.shape)
# Does it sum to 1?
print(probabilities.sum(dim=1))

''' TORCH.SUM() HOW IT WORKS:
torch.sum(y)            gives the sum of all elements in the tensor    
torch.sum(y, dim=0)     gives the sum across the rows of the tensor
torch.sum(y, dim=1)     gives the sum across the columns of the tensor '''

### SIMPLE NETWORK with PYTORCH
''' We initiliaze our class taking features from its "parent" class in nn.Module
For this we need "super().__init__()"
In the initialization method __init__(self) we define the architecture of our NN
and the different layers. (sparse order)
Load the activation functions from package (Setting dim=1 in nn.Softmax(dim=1) calculates softmax across the columns.)
Then we define the forward method that describes what the NN does in the forward pass.
Here, the operations between layers prviously defined have to be listed in order,
with same name of input/output variable (x, in this case).
The module automatically creates the weight and bias tensors which we'll use in the forward method.
'''
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
        # Load sigmoid activation and softmax output function
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x

# Create the network object and look at it's text representation
model = Network()
model

''' You can define the network somewhat more concisely and clearly using the 
torch.nn.functional module. This is the most common way you'll see networks 
defined as many operations are simple element-wise functions. 
We normally import this module as F, import torch.nn.functional as F.'''

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)
        
        return x

model = Network()
model

### EXERCIZE
'''Exercise: Create a network with 784 input units, a hidden layer with 128 units and a ReLU activation, 
then a hidden layer with 64 units and a ReLU activation, and finally an output layer with a softmax activation as shown above. 
You can use a ReLU activation with the nn.ReLU module or F.relu function.'''

class NetworkEx(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784,128)
        self.hidden2 = nn.Linear(128,64)
        self.output = nn.Linear(64,10)
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.hidden1(x)
        x = self.ReLU(x)
        x = self.hidden2(x)
        x = self.ReLU(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x

model = NetworkEx()
model

# Initializa weights and biases
'''The weights and such are automatically initialized for you, 
but it's possible to customize how they are initialized.
The weights and biases are tensors attached to the layer you defined, 
you can get them with model.fc1.weight for instance.'''

print(model.hidden1.weight)
print(model.hidden1.bias)

''' Notice that model.hidden1.bias is not a tensor itself, to access the tensor
you need to use model.hidden1.bias.data'''
# Set biases to all zeros
model.hidden1.bias.data.fill_(0)
# sample from random normal with standard dev = 0.01
model.hidden1.weight.data.normal_(std=0.01)

###  FORWARD PASS
# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels) 
images.resize_(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784) to automatically get batch size

# Forward pass through the network
img_idx = 0                                         # image ID            
ps = model.forward(images[img_idx,:])               # output tensor with class probabilites

img = images[img_idx]                               # this select a digit in the dataset
helper.view_classify(img.view(1, 28, 28), ps)       # command for nice display and comparison

''' As you can see above, our network has basically no idea what this digit is. 
It's because we haven't trained it yet, all the weights are random!'''

### NN.SEQUENTIAL
'''convenient way to build networks like this where a tensor is passed sequentially
 through operations, nn.Sequential (documentation).'''
 
# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(model)

# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0,:])
helper.view_classify(images[0].view(1, 28, 28), ps)

# Operations on the network
print(model[0])     # this returns the first Linear Operation (between input,hidden1 layers)
model[0].weight     # this returns weights of that operation

'''You can also pass in an OrderedDict to name the individual layers and operations, 
instead of using incremental integers. Note that dictionary keys must be unique, 
so each operation must have a different name.'''

model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(hidden_sizes[1], output_size)),
                      ('softmax', nn.Softmax(dim=1))]))
model

# Now you can access the layers and operations in both ways
print(model[0])
print(model.fc1)

