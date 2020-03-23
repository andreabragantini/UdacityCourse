# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 19:59:52 2020

@author: andre
"""

import torch
import numpy as np

def activation(x):
    """ Sigmoid activation function 
    
        Arguments
        ---------
        x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))

### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 5 random normal variables
features = torch.randn((1, 5))
# True weights for our data, random normal variables again
weights = torch.randn_like(features)
# and a true bias term
bias = torch.randn((1, 1))

# Calculate the output of the network with input features, weights, and bias
''' NB: Mind that features*weights is a dot product (scalar product) therefore
gives a scalar as result. The normal function sum() cannot be used. '''
y = activation((features * weights).sum() + bias)
y = activation(torch.sum(features * weights) + bias)

''' You can do the multiplication and sum in the same operation using a matrix multiplication.
For this we can use torch.mm() or torch.matmul()
Remember that for matrix multiplications, the number of columns in the first
tensor must equal to the number of rows in the second column.
Both features and weights have the same shape, (1, 5). 
This means we need to change the shape of weights to get the matrix multiplication to work.
There are a few options here: weights.reshape(), weights.resize_(), and weights.view().

weights.reshape(a, b) will return a new tensor with the same data as weights 
    with size (a, b) sometimes, and sometimes a clone, as in it copies the data to another part of memory.
    
weights.resize_(a, b) returns the same tensor with a different shape. 
    However, if the new shape results in fewer elements than the original tensor, 
    some elements will be removed from the tensor (but not from memory). 
    If the new shape results in more elements than the original tensor, 
    new elements will be uninitialized in memory. Here I should note that the 
    underscore at the end of the method denotes that this method is performed in-place. 
    Here is a great forum thread to read more about in-place operations in PyTorch.

weights.view(a, b) will return a new tensor with the same data as weights with size (a, b).'''

weightsT = weights.view(5, 1)

# Calculate the output of the network using matrix multiplication
y = activation(torch.mm(features, weightsT) + bias)
y = activation(torch.matmul(features, weightsT) + bias)

### MULTI LAYER  PERCEPTRONS

# Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 3 random normal variables
features = torch.randn((1, 3))

# Define the size of each layer in our network
n_input = features.shape[1]     # Number of input units, must match number of input features
n_hidden = 2                    # Number of hidden units 
n_output = 1                    # Number of output units

# Weights for input layer to hidden layer
W1 = torch.randn(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

# Calculate the output for this multi-layer network using the weights W1 & W2, and the biases, B1 & B2.
y = activation(torch.mm(activation(torch.mm(features,W1) + B1),W2) + B2)
# or
h = activation(torch.mm(features,W1) + B1)
y = activation(torch.mm(h,W2) + B2)
print(y)

''' hyperparameter = n# of hidden units '''

### FROM NUMPY TO TORCH AND BACK
# to create a tensor from a Numpy array, use torch.from_numpy(). 
# To convert a tensor to a Numpy array, use the .numpy() method.

a = np.random.rand(4,3)
a

b = torch.from_numpy(a)
b

''' The memory is shared between the Numpy array and Torch tensor, 
so if you change the values in-place of one object, the other will change as well.'''

# Multiply PyTorch Tensor by 2, in place
b.mul_(2)
# Numpy array matches new values from Tensor
a
