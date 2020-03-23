# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:00:39 2020

@author: andre
"""

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim
import helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


# Build a feed-forward network
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784,128)
        self.hidden2 = nn.Linear(128,64)
        self.output = nn.Linear(64,10)
        self.ReLU = nn.ReLU()
    
    def forward(self, x):
        x = self.hidden1(x)
        x = self.ReLU(x)
        x = self.hidden2(x)
        x = self.ReLU(x)
        x = self.output(x)
        
        return x        

model = Network()
model

# Define the loss, the criterion to compute the loss is by convention assigned 
# to the variable "criterion"
criterion = nn.CrossEntropyLoss()

# Get our data
images, labels = next(iter(trainloader))
# Flatten images
images = images.view(images.shape[0], -1)
''' images.shape[0] is gonna give us the batch size and -1 fills out the image 
with as many elements as it needs to keep the same total number of elements.
it gives back the same tensor as before but flattened in 1D.'''

# Forward pass, get our logits (scores. The logit is the inv fun of the sigmoid fun)
logits = model(images)
# NB: the logits tensor just calculated does not contain any probability belonging to the classes
# because to activation function is given to the output layer in this test network.
# That is why it only contains the scores (y_hat) calculated with the network.

# Calculate the loss with the logits and the labels
loss = criterion(logits, labels)

print(loss)

''' It's more convenient to build the model with a log-softmax output using 
nn.LogSoftmax or F.log_softmax (documentation), instead of a normal nn.Softmax as so far.
Then, you can get the actual probabilities by taking the exponential torch.exp(output). 
With a log-softmax output, you want to use the negative log likelihood loss, 
nn.NLLLoss (documentation) (in place of the crossEntropyloss).'''

### EXERCIZE
''' Build a model that returns the log-softmax as the output 
and calculate the loss using the negative log likelihood loss.'''
# TODO: Build a feed-forward network
model = nn.Sequential(nn.Linear(784,128),
                      nn.ReLU(),
                      nn.Linear(128,64),
                      nn.ReLU(),
                      nn.Linear(64,10),
                      nn.LogSoftmax(dim=1))

# TODO: Define the loss
criterion = nn.NLLLoss()

### Run this to check your work
# Get our data
images, labels = next(iter(trainloader))
# Flatten images
images = images.view(images.shape[0], -1)

# Forward pass, get our logits
logps = model(images)
# This logps contains the logarithm of the probabilities belonging to each class

# Calculate the loss with the logits and the labels
loss = criterion(logps, labels)

print(loss)

# you can simply extrapolate probabilities with:
ps = torch.exp(logps)


### AUTOGRAD
''' We can use it to calculate the gradients of all our parameters with respect to the loss.
Autograd works by keeping track of operations performed on tensors, t
hen going backwards through those operations, calculating gradients along the way.'''

x = torch.randn(2,2, requires_grad=True)
print(x)

y = x**2
print(y)
## grad_fn shows the function that generated this variable
print(y.grad_fn)

# another operations
z = y.mean()
print(z)

# check gradients for x (empty)
print(x.grad)
print(y.grad)

'''To calculate the gradients, you need to run the .backward method on a Variable, z for example. 
This will calculate the gradient for z with respect to x --> dz/dx
In fact, autograd is keeping trace on the operations performed and when asked
to calculate the gradient, it sees that the z depends on y which in turn depends
on x. Therefore the indipendent var is x. So the gradient would be: dz/dx
which for the chain rule is: dz/dx = dz/dy * dy/dx    '''

z.backward()     # this operation fills up the matrixes with the gradients

# In order to visualize them:
print(x.grad)
print(x/2) # derivative of x**2
'''Note that the derivative of the mean operation is just 1 so it cancels out'''
'''The gradient can be implicitly created only for scalar outputs, therefore
    .backward() can be applied only to scalars (that is why the passage through
    z and the calculation of the mean, because it returns a scalar.'''


### LOSS & AUTOGRAD TOGETHER
''' When we create a network with PyTorch, all of the parameters are initialized with requires_grad = True. 
This means that when we calculate the loss and call loss.backward(), the gradients for the parameters are calculated.
These gradients are used to update the weights with gradient descent.'''

# Let's see this with the previously defined NN and its loss:
print('Before backward pass: \n', model[0].weight.grad)

loss.backward()

print('After backward pass: \n', model[0].weight.grad)

### TRAINING
''' We first need an optimizer that we'll use to update the weights with the gradients.'''

# Optimizers require the parameters to optimize and a learning rate
# (typically optim.SGD or optim.Adam).
optimizer = optim.SGD(model.parameters(), lr=0.003)

# One training step
print('Initial weights - ', model[0].weight)

images, labels = next(iter(trainloader))            # laod an image with its label from the training set
images.resize_(64, 784)                             # resize the image

# Clear the gradients, do this because gradients are accumulated
optimizer.zero_grad()

# Forward pass, then backward pass, then update weights
output = model(images)                      # forward pass
loss = criterion(output, labels)            # calculate loss
loss.backward()                             # backward pass - calculate grads
print('Gradient -', model[0].weight.grad)

''' NB: When you do multiple backwards passes with the same parameters,
 the gradients are accumulated. 
 This means that you need to zero the gradients on each training pass 
 or you'll retain gradients from previous training batches.'''
 
# Take an update step and few the new weights
optimizer.step()
print('Updated weights - ', model[0].weight) 

# Training with loop 
''' here we're going to loop through trainloader to get our training batches'''

### EXERCIZE
'''Implement the training pass for our network. 
If you implemented it correctly, you should see the training loss drop with each epoch.'''

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # TODO: Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()        
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

### PREDICTIONS
# With the network trained, we can check out it's predictions
images, labels = next(iter(trainloader))

img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
helper.view_classify(img.view(1, 28, 28), ps)




