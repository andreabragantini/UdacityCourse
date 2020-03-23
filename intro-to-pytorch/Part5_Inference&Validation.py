# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:56:16 2020

@author: andre
Now that you have a trained network, you can use it for making predictions. 
This is typically called inference, a term borrowed from statistics.
However, neural networks have a tendency to perform too well on the training data 
and aren't able to generalize to data that hasn't been seen before. 
This is called overfitting and it impairs inference performance. 
To test for overfitting while training, we measure the performance on data not
 in the training set called the validation set. We avoid overfitting through 
 regularization such as dropout while monitoring the validation performance during training.
"""

import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import helper
import matplotlib.pyplot as plt

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Build the network (solution from exarcize in Part4)
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

'''# Training pass
epochs = 5

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()

        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")
        
# Testing
dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[1]

# TODO: Calculate the class probabilities (softmax) for img
ps = torch.exp(model(img))

# Plot the image and probabilities
helper.view_classify(img, ps, version='Fashion')'''

'''With the probabilities ps, we can get the most likely class using the ps.topk method. 
This returns the $k$ highest values. Since we just want the most likely class, we can use ps.topk(1). 
This returns a tuple of the top-$k$ values and the top-$k$ indices. 
If the highest value is the fifth element, we'll get back 4 as the index.'''
dataiter = iter(testloader)
images, labels = dataiter.next()
ps = torch.exp(model(images))

top_p, top_class = ps.topk(1, dim=1)
# Look at the most likely classes for the first 10 examples
print(top_class[:10,:])

'''Now we can check if the predicted classes match the labels. 
This is simple to do by equating top_class and labels, but we have to be careful of the shapes. 
Here top_class is a 2D tensor with shape (64, 1) while labels is 1D with shape (64). 
To get the equality to work out the way we want, top_class and labels must have the same shape.'''

equals = top_class == labels.view(*top_class.shape)

'''Now we need to calculate the percentage of correct predictions.
 equals has binary values, either 0 or 1. This means that if we just sum up all 
 the values and divide by the number of values, we get the percentage of correct predictions.
 This is the same operation as taking the mean, so we can get the accuracy with a call to torch.mean.
 If only it was that simple. If you try torch.mean(equals), you'll get an error
RuntimeError: mean is not implemented for type torch.ByteTensor
This happens because equals has type torch.ByteTensor but torch.mean isn't implemented 
for tensors with that type. So we'll need to convert equals to a float tensor. 
Note that when we take torch.mean it returns a scalar tensor, 
to get the actual value as a float we'll need to do accuracy.item().'''

accuracy = torch.mean(equals.type(torch.FloatTensor))
print(f'Accuracy: {accuracy.item()*100}%')                  # Accuracy 84% !

### TRAINING AND VALIDATION ALL TOGETHER (3h)
epochs = 30
steps = 0

train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    else:
        test_loss = 0
        accuracy = 0
        
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for images, labels in testloader:
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.savefig('Overfitting.png')

############################### OVERFFITTING!!
''' It is clear how the network has suffered from overfitting since the 
training loss is decreasing with epochs but the validation loss (the one that matters)
starts back increasing after say 10 epochs.'''

# DROP-OUT
'''Dropout is a way to reduce the problem of overfitting.
It is a type of regularization applied during model training that drop randomly 
some perceptrons out of the network and train separately the remaining network.
Adding dropout in PyTorch is straightforward using the nn.Dropout module.

During training we want to use dropout to prevent overfitting,
but during inference we want to use the entire network. 
So, we need to turn off dropout during validation, testing,
and whenever we're using the network to make predictions. 
To do this, you use model.eval(). 
This sets the model to evaluation mode where the dropout probability is 0. 
You can turn dropout back on by setting the model to train mode with model.train().'''

### EXERCIZE: Add dropout to your model and train/test it with same dataset.

class ClassifierWithDP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        # Now with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

model = ClassifierWithDP()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Training pass
epochs = 10

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()

        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

# turn off gradients
with torch.no_grad():

    # set model to evaluation mode
    model.eval()
    
    validation_loss = 0
    # validation pass here
    for images, labels in testloader:
        log_ps = model(images)
        loss = criterion(log_ps, labels)

        validation_loss += loss.item()
    else:
        print(f"Validation loss: {validation_loss/len(testloader)}")
         
''' This simplified training pass trains the network over 10 epochs and
displays the traning loss after each epoch.
Looking at the solutions the network is trained already after 10 epochs
that is why i made this sipmplfied training.
The validation error is displayed only at the end'''


