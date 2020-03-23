# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:42:52 2020

@author: andre

TRANSFER LEARNING
How to use pre-trained networks to solved challenging problems in computer vision. 
Specifically, you'll use networks trained on ImageNet available from torchvision.

ImageNet is a massive dataset with over 1 million labeled images in 1000 categories.
It's used to train deep neural networks using an architecture called convolutional layers.

Once trained, these models work astonishingly well as feature detectors for images they weren't trained on. 
Using a pre-trained network on images not in the training set is called transfer learning. 
Here we'll use transfer learning to train a network that can classify our cat and dog photos with near perfect accuracy.
"""

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
import helper

'''With torchvision.models you can download these pre-trained networks 
and use them in your applications. We'll include models in our imports now.

Most of the pretrained models require the input to be 224x224 images. 
Also, we'll need to match the normalization used when the models were trained.
Each color channel was normalized separately, the means are [0.485, 0.456, 0.406] 
and the standard deviations are [0.229, 0.224, 0.225].'''

data_dir = 'Cat_Dog_data'

if not (os.path.exists(data_dir + '/train') or os.path.exists(data_dir + '/test')) :
  os.mkdir(data_dir + '/train')
  os.mkdir(data_dir + '/test')
  
# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

# Display transformer images
# change this to the trainloader or testloader 
data_iter = iter(trainloader)

images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10,4), ncols=4)
for ii in range(4):
    ax = axes[ii]
    helper.imshow(images[ii], ax=ax, normalize=False)

###############################################################################
''' We can load in a model such as DenseNet. 
Let's print out the model architecture so we can see what's going on.'''

model = models.densenet121(pretrained=True)
model

'''
This model is built out of two main parts, the features and the classifier. 
The features part is a stack of convolutional layers and overall works as a feature detector 
that can be fed into a classifier. 
The classifier part is a single fully-connected layer (classifier): 
Linear(in_features=1024, out_features=1000). 
This layer was trained on the ImageNet dataset, so it won't work for our specific problem. 
hat means we need to replace the classifier, but the features will work perfectly on their own. 
In general, I think about pre-trained networks as amazingly good feature
 detectors that can be used as the input for simple feed-forward classifiers.'''
 

# Freeze feature parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# let's build the new classifier
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

# With our model built, we need to train the classifier. 
# However, now we're using a really deep neural network.

''' If you try to train this on a CPU like normal, it will take a long, long time. 
Instead, we're going to use the GPU to do the calculations. 
The linear algebra computations are done in parallel on the GPU leading to 
100x increased training speeds. It's also possible to train on multiple GPUs, 
further decreasing training time.

PyTorch uses CUDA to efficiently compute the forward and backwards passes on the GPU. 
In PyTorch, you move your model parameters and other tensors to the GPU memory using model.to('cuda'). 
You can move them back from the GPU with model.to('cpu')
 which you'll commonly do when you need to operate on the network output outside of PyTorch. 
As a demonstration of the increased speed, I'll compare how long it takes to 
perform a forward and backward pass with and without a GPU.'''

import time

for device in ['cpu', 'cuda']:

    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.to(device)

    for ii, (inputs, labels) in enumerate(trainloader):

        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        start = time.time()
 
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if ii==3:
            break
        
    print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")

''' The code above does the jon for ony 3 batches and compares performances between CPU-GPU'''

#### Training on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0000001)
# Then these methods will recursively go over all modules and convert their parameters and buffers to CUDA tensors:
model.to(device)

epochs = 5

train_losses = []

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Remember that you will have to send the inputs and targets at every step to the GPU too:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        
        log_ps = model.forward(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
                       
        train_losses.append(running_loss/len(trainloader))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.6f}.. ".format(running_loss/len(trainloader)))

plt.plot(train_losses, label='Training loss')
plt.legend(frameon=False)
plt.show('TrainError.png')
