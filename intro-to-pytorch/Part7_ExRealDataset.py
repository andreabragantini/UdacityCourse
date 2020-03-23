# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 19:00:45 2020

@author: andre
We'll use this dataset to train a neural network that can differentiate between cats and dogs.
"""

import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch import nn
from torch import optim
import helper
import os

# Load image data
#dataset = datasets.ImageFolder('path/to/data', transform=transform)

''' ImageFolder expects each class to have its own directory for the images.
The images are then labeled with the class taken from the directory name.'''

# TRANSFORMS
'''When you load in the data with ImageFolder, you'll need to define some transforms. 
For example, the images are different sizes but we'll need them to all be the same size for training. 
You can either resize them with transforms.Resize() or crop with transforms.CenterCrop(), 
transforms.RandomResizedCrop(), etc. 
We'll also need to convert the images to PyTorch tensors with transforms.ToTensor(). 
Typically you'll combine these transforms into a pipeline with transforms.Compose(), 
which accepts a list of transforms and runs them in sequence. 
It looks something like this to scale, then crop, then convert to a tensor,'''

#transform = transforms.Compose([transforms.Resize(255),
#                                 transforms.CenterCrop(224),
#                                 transforms.ToTensor()])

# DATA LOADERS
'''With the ImageFolder loaded, you have to pass it to a DataLoader. 
The DataLoader takes a dataset (such as you would get from ImageFolder) 
and returns batches of images and the corresponding labels. 
You can set various parameters like the batch size and if the data is shuffled after each epoch.'''

#dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

'''Here dataloader is a generator. 
To get data out of it, you need to loop through it or convert it to an iterator and call next().'''

# Looping through it, get a batch on each loop 
#for images, labels in dataloader:
#    pass

# Get one batch
#images, labels = next(iter(dataloader))

### EXERCIZE
'''Load images from the Cat_Dog_data/train folder, define a few transforms, then build the dataloader.'''

data_dir = 'Cat_Dog_data'

if not os.path.exists(data_dir):
  os.mkdir(data_dir)

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


# Run this to test your data loader
images, labels = next(iter(dataloader))
helper.imshow(images[0], normalize=False)
   
### DATA AUGMENTATION
# To randomly rotate, scale and crop, then flip your images,
# you would define your transforms like this:

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])])

'''DATA NORMALIZATION
You'll also typically want to normalize images with transforms.Normalize. 
You pass in a list of means and list of standard deviations, 
then the color channels are normalized like so

input[channel] = (input[channel] - mean[channel]) / std[channel]

Subtracting mean centers the data around zero and dividing by std squishes 
the values to be between -1 and 1. Normalizing helps keep the network weights 
near zero which in turn makes backpropagation more stable. Without normalization, 
networks will tend to fail to learn.'''

data_dir = 'Cat_Dog_data'

if not (os.path.exists(data_dir + '/train') or os.path.exists(data_dir + '/test')) :
  os.mkdir(data_dir + '/train')
  os.mkdir(data_dir + '/test')
 
# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()]) 

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])


# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

# Display transformer images
# change this to the trainloader or testloader 
data_iter = iter(trainloader)

images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10,4), ncols=4)
for ii in range(4):
    ax = axes[ii]
    helper.imshow(images[ii], ax=ax, normalize=False)

'''At this point you should be able to load data for training and testing. 
Now, you should try building a network that can classify cats vs dogs. 
This is quite a bit more complicated than before with the MNIST and Fashion-MNIST datasets. 
To be honest, you probably won't get it to work with a fully-connected network, no matter how deep. 
These images have three color channels and at a higher resolution 
(so far you've seen 28x28 images which are tiny).'''

### EXERCIZE
# Try to build the NN model which classifies cats and dogs

class ClassifierCatDogs(nn.Module):
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

model = ClassifierCatDogs()
model

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

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
plt.savefig('TestVSTrainError.png')
