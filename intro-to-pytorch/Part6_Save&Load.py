# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:11:37 2020

@author: andre
"""

import torch
import fc_model

### SAVE trained networks

'''The parameters for PyTorch networks are stored in a model's state_dict. 
We can see the state dict contains the weight and bias matrices for each of our layers.'''

print("Our model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())

# For the commands above to work, it is necessary to have a model loaded in memory

# The simplest thing to do is simply save the state dict with torch.save.
# For example, we can save it to a file 'checkpoint.pth'.
torch.save(model.state_dict(), 'checkpoint.pth')

# or better
# save both the statedict and info about model architecture
checkpoint = {'input_size': 784,
              'output_size': 10,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')

### LOAD

# load the previously saved statedict with all trained paramters
state_dict = torch.load('checkpoint.pth')
print(state_dict.keys())

# Then you need as well to load the state dict in to the network itself 
model.load_state_dict(state_dict)

'''ACHTUNG:
     Loading the state dict works only if the model architecture is exactly 
     the same as the checkpoint architecture. 
     If I create a model with a different architecture, this fails. 
     This means we need to rebuild the model exactly as it was when trained. '''
# Try this
model = fc_model.Network(784, 10, [400, 200, 100])
# This will throw an error because the tensor sizes are wrong!
model.load_state_dict(state_dict)

'''Information about the model architecture needs to be saved in the checkpoint, 
along with the state dict. To do this, you build a dictionary with all the 
information you need to compeletely rebuild the model.'''

### LOAD with FUNCTION
'''The following is just a demo function. You need to build your own load
function for each of the model you want to learn. '''

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
