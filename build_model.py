'''Command_line application for training a neural network and predicting image type
Build_model file defines function to build the neural network
returns neural network
Author: Saeed Sheikh
Date: Oct 26 2018'''

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

def build_network(model_name, hidden_sizes):

    vgg19 = models.vgg19(pretrained = True)
    densenet161 = models.densenet161(pretrained = True)
    alexnet = models.alexnet(pretrained = True)

    models_dict = {'densenet': densenet161, 'alexnet': alexnet, 'vgg': vgg19}

    input_size = 0
    model = models_dict[model_name]

    if model_name == 'vgg':
        input_size = 25088
    elif model_name == 'densenet':
        input_size = 2208
    elif model_name == 'alexnet':
        input_size = 9216
    else:
        print('Unsupported model used or model architecture enter incorrectly')
    
    #Freezing parameters for features
    for param in model.parameters():
        param.requires_grad = False

    #Building classifier network using OrderedDict  
    from collections import OrderedDict

    input_layer = input_size
    hidden_layers = hidden_sizes
    #Output size is set to 102 as model is beign trained to classify flower type out of 102 possible types
    output_size = 102

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_layer, hidden_layers)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_layers, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))


    model.classifier = classifier

    return model


   
    

