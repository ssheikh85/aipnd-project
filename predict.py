'''Command_line application for training a neural network and predicting image type
predict file loads saved model checkpoint, performs image processing on single image,
uses this image to make precictions about floor type
Author: Saeed Sheikh
Date: Oct 26 2018'''

import cmd_parser
import process_iamge

import json
import argparse
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def main():

    #Function calls, 1st call is to the arg_parser
    options = arg_parser()

    #Loads model from directory where model is saved
    model = load_checkpoint(options.save_dir)

    #Calls the predict function, which prints out topk probabilities and flower names
    predict(options.save_dir, model, options.topk, options.gpu)

#Function to load trained model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model.class_to_idx = (checkpoint['class_to_idx'])
    epochs = (checkpoint['epochs'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.classifier.load_state_dict(checkpoint['state_dict'])
    
    return model

#Code used to predict image
def predict(options.save_dir, model, options.topk, options.gpu):
    
    with open(options.cat_to_name, 'r') as f:
        cat_to_name = json.load(f)

    image_path = options.save_dir

    topk = options.topk

    model.eval()
    
    image = process_image(image_path)
    im_tensor = torch.from_numpy(image)
    
    if torch.cuda.is_available() and options.gpu:
            dtype = torch.cuda.FloatTensor
    else:
            dtype = torch.FloatTensor
    
    im_tensor = im_tensor.type(dtype)
    im_tensor.unsqueeze_(0)
    model.type(dtype)
    
    with torch.no_grad():
        output = model(im_tensor)
       
    all_probs = torch.exp(output)
    
    top_5_probs, indices = torch.topk(all_probs, topk, sorted=True)
    
    #Invets class to index dictionary
    #Coded from examples found during online search and from forums such as Stack Overflow
    invert_dict = {value: key for key, value in model.class_to_idx.items()}
    
    top_5_probs = top_5_probs.cpu().numpy()
    indices = indices.cpu().numpy()
    
    idx_array = [] 
    classes = []
    top_5_out = []
    flower_names = []

    print("Top {} highest probabilities and flower types".format(topk))

    for i in range(topk):
        idx_array.append(indices[0][i])
        classes.append(invert_dict[idx_array[i]])
        top_5_out.append(top_5_probs[0][i])
        flower_names.append(cat_to_name[i])

        print("{}".format(i+1),
              "Probability {:.3f}.. ".format(top_5_out[i]*100),
              "Flower Name: {:.3f}.. ".format(flower_names[i]))
                      
    
    
# Call to main function to run the program
if __name__ == "__main__":
    main()