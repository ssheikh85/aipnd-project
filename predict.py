'''Command_line application for training a neural network and predicting image type
predict file loads saved model checkpoint, performs image processing on single image,
uses this image to make precictions about floor type
Author: Saeed Sheikh
Date: Oct 26 2018'''

import json
import argparse
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from process_image import process_image
from cmd_parser import arg_parser
from build_model import build_network

def main():

    #Function calls, 1st call is to the arg_parser
    options = arg_parser()

    #Loads model from directory where model is saved
    model_out = load_model(options.arch, options.hidden_units, options.checkpoint_file)

    #Calls the predict function, which prints out topk probabilities and flower names
    predict(options.image_file, model_out, options.topk, options.gpu, options.category_names)

#Function to load trained model
def load_model(model_name, hidden_units, filepath):
    model = build_network(model_name, hidden_units)
    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = (checkpoint['class_to_idx'])
    #epochs = (checkpoint['epochs'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    return model

#Code used to predict image
def predict(img_filepath, model_out, topk_in, gpu, cat_names):
    
    model = model_out
    
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)

    image_path = img_filepath

    topk = topk_in

    model.eval()
    
    image = process_image(image_path)
    im_tensor = torch.from_numpy(image)
    
    if torch.cuda.is_available() and gpu:
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
   
    for idx in classes:
        flower_names.append(cat_to_name[idx])
    count = 0
    for x, y in zip(top_5_out, flower_names):
        count = count + 1
        #String formatting code adapted from code found on StackOverflow
        print(str(count)+') Flower Name:', '{}'.format(y).rjust(18), 'Probability: {:>10.2f}'.format((x*100)).rjust(28)+'%')
        
    
# Call to main function to run the program
if __name__ == "__main__":
    main()