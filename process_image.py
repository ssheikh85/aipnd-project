'''Command_line application for training a neural network and predicting image type
Process_image file takes in file image, resizes it, crops it to 224x224, and normalizes the image for
use in prediction returns an Numpy array
Author: Saeed Sheikh
Date: Oct 26 2018'''

import numpy as np
from PIL import Image


def process_image(image):
    
    # DONE: Processes a PIL image for use in a PyTorch model
    im = Image.open(image)
    
    horiz, vert = im.size
    new_vert = 256
    new_dims = (horiz * new_vert)/vert, new_vert
    
    im.thumbnail(new_dims)
    
    resize_horiz, resize_vert = im.size
 
    #These five lines of code were adapted from code found StackOverflow
    crop_dim1, crop_dim2 = 224, 224
    left = (resize_horiz - crop_dim1)//2
    upper = (resize_vert - crop_dim2)//2
    right = (resize_horiz + crop_dim1)//2
    bottom = (resize_vert + crop_dim1)//2
    
    cropped_im = im.crop((left, upper, right, bottom))
    
    np_image = np.array(cropped_im)
    
    #To handle PNG images with 4 color channels
    np_image = np_image[...,:3]
    
    np_image = np_image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image - mean)/std
    
    np_image = np_image.transpose((2, 1, 0))
    
    return np_image