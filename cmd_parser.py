'''Command_line application for training a neural network and predicting image type
cmd_parser file, calls argparse library to use for parsing command line arguments
returns arg parser
Author: Saeed Sheikh
Date: Oct 26 2018'''

import argparse


def arg_parser():
    
    
    # Creates parser 
    parser = argparse.ArgumentParser()

    # Creates command line arguments for options to select directory where data is stored, directory for checkpoint save,
    # model architecture to be used, use of JSON file with flower names, enable gpu, hyperparameters for building and training model: such
    # as learnrate, epochs, hidden_units, and an option to print out topk probabilities for image prediction
    
    parser.add_argument('--data_dir', type=str, default='/home/workspace/aipnd-project/flowers/', 
                        help='path to folder with training, test and validation flower images')

    parser.add_argument('--save_dir', type=str, default='/home/workspace/aipnd-project',
                        help='path to save location for checkpoints')
    
    parser.add_argument('--checkpoint_file', type=str, default='/home/workspace/aipnd-project/checkpoint.pth',
                        help='Image file for prediction')
        
    parser.add_argument('--image_file', type=str, default='/home/workspace/aipnd-project/predict_img.jpg',
                        help='Image file for prediction')

    parser.add_argument('--arch', type=str, default='vgg', 
                        help='chosen pretrained model')

    parser.add_argument('--gpu', type=bool, default=False, 
                        help='Enable use of CUDA based gpu')

    parser.add_argument('--learnrate', type=float, default=0.001,
                        help='Learning rate for training')

    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs to train for')

    parser.add_argument('--hidden_units', type=int, default=1000,
                        help='Number of units for hidden layers classifier')
    
    parser.add_argument('--print_in', type=int, default=40,
                        help='Print step for training')
    
    parser.add_argument('--topk', type=int, default=5,
                        help='topk probabilities for image prediction, where k is an int')
    
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='A JSON file with flower names corresponding to indices in training dataset')

    # returns parsed argument collection
    return parser.parse_args()