# AI Programming with Python Project
Project Completed by Saeed Sheikh on Oct. 27 2018

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

Part 1: Image Classifier Project.ipynb(Jupyter Notebook)

This part of the project displays has all the code in a jupyter notebook,
Code cells include code that Imports packages and modules, performs data augmentation, and data loading, loads the pre-trained model, builds the classifier network, a function for validation, a function for training the network, a function for checking the network accuracy on the test image set, code to save the network as a checkpoint, loading the saved checkpoint, a function to pre-process a single inputted image, and a function to predict the image type, and code to display the image and top 5 probabilities

Datasets from the Udacity workspace were used, a single image from the dataset was used for preprocessing and predict

Part 2: Command Line application

Two main files are train.py and predict.py

Helper modules are cmd_parser.py, build_model.py, and process-image.py

cmd_parser.py contains function arg_parser which is used to parse command line arguments,
Available arguments to parse are as follows, and cover arguments that can be used for both train.py and predict.py
     
    '--data_dir', type=str, default='/home/workspace/aipnd-project/flowers/', 
                        help='path to folder with training, test and validation flower images')

    '--save_dir', type=str, default='/home/workspace/aipnd-project',
                        help='path to save location for checkpoints')
    
    '--checkpoint_file', type=str, default='/home/workspace/aipnd-project/checkpoint.pth',
                        help='Image file for prediction')
        
    '--image_file', type=str, default='/home/workspace/aipnd-project/predict_img.jpg',
                        help='Image file for prediction')

    '--arch', type=str, default='vgg', 
                        help='chosen pretrained model')

    '--gpu', type=bool, default=False, 
                        help='Enable use of CUDA based gpu')

    '--learnrate', type=float, default=0.001,
                        help='Learning rate for training')

    '--epochs', type=int, default=3,
                        help='Number of epochs to train for')

    '--hidden_units', type=int, default=1000,
                        help='Number of units for hidden layers classifier')
    
    '--print_in', type=int, default=40,
                        help='Print step for training')
    
    '--topk', type=int, default=5,
                        help='topk probabilities for image prediction, where k is an int')
    
    '--category_names', type=str, default='cat_to_name.json',
                        help='A JSON file with flower names corresponding to indices in training dataset')
 
build_model.py: Takes in the command-line argument --arch, default is the VGG pretrained network, and --hidden_units, default = 1000, and returns a neural network for training and preduction
 
train.py: creates a new command line parser, builds the model, performs data augmentation, loads the datasets, and sets the criterion and optimizer, trains the network, checks the accuracy of the network using the test image dataset, and saves the trained network
to run in the command line use:
    
    python train.py
    
to use with gpu for example:

    python train.py --gpu=True

You may want to use non-default values for the above command line arguments, they are currently set to be used with the default values listed above, and for use in the Udacity worskpace

process_iamge.py: takes in a single image from a directory and returns a numpy array of the image preprocessed to be used for prediction

predict.py: Loads and rebuilds the trained network and uses it to predict the topk proabilities. Predict.py prints the flower names, and top k probabilities

to run in the command line use:
    
    python predict.py
    
to use with gpu for example:

    python predict.py --gpu=True


    
