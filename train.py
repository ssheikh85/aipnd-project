'''Command_line application for training a neural network and predicting image type
train file parses command line args, calls build_model function to create model, sets-up criterion
and optimizer, trains model, prints out training loss, validation loss, test loss and test accuracy. 
Finally successfully trained model is saved in checkpoint path file
Author: Saeed Sheikh
Date: Oct 26 2018'''

import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from build_model import build_network
from cmd_parser import arg_parser

def main():

    #Function calls, 1st call is to the arg_parser
    options = arg_parser()

    #Builds model by calling build model function
    model = build_network(options.arch, options.hidden_units)


    #performs data augementation on datasets
    dataloader_train, dataloader_test, dataloader_valid = load_datasets(options.data_dir)

    #sets device to gpu, if gpu option is selected and CUDA is available, else use the cpu
    device = torch.device("cuda:0" if (torch.cuda.is_available() and options.gpu) else "cpu")

    #sets criterion and optimizer for training
    criterion, optimizer = set_criterion_optim(model, options.learnrate)

    #trains network
    train_network(model, dataloader_train, dataloader_valid, options.epochs, options.print_in, criterion,       
                  optimizer, device, options.gpu)

    #checks accuracy of network on test set, returns number correct, total, and percentage correct
    check_accuracy(model, device, dataloader_test)

    #saves trained model as a checkpoint
    save_checkpoint(model, options.epochs, class_to_idx, optimizer)


#Image Load and Transforms function
def load_datasets(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Defines transforms for the training, validation, and testing sets
    data_transforms_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                                transforms.RandomRotation(30),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    data_transforms_valid = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    data_transforms_test = transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

    image_datasets_train = datasets.ImageFolder(train_dir, transform = data_transforms_train)
    image_datasets_test = datasets.ImageFolder(test_dir, transform = data_transforms_test)
    image_datasets_valid = datasets.ImageFolder(valid_dir, transform = data_transforms_valid)

    #Using the image datasets and the trainforms, defines the dataloaders
    dataloader_train = torch.utils.data.DataLoader(image_datasets_train, batch_size = 64, shuffle = True)
    dataloader_test = torch.utils.data.DataLoader(image_datasets_test, batch_size = 64)
    dataloader_valid = torch.utils.data.DataLoader(image_datasets_valid, batch_size = 64)

    return dataloader_train, dataloader_test, dataloader_valid

#Function to set criterion and optimizer
def set_criterion_optim(model, learn_rate):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)

    return criterion, optimizer

#Validation function, used to compute test accuracy during training
def validation(model, dataloader_valid, criterion, device, gpu):
    test_loss = 0
    accuracy = 0
    for images, labels in dataloader_valid:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        if torch.cuda.is_available() and gpu:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        accuracy += equality.type(dtype).mean()
    
    return test_loss, accuracy


#function to train network
def train_network(model, dataloader_train, dataloader_valid, epochs_in, print_in, criterion, optimizer, device, gpu):
    epochs = epochs_in
    print_every = print_in
    steps = 0
    
    model.to(device)
    
    for e in range(epochs):
        model.train()
        running_loss = 0
        
        for ii, (inputs, labels) in enumerate(dataloader_train):
            
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
    
                model.eval()
                
                with torch.no_grad():
                    test_loss, accuracy = validation(model, dataloader_valid, criterion, device, gpu)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(dataloader_valid)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(dataloader_valid)))
                
                running_loss = 0
                
                model.train()

#Function to check accuracy on test dataset
def check_accuracy(model, device, dataloader_test):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader_test:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            _,predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(correct)
    print(total)
    print('Network accuracy on test image set = %d %%' % (100 * correct/total))

#function to save checkpoint
def save_checkpoint(model, epochs, class_to_idx, optimizer, image_datasets_train):
    model.class_to_idx = image_datasets_train.class_to_idx
    checkpoint = {'class_to_idx': image_datasets_train.class_to_idx,
              'epochs' : epochs,
              'optimizer' : optimizer.state_dict(),
              'state_dict': model.classifier.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')

# Call to main function to run the program
if __name__ == "__main__":
    main()