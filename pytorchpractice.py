

#%%
import torch
import torch.nn as nn
import torchvision
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
#%%
#set the device to "gpu" this script was written on a 
# 'macOS-12.6-arm64-arm-64bit' platform with
# torch version '1.12.1'. The code to move to gpu might be different
#depending on your machine, or may not be available.

device = torch.device("mps") #I believe this is "cuda" for nvidia machines

#%%
'''define how to transform the images for processing'''
transform = transforms.Compose([transforms.ToTensors,\
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
#just use a really simple assumption that the mean and std dev
#of each channel is 0.5

#then we pull the training and test sets in
#using the cifar10 dataset, we will store in in a local
#folder called 'data'
cifartrain = torchvision.datasets.CIFAR10(root = "./data",\
    train = True, download = True, transform = transform)
cifartest = torchvision.datasets.CIFAR10(root = "./data"\,\
    train = False, download=True, transform = transform)
#%%
#then we build the data loaders to get the data 
#into the pytorch model

trainload = DataLoader(cifartrain, batch_size = 128,\
    shuffle = True, num_workers=4)
testloader = DataLoader(cifartest, batch_size = 128,\
    shuffle = False, num_workers=4)
#load in 128 at a time, shuffle the training set, but not
#the testing set

#%%
#we will then start to put together a modle
#going for simple here, so building a feed forward model utilizing
#the sequetial module offered in pytorch

#%%