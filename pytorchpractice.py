

#%%
from re import M
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
#define how to transform the images for processing
transform = transforms.Compose([transforms.ToTensors,\
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
#just use a really simple assumption that the mean and std dev
#of each channel is 0.5

#then we pull the training and test sets in
#using the cifar10 dataset, we will store in in a local
#folder called 'data'
cifartrain = torchvision.datasets.CIFAR10(root = "./data",\
    train = True, download = True, transform = transform)
cifartest = torchvision.datasets.CIFAR10(root = "./data",\
    train = False, download=True, transform = transform)
#and make a list of the images in the dataset
classes = ("airplane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
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

class CIFARNet(nn.Module):
    def __init__(self, n_classes = 10): #make it somewhat reusable by allowing user to define n_classes
        super(CIFARNet, self).__init()
        self.convo = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=16, out_channel = 32, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride = 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.linear = nn.Sequential(

        )

        
            
            
        
        

#%%