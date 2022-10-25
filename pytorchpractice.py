

#%%
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
#%%
'''define how to transform the images for processing'''
transform = transforms.Compose([transforms.ToTensors,])
#%%