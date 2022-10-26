

#%%

from turtle import color
import torch
import torch.nn as nn
import torchvision
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt

#%%
#set the device to "gpu" this script was written on a 
#'macOS-12.6-arm64-arm-64bit' platform with
#torch version '1.12.1'. The code to move to gpu might be different
#depending on your machine, or may not be available.
#python version 3.10.6
mps_on = torch.has_mps

if mps_on:
    devicemps = torch.device("mps")
#I believe this is "cuda" for nvidia machines
#%%
#define accuracy function

def accuracy(outputs, labels):
    _, pred = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(pred == labels).item()/len(pred))


#%%
#estimate mean and standard deviation of images from test data
cifartraingetmean = torchvision.datasets.CIFAR10(root = "./data",\
    train = False, download=True, transform=transforms.ToTensor())
imgs = [item[0] for item in cifartraingetmean]
imgs = torch.stack(imgs, dim = 0).numpy()
mean_red = imgs[:,0,:,:].mean()
std_red = imgs[:,0,:,:].std()
mean_green = imgs[:,1,:,:].mean()
std_green = imgs[:,1,:,:].std()
mean_blue = imgs[:,2,:,:].mean()
std_blue = imgs[:,2,:,:].std()
print(mean_red,mean_green, mean_blue)

#%%
#define how to transform the images for processing
transform = transforms.Compose([transforms.ToTensor(),\
    transforms.Normalize((mean_red,mean_green,mean_blue), (std_red,std_green,std_blue))])


#then we pull the training and test sets in
#using the cifar10 dataset, we will store in in a local
#folder called 'data'
cifartrain = torchvision.datasets.CIFAR10(root = "./data",\
    train = True, download = True, transform = transform)
cifartest = torchvision.datasets.CIFAR10(root = "./data",\
    train = False, download=True, transform = transform)

#and make a list of the images in the dataset
classes = cifartrain.classes
#%%
#then we build the data loaders to get the data 
#into the pytorch model
#we will manually build a k-fold CV from the training set
#so no just put 
indxs = np.arange(len(cifartrain))
np.random.shuffle(indxs)
split = int(np.floor(len(cifartrain) * 0.1)) #10% set aside for validation
train_idx, val_idx = indxs[split:], indxs[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(val_idx)
#%%
trainload = DataLoader(cifartrain, batch_size = 128,\
    sampler=train_sampler, num_workers=0, pin_memory=True)

valload = DataLoader(cifartrain, batch_size = 128,\
    sampler=valid_sampler, num_workers=0, pin_memory=True)

testloader = DataLoader(cifartest, batch_size = 128,\
    num_workers=0, pin_memory=True)


#load in 128 at a time
#workers set to 0 because it freezes otherwise and I cannot figure out why
#seems to be a unique problem to mps GPU usage


#%%
#plot some of the images
def showimg(img):
    img = img/2 + 0.5
    plt.imshow(np.transpose(img, (1,2,0)))
def get_show_image(d_loader = trainload, n_show = 20):
    set_row = n_show//10 if n_show%10 == 0 else n_show//10 + 1
    diter = iter(d_loader)
    images, label = diter.next()
    images = images.numpy()
    fig = plt.figure(figsize=(25,4))
    for idx in np.arange(n_show):
        ax = fig.add_subplot(set_row, 10, idx + 1, xticks = [], yticks = [])
        showimg(images[idx])
        ax.set_title(classes[label[idx]])


#%%
#we will then start to put together a model
#going for simple here, so building a feed forward model utilizing
#the sequetial module offered in pytorch
#model is loosely based on AlexNET

class CIFARNet(nn.Module):
    def __init__(self, n_classes = 10): #make it somewhat reusable by allowing user to define n_classes
        super(CIFARNet, self).__init__()
        self.convo = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=16, out_channels = 32, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride = 2), #size 32 x 16 x 16
            nn.Conv2d(in_channels=32, out_channels = 64, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels = 128, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2), #size 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1), #size 256 x 8 x 8
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6,6)) # out: 256 x 6 x 6
        self.linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 1024),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace = True),
            nn.Linear(1024,n_classes)            
        )
    def forward(self, x):
        x = self.convo(x)
        x = self.avgpool(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.linear(x)
        return x
#%%
#set up hyperparameters and optimizer
model = CIFARNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 3e-3, weight_decay=0.0001)
if mps_on: model.to(devicemps)
#%%
"""make sure the model is outputting the expected results"""
for images, labels in trainload:
    if mps_on: 
            images = images.to(devicemps)
            labels = labels.to(devicemps)
    print("images.shape", images.shape)
    out = model(images)
    print("out.shape", out.shape)
    print("out[0]", out[0])
    break
#%%
start_time = time.time()
train_epoch_loss, val_epoch_loss, train_accuracy, pred_accuracy = [],[],[],[]
epoch_val_acc, epoch_train_acc = [],[]
best_val_loss, best_epoch = np.inf, 0
n_epochs = 10
for epoch in range(n_epochs):
    epoch_start_time = time.time()
    model.train()
    train_loss = []
    for batch_idx, (feats, targets) in enumerate(trainload):
        if mps_on: 
            feats = feats.to(devicemps)
            targets = targets.to(devicemps)
        optimizer.zero_grad()
        output = model(feats)
        loss = criterion(output, targets)
        train_loss.append(loss)
        train_accuracy.append(accuracy(output, targets))
        loss.backward()
        optimizer.step()
    print("Epoch: {} train runtime: {:.3f} minutes".format(epoch + 1,(time.time()-epoch_start_time)/60))
    epoch_train_acc.append(torch.stack(train_accuracy).mean().item())
    train_epoch_loss.append(torch.stack(train_loss).mean().item())

    val_start_time = time.time()
    model.eval()
    val_loss = []
    for val_idx, (feats, targets)  in enumerate(valload):
        if mps_on:
            feats, targets = feats.to(devicemps), targets.to(devicemps)
        output = model(feats)
        loss = criterion(output, targets)
        val_loss.append(loss)
        temp = accuracy(output, targets)
        pred_accuracy.append(temp)
        

    print("Epoch: {} validation runtime: {:.3f} minutes".format(epoch + 1,(time.time()-val_start_time)/60))
    epoch_val_acc.append(torch.stack(pred_accuracy).mean().item())
    val_epoch_loss.append(torch.stack(val_loss).mean().item())
    if val_epoch_loss[-1] < best_val_loss:
        best_val_loss = val_epoch_loss[-1]
        best_epoch = epoch
        torch.save(model.state_dict(), "nnmodel_cifar10.pt")

    


print("Total time: {:.3f} minutes".format((time.time()-start_time)/60))
#%%
#plot training and validation epoch 
fig, ax = plt.subplots()
ax.plot(train_epoch_loss, "-bx")
ax.plot(val_epoch_loss, "-rx")
ax.set_title("Training loss and validation loss")
ax.legend(["Training", "Validation"])
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ymin,ymax = plt.ylim()
ax.axvline(x = best_epoch, color = 'k')
ax.annotate("Lowest validation loss", xy = (best_epoch,ymax/2),\
    textcoords='offset points')
plt.show()
fig.savefig("nnmodel_loss.png")
#%%
figa, axa = plt.subplots()
axa.plot(train_accuracy)
#axa.plot(pred_accuracy)
axa.set_title("Training accuracy for each batch")
axa.legend(["Training", "Validation"])
axa.set_xlabel("Batch")
axa.set_ylabel("Accuracy")
_,ymax = plt.ylim()
_,xmax = plt.xlim()
epoch_brk = len(train_accuracy)/n_epochs
line_loc = 0
for e in range(n_epochs):
    axa.axvline(x = line_loc, color = "k")
    axa.annotate("E{}".format(e + 1), xy = (7+ line_loc/11.75,ymax/4),\
        textcoords="offset points")
    line_loc += epoch_brk
plt.show()
figa.savefig("nnmodel_accuracy.png")
#%%
#predict on new data, just one batch, 128 images, take 3rd iteration
i = 0
for test_idx, (feats, targets) in enumerate(testloader):
    model.eval()
    if mps_on:
            feats, targets = feats.to(devicemps), targets.to(devicemps)
    testout = model(feats)
    test_acc = accuracy(testout, targets)
    _, pred = torch.max(testout, dim=1)
    pred = pred
    actuals = targets
    imgs = feats.cpu().numpy()
    i += 1
    if i > 1:
        break
#%%
#plot test images labeled by predicted class and actual class
figp = plt.figure(figsize=(25,6))
for idx in np.arange(20):
    ax = figp.add_subplot(2, 10, idx+1, xticks = [], yticks = [])
    showimg(imgs[idx])
    ax.set_title("Predicted: " + classes[pred[idx]] + "\nActual: " + classes[actuals[idx]])
figp.savefig("test_images_pred_actual.png")
#%%
