# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 22:23:58 2021

@author: MARCOFIORAVANTIPC
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
    )

train_dataset.data.max()

train_dataset.data.shape 

train_dataset.targets = np

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transforms.ToTensor(),
    download=True
    )


# number of classes
K = len(set(train_dataset.targets.numpy()))

print("number of classes:", K)

# Define the model

class CNN(nn.Module):
    def __init__(self,K):
        super(CNN,self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channles=32, out_channels=64, kernel_size = 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size = 3, stride = 2),
            nn.ReLU(),
            )
        self.dense_layers = nn.Sequential(
             nn.Dropout(0.2),
             nn.Linear(128*2*2,512),
             nn.ReLU(),
             nn.Dropout(0.2),
             nn.Linear(512,K)
               )
    def forward(self,X):
        out = self.conv_layers(X)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out
    
    
# %% instatiate the model
model = CNN(K)


# The same model using "Flatten"
# model = nn.Sequential(
#         nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride = 2),
#         nn.ReLU(),
#         nn.Conv2d(in_channels=32, out_channels=64, kernel_size = 3, stride = 2),
#         nn.ReLU(),
#         nn.Conv2d(in_channels=64, out_channels= 128, kernel_size = 3, stride = 2),
#         nn.ReLU(),
#         nn.Flatten(),
#         nn.Dropout(0.2),
#         nn.Linear(128*2*2, 512),
#         nn.ReLU(),
#         nn.Dropout(0.2),
#         nn.Linear(512,K)
        
#         )
    
    
    
# Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = nn.torch.optim.Adam(model.parameters())


# Data loader
# Useful because it automatically generates batches in the training loop
# and takes care of shuffling

batch_size = 128

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True                                     
                                           )

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = False          
                                                                           
                                          )

# A function to encapsulate the train loop
def batch_gd(model,criterion,optimizer, train_loader, test_loader, epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zerso(epochs)
    
    for it in range(epochs):
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            #move data to GPU
            inputs, targets = inputs.to(device), targets.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            #Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
            
        test_loss = np.mean(test_loss)
        
        #Save losses
        train_losses[it]= train_loss
        test_losses[it]= test_loss
        
        dt = datetime.now() - t0
        print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \ 
              Test Loss: {test_loss:.4f},Duration:{dt}')
              
        return train_losses, test_losses
     
train_losses, test_losses = batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs = 15)


# %% Plot the train loss and test loss per iteration

plt.plot(train_losses, label = 'train loss')
plt.plot(test_losses, label  = 'test loss')
plt.legend()
plt.show()

# %% Accuracy

n_correct = 0.
n_total = 0.
for inputs, targets in train_loader:
    #move data to GPU
    inputs, targets = inputs.to(device),targets.to(device)
    
    #Forward pass
    outputs = model(inputs)
    
    #Get predicition
    #torch.max returns both max and argmax
    _, predictions = torch.max(outputs, 1)
    
    # update counts
    n_correct += (prediction == targets).sum().item()
    n_total += targets.shape[0]
    
              
            
test_acc = n_correct / n_total
print(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")
        
        
# confusion matrix
from sklearn.metrics import confusion_matrix
import intertools

def plot_confusion_matrix(cm, classes, normalize = False, title = 'Cofusion matrix', cmaap = plt.cm.Blues):
    
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize = True.
    
    """
    
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print("Normalized confusion matrix")
        
    else:
        print('Confusion matrix, without normalization')
        
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colobar()
    tick_marks = np.arange(len(classes))
    plt.xtick(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in intertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i, format(cm[i,j],fmt),
                horizontalignment = "center",
                
    






