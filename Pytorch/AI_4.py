# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 20:32:29 2021

@author: MARCOFIORAVANTIPC
"""

#########study sigmoid function and review the video
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

#Load data
data = load_breast_cancer()

#check the type of data
type(data)
#is a Bunch object --> dictionary where you can treat the keys like attributes---> data.attribute##
data.keys()
#'data' (the attribute) means the input data
data.data.shape

data.target
data.target_names

data.target.shape
data.feature_names

## normally we would put all of our imports at the top
## but this lets us tell a story
from sklearn.model_selection import train_test_split

# split the data into train and test sets
# this lets us simulate how our model will perform in the future

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=5)

N, D = X_train.shape

## Scale the data
## you'll learn why scaling is needed in a later course

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Now all the fun of pytorch stuff 
# Build the model

model = nn.Linear(D,1)

#Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

#Convert data into torch tensor
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32).reshape(-1,1))
y_test = torch.from_numpy(y_test.astype(np.float32).reshape(-1,1))

#train the model
n_epochs = 10000

#Stuff to store
train_losses = np.zeros(n_epochs)
test_losses = np.zeros(n_epochs)
train_acc = np.zeros(n_epochs)
test_acc = np.zeros(n_epochs)

for it in range(n_epochs):
    #zero the parameter gradients
    optimizer.zero_grad()
    
    #Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    #Backward and optimize
    loss.backward()
    optimizer.step()
    
    #Get the loss
    outputs_test = model(X_test)
    loss_test = criterion(outputs_test, y_test)
    
    #Save losses
    train_losses[it] = loss.item()
    test_losses[it] = loss_test.item()
        
    if(it+1)% 50 == 0:
        print(f'Epoch {it+1}/{n_epochs}, Train Loss: {loss.item():.4f}, Test Loss: {loss_test.item():.4f}')


plt.plot(train_losses, label= 'train loss')
plt.plot(test_losses, label = 'test loss')
plt.legend()
plt.show()


# Get accuracy
with torch.no_grad():
    p_train = model(X_train)
    p_train = (p_train.numpy() > 0)
    train_acc = np.mean(y_train.numpy() == p_train)
    
    p_test = model(X_test)
    p_test = (p_test.numpy() > 0)
    test_acc = np.mean(y_test.numpy() == p_test)

print(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")
    
    