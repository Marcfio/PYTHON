# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 21:31:48 2021

@author: MARCOFIORAVANTIPC
"""
########################## PYTORCH FOR REGRESSION

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# make the dataset
N = 1000
X = np.random.random((N,2))*6-3
Y = np.cos(2*X[:,0])+np.cos(3*X[:,1]) #y = cos(2x_1)+ cos(3x_2)

# Plot it
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:,0], X[:,1],Y)
# plt.show()

# Build the model
model = nn.Sequential( 
    nn.Linear(2,128),
    nn.ReLU(), #activation function
    nn.Linear(128,1)    
    )

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

#Train the model
def full_gd(model,criterion,optimizer, X_train, ytrain, epochs = 1000):
    train_losses= np.zeros(epochs)
    
    for it in range(epochs):
        # zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        #Backward and optimize
        loss.backward()
        optimizer.step()
        
        #Save losses
        train_losses[it] = loss.item()
        
        if (it+1)%50 == 0:
            print(f'Epoch {it+1}/{epochs}, Train Loss: {loss.item():.4f}')
            
    return train_losses


X_train = torch.from_numpy(X.astype(np.float32))
y_train = torch.from_numpy(Y.astype(np.float32).reshape(-1,1))
train_losses = full_gd(model,criterion,optimizer,X_train,y_train)
        
plt.plot(train_losses)

#Plot the prediction surface
fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
ax.scatter(X[:,0], X[:,1],Y)

# surface plot
with torch.no_grad():
    line = np.linspace(-3,3,50)
    xx, yy = np.meshgrid(line,line)
    Xgrid = np.vstack((xx.flatten(),yy.flatten())).T
    Xgrid_torch= torch.from_numpy(Xgrid.astype(np.float32))
    Yhat = model(Xgrid_torch).numpy().flatten()
    ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth = 0.2, antialiased = True)
    plt.show()
    
    