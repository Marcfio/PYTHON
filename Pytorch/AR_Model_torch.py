# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 18:07:44 2021

@author: MARCOFIORAVANTIPC
"""


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# make the original data
N = 1000
series = np.sin(0.1*np.arange(N)) # + np.random.randn(N)*0.1

# plot it
plt.plot(series)
plt.show()

## build the dataset
# Let's see if we can use T past values to predict the next value

T = 10


# build the dataset

T = 10
X = []
Y = []
for t in range (len(series) - T):
    x = series[t:t + T]
    X.append(x)
    y = series[t+T]
    Y.append(y)

X = np.array(X).reshape(-1,T)
Y = np.array(Y).reshape(-1,1)
N = len(X)
print("X.shape",X.shape,"Y.shape", Y.shape)

## try autoregressive linear model
model = nn.Linear(T,1)

#Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

#Make inputs and targets
X_train = torch.from_numpy(X[:-N//2].astype(np.float32))
y_train = torch.from_numpy(Y[:-N//2].astype(np.float32))
X_test = torch.from_numpy(X[:-N//2].astype(np.float32))
y_test = torch.from_numpy(Y[:-N//2].astype(np.float32))

# Training
def full_gd(model,
            criterion,
            optimizer,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs = 200):
    
    # stuff to store
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    
    for it in range(epochs):
        #zero the parameters gradients
        optimizer.zero_grad()
        
        #Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        #Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Save losses
        train_losses[it] = loss.item()
        
        #Test loss
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses[it] = test_loss.item()
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
                
        if (it+1)% 5 == 0:
            print(f'Epoch {it+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
        
    return train_losses, test_losses

train_losses, test_losses = full_gd(model,
            criterion,
            optimizer,
            X_train,
            y_train,
            X_test,
            y_test)
    


## plot the train loss and test loss per iteration
plt.plot(train_losses, label = 'train loss')
plt.plot(test_losses, label = 'test loss')
plt.legend()
plt.show()

# Wrong forecast using true targets
validation_target = Y[-N//2:]
validation_predictions = []

# index of first validation input
 
i = 0

while len(validation_predictions)< len(validation_target):
    input_ = X_test[i].view(1,-1)
    p = model(input_)[0,0].item() # 1x1 array -> scalar
    i += 1
    #update the prediction list
    validation_predictions.append(p)

plt.plot(validation_target, label = 'forecast target')
plt.plot(validation_predictions, label = 'forecast prediction')
plt.legend()

# Forecast future values (use only self-predictions for making future prediction)

validation_target = Y[-N//2:]
validation_prediction = []

# last train input
# 1-D array of length T

last_x = torch.from_numpy(X[-N//2].astype(np.float32))

while len(validation_predictions) < len(validation_target):
    input_ = last_x.view(1, -1)
    p = model(input_)
    
    # [0,0] # 1x1 array -> scalar
    
    # update the predictions list
    validation_prediction.append(p[0,0].item())
    
    # make the new input
    last_x = torch.cat((last_x[1:], p[0]))


plt.plot(validation_target, label = 'forecast target')
plt.plot(validation_predictions, label = 'forecast prediction')
plt.legend()







